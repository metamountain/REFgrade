"""Tests für clip_overlay (scope.py) und clip_view (worker /preview)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import torch
from PIL import Image

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
PYTHON = sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 60) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 60) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestClipOverlay(unittest.TestCase):
    """Reine Funktion clip_overlay aus scope.py."""

    def test_2x2(self):
        from refgrade.scope import clip_overlay
        # 2×2-Bild: [1.0,0.5,0.5] [0.0,0.5,0.5] / [0.5,0.5,0.5] [1.0,0.0,0.5]
        x = torch.tensor([[[[1.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
                           [[0.5, 0.5, 0.5], [1.0, 0.0, 0.5]]]])
        out = clip_overlay(x)
        self.assertEqual(out.shape, x.shape)
        # Pixel (0,0): R=1.0 >= _HI_THRESH → rot
        self.assertEqual(out[0, 0, 0].tolist(), [1.0, 0.0, 0.0])
        # Pixel (0,1): R=0.0 <= _LO_THRESH → blau
        self.assertEqual(out[0, 0, 1].tolist(), [0.0, 0.0, 1.0])
        # Pixel (1,0): alle 0.5 → unverändert
        self.assertEqual(out[0, 1, 0].tolist(), [0.5, 0.5, 0.5])
        # Pixel (1,1): R=1.0 >= _HI_THRESH → rot
        self.assertEqual(out[0, 1, 1].tolist(), [1.0, 0.0, 0.0])


class TestClipViewWorker(unittest.TestCase):
    """Worker /preview mit clip_view (echter Subprozess)."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp(prefix="refgrade_clip_")
        env = os.environ.copy()
        cls.proc = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(cls.port), "--tmp", cls.tmp],
            cwd=str(CORE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise RuntimeError(f"Worker sofort beendet: {err.decode()}")
            try:
                r = _get(f"{cls.base}/health")
                if r.get("ok"):
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("Worker nicht innerhalb von 20 s erreichbar")

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            try:
                _post(f"{cls.base}/shutdown", {})
            except Exception:
                pass
            try:
                cls.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self, width: int, height: int, depth: str = "U16") -> str:
        arr = np.random.randint(0, 65536, (height, width, 4), dtype=np.uint16)
        path = os.path.join(self.tmp, f"raw_{width}x{height}_{depth}.raw")
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        r = _post(f"{self.base}/load", {"raw": path, "width": width, "height": height, "depth": depth})
        return r["session"]

    def test_clip_view_exposure(self):
        """Exposure +3: mit clip_view=true enthält das Bild Pixel (255,0,0), ohne nicht."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        values = {"exposure": 3.0}

        r_on = _post(f"{self.base}/preview", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": values, "max_side": 64, "clip_view": True,
        })
        r_off = _post(f"{self.base}/preview", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": values, "max_side": 64, "clip_view": False,
        })

        img_on = np.array(Image.open(r_on["image"]).convert("RGB"))
        img_off = np.array(Image.open(r_off["image"]).convert("RGB"))

        # Mit clip_view: mindestens ein Pixel ist exakt (255, 0, 0)
        red_mask = (img_on[:, :, 0] == 255) & (img_on[:, :, 1] == 0) & (img_on[:, :, 2] == 0)
        self.assertGreater(red_mask.sum(), 0, "clip_view=true: erwartet rote Pixel (255,0,0)")

        # Ohne clip_view: kein Pixel ist exakt (255, 0, 0)
        red_mask_off = (img_off[:, :, 0] == 255) & (img_off[:, :, 1] == 0) & (img_off[:, :, 2] == 0)
        self.assertEqual(red_mask_off.sum(), 0, "clip_view=false: keine roten Pixel erwartet")

    def test_clip_histogram_unchanged(self):
        """clip und histogram sind mit und ohne clip_view identisch."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        values = {"exposure": 3.0}

        r_on = _post(f"{self.base}/preview", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": values, "max_side": 64, "clip_view": True,
        })
        r_off = _post(f"{self.base}/preview", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": values, "max_side": 64, "clip_view": False,
        })

        self.assertEqual(r_on["clip"], r_off["clip"])
        self.assertEqual(r_on["histogram"], r_off["histogram"])


if __name__ == "__main__":
    unittest.main()
