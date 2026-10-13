"""Regressions-Tests für Paket 78 (combo_stress).

Fester Sollwert: Worker-/preview mit exposure 2 weicht im Mittel ≥ 0,1 vom Standard ab.
"""
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
from PIL import Image

CORE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORE_DIR))

import torch
import refgrade.session as sess
from refgrade.sliders import SLIDERS

POOL = CORE_DIR.parent / "bench" / "pool"
PYTHON = sys.executable


def _jpg_to_raw(jpg: str, raw: str) -> tuple[int, int]:
    img = Image.open(jpg).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint16)
    bgra = np.zeros((h, w, 4), dtype=np.uint16)
    bgra[:, :, 0] = arr[:, :, 2]
    bgra[:, :, 1] = arr[:, :, 1]
    bgra[:, :, 2] = arr[:, :, 0]
    bgra[:, :, 3] = 65535
    with open(raw, "wb") as f:
        f.write(bgra.tobytes())
    return w, h


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 60) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 30) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _img_diff(path_a: str, path_b: str) -> float:
    a = np.array(Image.open(path_a).convert("RGB"), dtype=np.float64) / 255.0
    b = np.array(Image.open(path_b).convert("RGB"), dtype=np.float64) / 255.0
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


class TestComboRegressionsDirect(unittest.TestCase):
    """Direkte Tests über session.preview (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test_combo_reg_")
        jpg01 = POOL / "01.jpg"
        if not jpg01.exists():
            raise unittest.SkipTest(f"{jpg01} fehlt")
        raw = os.path.join(cls.tmp, "01.raw")
        w, h = _jpg_to_raw(str(jpg01), raw)
        cls.session = sess.open_session(raw, w, h, "U16", device=torch.device("cpu"))
        cls.refs = [str(POOL / "02.jpg")]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exposure_2_diff_ge_0_1(self):
        """Fester Sollwert: exposure=2 weicht im Mittel ≥ 0,1 vom Standard ab."""
        defaults = {s.name: s.default for s in SLIDERS}
        p_base, _ = sess.preview(self.session, self.refs, "reinhard", defaults, 512)
        p_exp, _ = sess.preview(self.session, self.refs, "reinhard",
                                {**defaults, "exposure": 2.0}, 512)
        diff = float((p_base - p_exp).abs().mean())
        self.assertGreaterEqual(diff, 0.1,
                                f"exposure=2: diff={diff:.6f} < 0.1 (Sollwert)")

    def test_determinism(self):
        """Determinismus: dieselbe Anfrage zweimal → identisch."""
        vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3}
        p1, _ = sess.preview(self.session, self.refs, "reinhard", vals, 512)
        p2, _ = sess.preview(self.session, self.refs, "reinhard", vals, 512)
        diff = float((p1 - p2).abs().max())
        self.assertEqual(diff, 0.0, f"Determinismus verletzt: max_diff={diff}")


class TestComboRegressionsWorker(unittest.TestCase):
    """Worker-basierte Tests: HTTP, BMP-Roundtrip, values, mix.

    Prüft den vollständigen Worker-Pfad, über den das Plugin kommuniziert.
    """

    @classmethod
    def setUpClass(cls):
        jpg01 = POOL / "01.jpg"
        if not jpg01.exists():
            raise unittest.SkipTest(f"{jpg01} fehlt")
        cls.tmp = tempfile.mkdtemp(prefix="test_combo_worker_")
        cls.port = _free_port()
        cls.proc = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(cls.port), "--tmp", cls.tmp],
            cwd=str(CORE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        # Warten auf Health
        deadline = time.time() + 30
        healthy = False
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise RuntimeError(f"Worker beendet: {err.decode()[:300]}")
            try:
                r = _get(f"{cls.base}/health")
                if r.get("ok"):
                    healthy = True
                    break
            except Exception:
                time.sleep(0.3)
        if not healthy:
            cls.proc.kill()
            raise RuntimeError("Worker nicht erreichbar")
        # Bild laden
        raw = os.path.join(cls.tmp, "01.raw")
        w, h = _jpg_to_raw(str(jpg01), raw)
        r = _post(f"{cls.base}/load", {"raw": raw, "width": w, "height": h, "depth": "U16"})
        cls.sid = r["session"]
        cls.refs = [str(POOL / "02.jpg")]

    @classmethod
    def tearDownClass(cls):
        try:
            _post(f"{cls.base}/shutdown", {})
        except Exception:
            pass
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_worker_exposure_2_diff_ge_0_1(self):
        """Fester Sollwert über Worker: exposure=2 → diff ≥ 0,1 (BMP-Roundtrip)."""
        defaults = {s.name: s.default for s in SLIDERS}
        r_base = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": defaults, "max_side": 512,
        })
        r_exp = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": {**defaults, "exposure": 2.0}, "max_side": 512,
        })
        diff = _img_diff(r_base["image"], r_exp["image"])
        self.assertGreaterEqual(diff, 0.1,
                                f"Worker exposure=2: diff={diff:.6f} < 0.1 (Sollwert)")

    def test_worker_determinism(self):
        """Worker-Determinismus: dieselbe Anfrage zweimal → bytegleich."""
        vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3}
        r1 = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": vals, "max_side": 512,
        })
        r2 = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": vals, "max_side": 512,
        })
        with open(r1["image"], "rb") as f1:
            b1 = f1.read()
        with open(r2["image"], "rb") as f2:
            b2 = f2.read()
        self.assertEqual(b1, b2, "Worker-Determinismus verletzt: BMPs unterscheiden sich")

    def test_worker_master0_identity(self):
        """Worker: master=0 mit extremen Werten ≈ master=0 mit Defaults (Identität).

        Beide Configs sind Identität (is_identity → True). Kleine Abweichung
        durch BMP-Rounding toleriert (≤ 1/255).
        """
        defaults = {s.name: s.default for s in SLIDERS}
        extreme = dict(defaults)
        for s in SLIDERS:
            extreme[s.name] = s.max if s.default == s.min else s.min
        mix_zero = {"weights": {k: 1.0 for k in [
            "reference", "tone", "color", "wheels", "lut", "curves"
        ]}, "master": 0.0, "solo": None}
        r_ext = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": extreme, "max_side": 512, "mix": mix_zero,
        })
        r_def = _post(f"{self.base}/preview", {
            "session": self.sid, "references": self.refs, "method": "reinhard",
            "values": defaults, "max_side": 512, "mix": mix_zero,
        })
        diff = _img_diff(r_ext["image"], r_def["image"])
        self.assertLessEqual(diff, 1.0 / 255.0,
                             f"master=0: extreme vs defaults diff={diff:.8f} (> 1/255)")


if __name__ == "__main__":
    unittest.main()
