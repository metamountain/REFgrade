"""K31a: /preview mit timing:true liefert Teilzeiten; ohne timing unverändert."""

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

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
PYTHON = sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 30) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _jpg_to_raw(jpg_path: str, tmp: str) -> tuple[str, int, int]:
    from PIL import Image
    img = Image.open(jpg_path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)
    bgra = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0],
                     np.full((h, w), 255, dtype=np.uint8)], axis=-1)
    raw_path = os.path.join(tmp, "test_src.raw")
    bgra.tofile(raw_path)
    return raw_path, w, h


EXPECTED_TIMING_KEYS = {
    "lut_method_fn", "lut_section_fns", "lut_build", "lut_total",
    "lut_apply", "spatial", "histograms", "png_write", "scopes",
    "scope_save", "json",
}


class TestLatencyTiming(unittest.TestCase):
    """Worker /preview timing-Feld."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp(prefix="refgrade_timing_")
        cls.proc = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(cls.port), "--tmp", cls.tmp],
            cwd=str(CORE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise RuntimeError(f"Worker beendet: {err.decode()}")
            try:
                r = _get(f"{cls.base}/health")
                if r.get("ok"):
                    break
            except Exception:
                time.sleep(0.3)
        else:
            cls.proc.kill()
            raise RuntimeError("Worker nicht erreichbar")

        jpg = str(POOL_DIR / "04.jpg")
        if not Path(jpg).exists():
            jpg = str(POOL_DIR / "01.jpg")
        raw, w, h = _jpg_to_raw(jpg, cls.tmp)
        r = _post(f"{cls.base}/load", {"raw": raw, "width": w, "height": h, "depth": "U8"})
        cls.sid = r["session"]
        cls.ref = str(POOL_DIR / "02.jpg")
        if not Path(cls.ref).exists():
            cls.ref = str(POOL_DIR / "01.jpg")

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            _post(f"{cls.base}/shutdown", {})
        except Exception:
            pass
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
            cls.proc.wait()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _base_body(self) -> dict:
        return {
            "session": self.sid,
            "references": [self.ref],
            "method": "reinhard",
            "values": {"exposure": 0.3},
            "max_side": 512,
            "mix": {"weights": {}, "master": 0.5},
            "stage": "lut",
        }

    def test_timing_present_with_all_keys(self) -> None:
        """timing:true liefert timing mit allen erwarteten Schlüsseln, jeder ≥ 0."""
        body = self._base_body()
        body["timing"] = True
        resp = _post(f"{self.base}/preview", body)
        self.assertIn("timing", resp)
        timing = resp["timing"]
        for key in EXPECTED_TIMING_KEYS:
            self.assertIn(key, timing, f"timing fehlt Schlüssel: {key}")
            self.assertGreaterEqual(timing[key], 0.0, f"{key} ist negativ")

    def test_timing_sum_le_total(self) -> None:
        """Summe der Teilzeiten ≤ Gesamtzeit + 5 ms."""
        body = self._base_body()
        body["timing"] = True
        resp = _post(f"{self.base}/preview", body)
        timing = resp["timing"]
        total = sum(timing.values())
        self.assertLessEqual(total, resp["ms"] + 5.0,
                             f"Summe {total:.1f} ms > Gesamt {resp['ms']:.1f} ms + 5")

    def test_no_timing_key_without_flag(self) -> None:
        """Ohne timing:true ist das timing-Feld nicht vorhanden."""
        body = self._base_body()
        resp = _post(f"{self.base}/preview", body)
        self.assertNotIn("timing", resp)

    def test_response_keys_unchanged_without_timing(self) -> None:
        """Ohne timing sind die Antwort-Schlüssel gleich wie vorher."""
        body = self._base_body()
        resp = _post(f"{self.base}/preview", body)
        expected_keys = {"image", "histogram", "ms", "vectorscope",
                         "vectorscope_source", "parade", "wave", "clip"}
        self.assertTrue(expected_keys.issubset(set(resp.keys())),
                        f"Fehlende Schlüssel: {expected_keys - set(resp.keys())}")


if __name__ == "__main__":
    unittest.main()
