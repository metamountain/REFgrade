"""Tests für plugin/ref_grade/pixel_rules.py und Ende-zu-Ende Pixelbrücke."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import pixel_rules  # noqa: E402
from ref_grade import config  # noqa: E402
from ref_grade import worker_process  # noqa: E402

CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 60) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CheckTest(unittest.TestCase):
    """pixel_rules.check."""

    def test_rgba_u8_srgb(self):
        self.assertIsNone(pixel_rules.check("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc"))

    def test_rgba_u16_srgb_builtin(self):
        self.assertIsNone(pixel_rules.check("RGBA", "U16", "sRGB built-in"))

    def test_rgba_f32(self):
        result = pixel_rules.check("RGBA", "F32", "sRGB-elle-V2-srgbtrc.icc")
        self.assertIsNotNone(result)
        self.assertIn("F32", result)

    def test_cmyka_u8(self):
        result = pixel_rules.check("CMYKA", "U8", "sRGB-elle-V2-srgbtrc.icc")
        self.assertIsNotNone(result)
        self.assertIn("CMYKA", result)

    def test_rgba_u8_rec2020(self):
        result = pixel_rules.check("RGBA", "U8", "Rec2020")
        self.assertIsNotNone(result)
        self.assertIn("Rec2020", result)


class LayerNameTest(unittest.TestCase):
    """pixel_rules.layer_name."""

    def test_simple(self):
        self.assertEqual(pixel_rules.layer_name("Haus G.jpg"), "REF · Haus G")

    def test_long_name(self):
        name = "x" * 100 + ".png"
        result = pixel_rules.layer_name(name)
        self.assertLessEqual(len(result), 66)

    def test_no_extension(self):
        self.assertEqual(pixel_rules.layer_name("photo"), "REF · photo")


class ByteCountTest(unittest.TestCase):
    """pixel_rules.byte_count."""

    def test_u16(self):
        self.assertEqual(pixel_rules.byte_count(3, 2, "U16"), 48)

    def test_u8(self):
        self.assertEqual(pixel_rules.byte_count(4, 3, "U8"), 48)


class EndToEndTest(unittest.TestCase):
    """Ende-zu-Ende: 4×3 BGRA-U16 → Worker /load + /apply."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        python = config.python_exe()
        if python is None:
            raise unittest.SkipTest("Kein Python-Interpreter gefunden")
        worker_process.ensure(cls.port, python, CORE_DIR, timeout=30.0)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k4_"))

        # 4×3 BGRA U16 Muster: B=10, G=20, R=30, A=255 (U8) → U16 skaliert
        w, h = 4, 3
        arr = np.zeros((h, w, 4), dtype=np.uint16)
        arr[..., 0] = 10 * 257   # B
        arr[..., 1] = 20 * 257   # G
        arr[..., 2] = 30 * 257   # R
        arr[..., 3] = 65535      # A = 255*257 = 65535
        cls.source_bytes = arr.tobytes()
        source_path = cls.tmp / "source.raw"
        source_path.write_bytes(cls.source_bytes)
        cls.source_path = source_path
        cls.w, cls.h = w, h

    @classmethod
    def tearDownClass(cls):
        worker_process.stop(cls.port)
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self) -> str:
        r = _post(f"http://127.0.0.1:{self.port}/load", {
            "raw": str(self.source_path),
            "width": self.w,
            "height": self.h,
            "depth": "U16",
        })
        return r["session"]

    def _apply(self, sid: str, values: dict, out_name: str) -> Path:
        out_path = self.tmp / out_name
        ref = str(POOL_DIR / "01.jpg")
        _post(f"http://127.0.0.1:{self.port}/apply", {
            "session": sid,
            "references": [ref],
            "method": "reinhard",
            "values": values,
            "out": str(out_path),
        })
        return out_path

    def test_identity(self):
        """Stärke 0 → bytegleich zur Quelle."""
        sid = self._load()
        out = self._apply(sid, {"light_strength": 0.0, "color_strength": 0.0}, "identity.raw")
        result_bytes = out.read_bytes()
        self.assertEqual(result_bytes, self.source_bytes)

    def test_strength1_alpha(self):
        """Stärke 1 → gleiche Länge, Alphakanal bytegleich."""
        sid = self._load()
        out = self._apply(sid, {}, "graded.raw")
        result_bytes = out.read_bytes()
        self.assertEqual(len(result_bytes), len(self.source_bytes))
        # Alpha-Kanal: letzter Kanal (Index 3), U16 little-endian
        src_arr = np.frombuffer(self.source_bytes, dtype=np.uint16).reshape(self.h, self.w, 4)
        res_arr = np.frombuffer(result_bytes, dtype=np.uint16).reshape(self.h, self.w, 4)
        self.assertEqual(res_arr[..., 3].tobytes(), src_arr[..., 3].tobytes())


if __name__ == "__main__":
    unittest.main()
