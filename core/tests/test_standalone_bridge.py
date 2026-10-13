"""Tests für standalone/image_bridge.py: Bilddatei/QImage → Rohdaten → Bilddatei."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Projekt-Root (standalone/) in den Pfad legen; core/ ist bereits dabei (unittest -m).
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from standalone import image_bridge  # noqa: E402

try:
    from PyQt6.QtGui import QImage

    HAS_QIMAGE = True
except Exception:  # pragma: no cover
    HAS_QIMAGE = False


class ImageBridgeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="refgrade_s1_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_16bit_png_umlaut_roundtrip(self):
        """16-Bit-PNG mit Umlaut im Namen → read_image → write_image → bytegleich."""
        rng = np.random.default_rng(0)
        h, w = 4, 5
        bgr = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
        src = self.tmp / "äöü_16bit.png"
        ok, buf = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        np.asarray(buf).tofile(str(src))

        source = image_bridge.read_image(str(src), self.tmp)
        self.assertEqual(source["depth"], "U16")
        self.assertEqual(source["width"], w)
        self.assertEqual(source["height"], h)
        self.assertEqual(source["x"], 0)
        self.assertEqual(source["y"], 0)

        expected_bgra = np.empty((h, w, 4), dtype=np.uint16)
        expected_bgra[..., :3] = bgr
        expected_bgra[..., 3] = 65535
        self.assertEqual(Path(source["raw"]).read_bytes(), expected_bgra.tobytes())

        out = self.tmp / "äöü_ergebnis.png"
        image_bridge.write_image(source, Path(source["raw"]), out)
        result = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(result.dtype, np.uint16)
        self.assertTrue(np.array_equal(result, expected_bgra), "nicht bytegleich")

    def test_8bit_jpg_alpha_u8(self):
        """8-Bit-JPG → read_image: Alpha 255, Tiefe U8."""
        rng = np.random.default_rng(1)
        h, w = 3, 4
        bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        src = self.tmp / "bild.jpg"
        ok, buf = cv2.imencode(".jpg", bgr)
        self.assertTrue(ok)
        np.asarray(buf).tofile(str(src))

        source = image_bridge.read_image(str(src), self.tmp)
        self.assertEqual(source["depth"], "U8")
        self.assertEqual(source["width"], w)
        self.assertEqual(source["height"], h)
        raw = np.fromfile(source["raw"], np.uint8).reshape(h, w, 4)
        self.assertTrue(np.all(raw[..., 3] == 255), "Alpha nicht 255")

    @unittest.skipUnless(HAS_QIMAGE, "PyQt6/QtGui nicht verfügbar")
    def test_qimage_3x2_bgra(self):
        """QImage 3×2 → Rohbytes in B,G,R,A-Reihenfolge, Tiefe U8."""
        img = QImage(3, 2, QImage.Format.Format_ARGB32)
        img.setPixel(0, 0, 0xFF112233)  # B=0x33 G=0x22 R=0x11 A=0xFF
        img.setPixel(1, 0, 0xFF445566)  # B=0x66 G=0x55 R=0x44 A=0xFF
        img.setPixel(2, 1, 0x80AABBCC)  # B=0xCC G=0xBB R=0xAA A=0x80

        source = image_bridge.read_qimage(img, self.tmp)
        self.assertEqual(source["depth"], "U8")
        self.assertEqual(source["width"], 3)
        self.assertEqual(source["height"], 2)

        raw = np.fromfile(source["raw"], np.uint8).reshape(2, 3, 4)
        self.assertEqual(raw[0, 0].tolist(), [0x33, 0x22, 0x11, 0xFF])
        self.assertEqual(raw[0, 1].tolist(), [0x66, 0x55, 0x44, 0xFF])
        self.assertEqual(raw[1, 2].tolist(), [0xCC, 0xBB, 0xAA, 0x80])

    def test_wrong_raw_length_valueerror(self):
        """Falsche Rohlänge → ValueError."""
        source = {"width": 4, "height": 4, "depth": "U8"}  # erwartet 64 Bytes
        bad = self.tmp / "bad.raw"
        bad.write_bytes(b"\x00" * 10)
        with self.assertRaises(ValueError):
            image_bridge.write_image(source, bad, self.tmp / "out.png")


if __name__ == "__main__":
    unittest.main()
