"""Tests für rawio: Rohdaten-I/O, Maske, Bild laden, PNG speichern."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch

from refgrade import rawio


class TestReadRaw(unittest.TestCase):
    """read_raw: Bytes → Tensor + Alpha."""

    def test_single_pixel_u8(self):
        """Ein Pixel (10,20,30,255) als U8 → rgb = (30,20,10)/255 ±1e-7."""
        # Bytes: B=10, G=20, R=30, A=255
        data = bytes([10, 20, 30, 255])
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            rgb, alpha = rawio.read_raw(path, 1, 1, "U8")
            self.assertEqual(rgb.shape, (1, 1, 1, 3))
            expected = torch.tensor([[[[30/255, 20/255, 10/255]]]])
            self.assertTrue(torch.allclose(rgb, expected, atol=1e-7))
            self.assertEqual(alpha.shape, (1, 1))
            self.assertEqual(alpha[0, 0], 255)
        finally:
            os.unlink(path)

    def test_roundtrip_u8(self):
        """Rundreise 7×5 Zufallsbytes (Seed 0) U8: bytegleich."""
        rng = np.random.default_rng(0)
        h, w = 5, 7
        rgba = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            rgb, alpha = rawio.read_raw(path, w, h, "U8")
            rawio.write_raw(path, rgb, alpha, "U8")
            with open(path, "rb") as f:
                result = f.read()
            self.assertEqual(result, rgba.tobytes())
        finally:
            os.unlink(path)

    def test_roundtrip_u16(self):
        """Rundreise 7×5 Zufallsbytes (Seed 0) U16: bytegleich."""
        rng = np.random.default_rng(0)
        h, w = 5, 7
        rgba = rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            rgb, alpha = rawio.read_raw(path, w, h, "U16")
            rawio.write_raw(path, rgb, alpha, "U16")
            with open(path, "rb") as f:
                result = f.read()
            self.assertEqual(result, rgba.tobytes())
        finally:
            os.unlink(path)

    def test_wrong_length(self):
        """Falsche Dateilänge → ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(b"\x00" * 10)  # zu kurz für 4×4 U8 (erwartet 64)
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                rawio.read_raw(path, 4, 4, "U8")
            self.assertIn("64", str(ctx.exception))
            self.assertIn("10", str(ctx.exception))
        finally:
            os.unlink(path)


class TestReadMask(unittest.TestCase):
    """read_mask: 1 Byte/Pixel → [H,W] float32 0..1."""

    def test_basic(self):
        h, w = 3, 4
        data = np.array([0, 128, 255, 64, 32, 16, 8, 4, 2, 1, 0, 255], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            f.write(data.tobytes())
            path = f.name
        try:
            mask = rawio.read_mask(path, w, h)
            self.assertEqual(mask.shape, (h, w))
            self.assertAlmostEqual(mask[0, 0].item(), 0.0, places=5)
            self.assertAlmostEqual(mask[0, 2].item(), 1.0, places=5)
        finally:
            os.unlink(path)


class TestLoadImage(unittest.TestCase):
    """load_image: PIL → Tensor."""

    def test_small_image(self):
        """Kleines Bild (100×50) wird nicht vergrößert."""
        from PIL import Image
        img = Image.new("RGB", (100, 50), color=(128, 64, 32))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            path = f.name
        try:
            tensor = rawio.load_image(path, max_side=1536)
            self.assertEqual(tensor.shape, (1, 50, 100, 3))
            self.assertEqual(tensor.dtype, torch.float32)
        finally:
            os.unlink(path)

    def test_resize(self):
        """Großes Bild wird auf max_side verkleinert."""
        from PIL import Image
        img = Image.new("RGB", (2000, 1000), color=(128, 64, 32))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            path = f.name
        try:
            tensor = rawio.load_image(path, max_side=500)
            h, w = tensor.shape[1], tensor.shape[2]
            self.assertLessEqual(max(h, w), 500)
        finally:
            os.unlink(path)


class TestSavePng(unittest.TestCase):
    """save_png: Tensor → 8-Bit-PNG."""

    def test_roundtrip(self):
        from PIL import Image
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, size=(10, 20, 3), dtype=np.uint8)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            rawio.save_png(tensor, path)
            img = Image.open(path)
            self.assertEqual(img.size, (20, 10))
            self.assertEqual(img.mode, "RGB")
            result = np.array(img)
            self.assertTrue(np.array_equal(result, arr))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
