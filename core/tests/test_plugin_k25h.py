"""Tests für Paket K25h: Querformat (H) — Bild im Seitenverhältnis oben, Band darunter."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ref_grade import scale  # noqa: E402


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True if two (x, y, w, h) rects overlap (area > 0)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class TestHGeometryWideImage(unittest.TestCase):
    """1920×1032 (s=206.4), Bild 3000×1500 in H."""

    def setUp(self):
        self.r = scale.layout_rects(1920, 1032, "H", 3000, 1500)

    def test_view_width(self):
        self.assertAlmostEqual(self.r["view"][2], 1293, delta=1)

    def test_view_height(self):
        self.assertAlmostEqual(self.r["view"][3], 646, delta=1)

    def test_band_height(self):
        b = 1032 - self.r["view"][3]
        self.assertAlmostEqual(b, 386, delta=1)

    def test_square_side(self):
        m = round(scale.unit(1920, 1032) * 0.04)
        b = 1032 - self.r["view"][3]
        for sq in self.r["squares"]:
            self.assertAlmostEqual(sq[2], b - m, delta=1)

    def test_no_parade_key(self):
        """K27a: parade key removed from layout_rects."""
        self.assertNotIn("parade", self.r)


class TestHGeometryWideWindowParade(unittest.TestCase):
    """2760×1032 (s=206.4), Bild 3000×1500: parade key removed (K27a)."""

    def setUp(self):
        self.r = scale.layout_rects(2760, 1032, "H", 3000, 1500)

    def test_no_parade_key(self):
        self.assertNotIn("parade", self.r)


class TestHGeometryFourThree(unittest.TestCase):
    """1920×1032, Bild 1600×1200 (4:3) in H: view_h = H−s (Deckel)."""

    def setUp(self):
        self.r = scale.layout_rects(1920, 1032, "H", 1600, 1200)

    def test_view_height_clamped(self):
        s = scale.unit(1920, 1032)
        self.assertAlmostEqual(self.r["view"][3], 1032 - round(s), delta=1)

    def test_no_parade_key(self):
        self.assertNotIn("parade", self.r)


class TestHGeometryUltraWide(unittest.TestCase):
    """2560×1392, Bild 6000×2000 (3:1): view_h = H−2s (Deckel unten)."""

    def setUp(self):
        self.r = scale.layout_rects(2560, 1392, "H", 6000, 2000)

    def test_view_height_lower_clamp(self):
        s = scale.unit(2560, 1392)
        self.assertAlmostEqual(self.r["view"][3], 1392 - 2 * round(s), delta=1)

    def test_square_side(self):
        s = scale.unit(2560, 1392)
        m = round(s * 0.04)
        b = 1392 - self.r["view"][3]
        for sq in self.r["squares"]:
            self.assertAlmostEqual(sq[2], b - m, delta=1)


class TestHGeometryBounds(unittest.TestCase):
    """Alle H-Fälle: keine Überlappung, alles im Fenster, Quadrate |w−h| ≤ 1."""

    CASES = [
        (1920, 1032, 3000, 1500),
        (1920, 1032, 1600, 1200),
        (2560, 1392, 6000, 2000),
    ]

    def _check(self, W: int, H: int, iw: int, ih: int):
        r = scale.layout_rects(W, H, "H", iw, ih)
        blocks = [r["panel"], r["strip"], r["bottom"], r["view"]]
        for i, (x, y, w, h) in enumerate(blocks):
            self.assertGreaterEqual(x, 0, f"block {i} x<0")
            self.assertGreaterEqual(y, 0, f"block {i} y<0")
            self.assertLessEqual(x + w, W, f"block {i} exceeds W")
            self.assertLessEqual(y + h, H, f"block {i} exceeds H")
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertFalse(_overlaps(blocks[i], blocks[j]),
                                 f"blocks {i},{j} overlap")
        for k, (x, y, w, h) in enumerate(r["squares"]):
            self.assertLessEqual(abs(w - h), 1, f"square {k} not square")
            self.assertGreaterEqual(x, 0, f"square {k} x<0")
            self.assertGreaterEqual(y, 0, f"square {k} y<0")
            self.assertLessEqual(x + w, W, f"square {k} exceeds W")
            self.assertLessEqual(y + h, H, f"square {k} exceeds H")

    def test_all_cases(self):
        for W, H, iw, ih in self.CASES:
            with self.subTest(W=W, H=H, iw=iw, ih=ih):
                self._check(W, H, iw, ih)


class TestChooseModeK25h(unittest.TestCase):
    """choose_mode nutzt die neue H-Geometrie."""

    def test_landscape_image_h(self):
        result = scale.choose_mode(1920, 1032, 3000, 1500, None)
        self.assertEqual(result, "H")

    def test_portrait_image_v(self):
        result = scale.choose_mode(1920, 1032, 1000, 1500, None)
        self.assertEqual(result, "V")

    def test_ultra_wide_h(self):
        result = scale.choose_mode(2560, 1392, 6000, 2000, None)
        self.assertEqual(result, "H")

    def test_zero_image(self):
        self.assertIsNone(scale.choose_mode(1920, 1032, 0, 0, None))


class TestNoImageFallback(unittest.TestCase):
    """Ohne Bildgröße gilt K25b (parade key removed, K27a)."""

    def test_h_no_image(self):
        r = scale.layout_rects(1920, 1032, "H")
        self.assertNotIn("parade", r)
        s = scale.unit(1920, 1032)
        si = round(s)
        self.assertAlmostEqual(r["strip"][3], si, delta=1)

    def test_v_no_image(self):
        r = scale.layout_rects(1920, 1032, "V")
        self.assertNotIn("parade", r)


if __name__ == "__main__":
    unittest.main()
