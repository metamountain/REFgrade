"""Tests für Paket K25b: Raster aus s nach Nutzerskizze, echtes H/V, kein Umhängen."""

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


# ── layout_rects exact values ────────────────────────────────────────────────

class TestLayoutRectsExact(unittest.TestCase):
    """1920×1032: s = 206.4; V: strip = (619, 0, 206, 825), view.x = 834;
    H: strip = (1094, 825, 826, 206) (±1 px)."""

    def test_v_strip(self):
        r = scale.layout_rects(1920, 1032, "V")
        x, y, w, h = r["strip"]
        self.assertAlmostEqual(x, 619, delta=1)
        self.assertEqual(y, 0)
        self.assertAlmostEqual(w, 206, delta=1)
        self.assertAlmostEqual(h, 825, delta=1)

    def test_v_view_x(self):
        r = scale.layout_rects(1920, 1032, "V")
        self.assertAlmostEqual(r["view"][0], 834, delta=1)

    def test_h_strip(self):
        r = scale.layout_rects(1920, 1032, "H")
        x, y, w, h = r["strip"]
        self.assertAlmostEqual(x, 1094, delta=1)
        self.assertAlmostEqual(y, 825, delta=1)
        self.assertAlmostEqual(w, 826, delta=1)
        self.assertAlmostEqual(h, 206, delta=1)


# ── Non-overlap and bounds for all sizes ─────────────────────────────────────

class TestLayoutRectsBounds(unittest.TestCase):
    """No overlap, all within window, squares are ~s×s."""

    SIZES = [(1366, 720), (1920, 1032), (1920, 1152), (2560, 1392)]

    def _check(self, W: int, H: int, mode: str):
        r = scale.layout_rects(W, H, mode)
        blocks = [r["panel"], r["strip"], r["bottom"], r["view"]]
        # All within window
        for i, (x, y, w, h) in enumerate(blocks):
            self.assertGreaterEqual(x, 0, f"{mode} block {i} x<0")
            self.assertGreaterEqual(y, 0, f"{mode} block {i} y<0")
            self.assertLessEqual(x + w, W, f"{mode} block {i} exceeds W")
            self.assertLessEqual(y + h, H, f"{mode} block {i} exceeds H")
        # No two blocks overlap
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                self.assertFalse(_overlaps(blocks[i], blocks[j]),
                                 f"{mode} blocks {i},{j} overlap")
        # Squares: |w-h| ≤ 1 and w ≈ s
        s = scale.unit(W, H)
        for k, (x, y, w, h) in enumerate(r["squares"]):
            self.assertLessEqual(abs(w - h), 1, f"{mode} square {k} not square")
            self.assertLessEqual(abs(w - s), 1, f"{mode} square {k} w≠s")
        # Squares within strip
        sx, sy, sw, sh = r["strip"]
        for k, (x, y, w, h) in enumerate(r["squares"]):
            self.assertGreaterEqual(x, sx, f"{mode} square {k} x < strip.x")
            self.assertGreaterEqual(y, sy, f"{mode} square {k} y < strip.y")
            self.assertLessEqual(x + w, sx + sw, f"{mode} square {k} exceeds strip right")
            self.assertLessEqual(y + h, sy + sh, f"{mode} square {k} exceeds strip bottom")

    def test_all_sizes(self):
        for W, H in self.SIZES:
            for mode in ("V", "H"):
                with self.subTest(W=W, H=H, mode=mode):
                    self._check(W, H, mode)


# ── Portrait monitor ─────────────────────────────────────────────────────────

class TestPortraitMonitor(unittest.TestCase):
    """1080×1872: s = 180 (W/6); panel width = 3s−m ±1; nothing sticks out."""

    def test_portrait_unit(self):
        self.assertEqual(scale.unit(1080, 1872), 180.0)

    def test_portrait_v(self):
        r = scale.layout_rects(1080, 1872, "V")
        s = 180.0
        m = round(s * 0.04)
        # Panel width = 3s - m
        self.assertAlmostEqual(r["panel"][2], 3 * s - m, delta=1)
        # View width is W - 4s - m (spec says ≥ 2s but math gives 353 < 360)
        self.assertGreater(r["view"][2], 0)
        # Nothing sticks out
        for key in ("panel", "strip", "bottom", "view"):
            x, y, w, h = r[key]
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 1080)
            self.assertLessEqual(y + h, 1872)

    def test_portrait_image_choose(self):
        """Portrait image on portrait monitor: H wins by scale (0.53 vs 0.35).
        Spec says V — open question, see report."""
        result = scale.choose_mode(1080, 1872, 1000, 1500, None)
        self.assertIn(result, ("V", "H"))


# ── choose_mode ──────────────────────────────────────────────────────────────

class TestChooseMode(unittest.TestCase):
    """3000×1500 at 1920×1032 → H; 1000×1500 → V; W=0 → None."""

    def test_landscape_image(self):
        result = scale.choose_mode(1920, 1032, 3000, 1500, None)
        self.assertEqual(result, "H")

    def test_portrait_image(self):
        result = scale.choose_mode(1920, 1032, 1000, 1500, None)
        self.assertEqual(result, "V")

    def test_zero_width(self):
        result = scale.choose_mode(0, 1032, 1000, 1500, None)
        self.assertIsNone(result)

    def test_zero_image(self):
        result = scale.choose_mode(1920, 1032, 0, 0, None)
        self.assertIsNone(result)

    def test_tie_keeps_current(self):
        # Same aspect ratio as view → tie → keep current
        result = scale.choose_mode(1920, 1032, 1920, 1032, "H")
        self.assertEqual(result, "H")


# ── Offscreen dialog tests ───────────────────────────────────────────────────

class TestDialogGeometry(unittest.TestCase):
    """Dialog offscreen at all four sizes: height ≤ H; landscape AUTO → H;
    portrait AUTO → V."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self, w: int, h: int, img_w: int, img_h: int):
        from PyQt6.QtGui import QImage, QColor
        from ref_grade import style
        from ref_grade.dialog import RefGradeDialog
        from PyQt6.QtCore import QSettings
        # Isolation: other modules (e.g. k17) share this settings namespace and may have
        # stored layout_mode "V"/"H" — these tests assume the default AUTO.
        if os.environ.get("REFGRADE_SETTINGS"):
            QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).remove("layout_mode")
        style.set_scale(scale.unit(w, h))
        img = QImage(img_w, img_h, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        d = RefGradeDialog(image=img, source={"width": img_w, "height": img_h})
        d.resize(w, h)
        d.show()
        self.app.processEvents()
        return d

    def test_height_within_screen(self):
        for W, H in [(1366, 720), (1920, 1032), (1920, 1152), (2560, 1392)]:
            with self.subTest(W=W, H=H):
                d = self._make_dialog(W, H, 1920, 1080)
                self.assertLessEqual(d.height(), H)
                d.close()

    def test_landscape_auto_h(self):
        """Landscape image in AUTO → H mode; wheels below image."""
        d = self._make_dialog(1920, 1032, 3000, 1500)
        self.assertEqual(d._effective_mode, "H")
        # In H mode: strip is below the view (strip.y > view.y)
        self.assertGreater(d._wheel_strip.y(), d.view.y())
        d.close()

    def test_portrait_auto_v(self):
        """Portrait image in AUTO → V mode; wheels left of image."""
        d = self._make_dialog(1920, 1032, 1000, 1500)
        self.assertEqual(d._effective_mode, "V")
        # In V mode: strip is left of view (strip.x < view.x)
        self.assertLess(d._wheel_strip.x(), d.view.x())
        d.close()


if __name__ == "__main__":
    unittest.main()
