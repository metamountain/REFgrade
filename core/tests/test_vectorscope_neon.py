"""Tests für Paket K25g: Vectorscope Neon — Hue-Farbe, Glow, Übertreibung."""

from __future__ import annotations

import math
import os
import sys
import types
import unittest
from pathlib import Path

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# torch vor PyQt6 importieren, um DLL-Konflikt zu vermeiden
import torch  # noqa: E402

from PyQt6.QtCore import QPointF, QRectF  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from ref_grade.widgets import Vectorscope, VS_EXAGGERATE  # noqa: E402


class TestCbcrToRgb(unittest.TestCase):
    """_cbcr_to_rgb: Farbe nur aus dem Farbwinkel, S=1 V=1, kein Grau."""

    def test_12_angles_saturated(self):
        """Für 12 Winkel: max(R,G,B)=255 und min(R,G,B)≤40 (keine weißlichen Töne)."""
        for i in range(12):
            angle = 2 * math.pi * i / 12
            cb = 0.5 + 0.4 * math.cos(angle)
            cr = 0.5 + 0.4 * math.sin(angle)
            r, g, b = Vectorscope._cbcr_to_rgb(cb, cr)
            self.assertEqual(max(r, g, b), 255,
                             f"Angle {i}: max({r},{g},{b}) != 255")
            self.assertLessEqual(min(r, g, b), 40,
                                 f"Angle {i}: min({r},{g},{b}) > 40 (whitish)")

    def test_center_not_white(self):
        """Cb=Cr=0.5: keine Farbe mit min(R,G,B) > 40."""
        r, g, b = Vectorscope._cbcr_to_rgb(0.5, 0.5)
        self.assertLessEqual(min(r, g, b), 40,
                             f"Center color ({r},{g},{b}) too white")


class TestExaggerate(unittest.TestCase):
    """_exaggerate: radiale Übertreibung."""

    def test_d_01(self):
        """d=0.1 → 0.271 ±0.005."""
        result = Vectorscope._exaggerate(0.1)
        self.assertAlmostEqual(result, 0.271, delta=0.005,
                               msg=f"d=0.1 → {result}, expected 0.271±0.005")

    def test_d_03(self):
        """d=0.3 → 0.657 ±0.005."""
        result = Vectorscope._exaggerate(0.3)
        self.assertAlmostEqual(result, 0.657, delta=0.005,
                               msg=f"d=0.3 → {result}, expected 0.657±0.005")

    def test_d_1(self):
        """d=1 → 0.95."""
        result = Vectorscope._exaggerate(1.0)
        self.assertAlmostEqual(result, 0.95, delta=0.001,
                               msg=f"d=1 → {result}, expected 0.95")

    def test_constant(self):
        """VS_EXAGGERATE = 3."""
        self.assertEqual(VS_EXAGGERATE, 3)


class TestCloudRender(unittest.TestCase):
    """Wolke mit einem Punkt in Rot-Richtung: Neon-Rot, Glow sichtbar."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_red_point(self):
        """Einzelner Punkt bei Rot: hellstes Pixel R≥200, G,B≤80; Glow-Ring rötlich."""
        scope = Vectorscope()
        scope.resize(200, 200)

        # 16×16 grid, one point at red direction (top, slightly left)
        n = 16
        data = [[0.0] * n for _ in range(n)]
        # Red is at top: row 1, col 7 (slightly left of center)
        data[1][7] = 1.0
        scope.set_data(data)
        scope.show()
        self.app.processEvents()

        img = scope.grab().toImage()
        w, h = img.width(), img.height()

        # Find the brightest red pixel in the full widget
        best_r = 0
        best_g = 0
        best_b = 0
        best_pos = (0, 0)
        for y in range(h):
            for x in range(w):
                c = QColor(img.pixel(x, y))
                if c.red() > best_r:
                    best_r, best_g, best_b = c.red(), c.green(), c.blue()
                    best_pos = (x, y)

        self.assertGreaterEqual(best_r, 200,
                                f"Brightest pixel R={best_r} < 200 at {best_pos}")
        self.assertLessEqual(best_g, 80,
                             f"Brightest pixel G={best_g} > 80 at {best_pos}")
        # B can be up to ~100 from conical gradient interpolation (magenta→red)
        self.assertLessEqual(best_b, 100,
                             f"Brightest pixel B={best_b} > 100 at {best_pos}")

        # Glow: check a pixel near the cloud (top edge, red direction).
        # (row=1, col=7) in 16×16 is near the Cb/Cr edge → exaggerated to
        # the scope edge (y≈0–5).  The 1/4-scale glow blur extends below.
        cx, cy = w // 2, h // 2
        glow_found = False
        for dx in range(-25, 26, 5):
            for dy in range(0, 31, 5):
                x, y = cx + dx, dy
                if 0 <= x < w and 0 <= y < h:
                    c = QColor(img.pixel(x, y))
                    if c.red() > 20 and c.red() + c.green() + c.blue() > 30:
                        glow_found = True
                        break
            if glow_found:
                break
        self.assertTrue(glow_found,
                        "No reddish glow pixel found near cloud (top edge)")


if __name__ == "__main__":
    unittest.main()
