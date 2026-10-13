"""Tests für Paket K27g: Vectorscope als weiche Wolke (QImage), Hautton-Linie halb."""

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

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import Vectorscope  # noqa: E402


def _gaussian_blob(n: int, center_col: int, center_row: int,
                   sigma: float = 3.0) -> list[list[float]]:
    """Create a 2D Gaussian blob in an n×n grid."""
    data = [[0.0] * n for _ in range(n)]
    for row in range(n):
        for col in range(n):
            dx = col - center_col
            dy = row - center_row
            v = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            if v > 0.001:
                data[row][col] = v
    return data


class TestNoRaster(unittest.TestCase):
    """Kein Raster: zwischen zwei Pixeln mit Alpha > 0 auf derselben Zeile
    liegt kein Hintergrund-Pixel (keine Lücken)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_contiguous_blob_no_gaps(self):
        """Ein zusammenhängender Gauß-Fleck zeigt keine Raster-Lücken."""
        scope = Vectorscope()
        scope.resize(300, 300)

        n = 192
        # Blob in Rot-Richtung: Cb ≈ 0.39 (left of center), Cr ≈ 0.75 (above center)
        # col for Cb=0.39: col = 0.39*192 - 0.5 ≈ 74
        # row for Cr=0.75: row = (1-0.75)*192 - 0.5 ≈ 47.5
        center_col = 74
        center_row = 47
        data = _gaussian_blob(n, center_col, center_row, sigma=3.0)
        scope.set_data(data)
        scope.show()
        self.app.processEvents()

        img = scope.grab().toImage()
        w, h = img.width(), img.height()
        cx, cy = w // 2, h // 2

        # Check rows in the blob area: find rows with at least 2 alpha>0 pixels,
        # then verify no background pixel between them.
        rows_checked = 0
        for y in range(cy - 60, cy + 60):
            if y < 0 or y >= h:
                continue
            # Find all x positions with non-background pixel
            active_xs = []
            for x in range(cx - 80, cx + 80):
                if x < 0 or x >= w:
                    continue
                c = QColor(img.pixel(x, y))
                a = c.alpha()
                if a > 10:  # above background
                    active_xs.append(x)
            if len(active_xs) < 2:
                continue
            rows_checked += 1
            # Check for gaps: between min and max active x, all pixels should be non-background
            x_min, x_max = active_xs[0], active_xs[-1]
            gap_found = False
            for x in range(x_min, x_max + 1):
                c = QColor(img.pixel(x, y))
                if c.alpha() <= 5:
                    gap_found = True
                    break
            self.assertFalse(
                gap_found,
                f"Row y={y}: gap found between x={x_min} and x={x_max} (raster artifact)")
        self.assertGreater(rows_checked, 5,
                           "Too few rows checked — blob may not be visible")
        scope.deleteLater()


class TestCache(unittest.TestCase):
    """Zwei paintEvent mit gleichen Daten → Bild wird nur einmal berechnet."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_single_computation(self):
        scope = Vectorscope()
        scope.resize(300, 300)

        n = 192
        data = _gaussian_blob(n, 96, 96, sigma=5.0)
        scope.set_data(data)
        scope.show()
        self.app.processEvents()

        # Count how many times _compute_cloud_image is called
        original = Vectorscope._compute_cloud_image
        call_count = [0]

        def counting(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        Vectorscope._compute_cloud_image = staticmethod(counting)
        try:
            # Trigger two more paint events
            scope.update()
            self.app.processEvents()
            scope.update()
            self.app.processEvents()
        finally:
            Vectorscope._compute_cloud_image = staticmethod(original)

        # First paint computes, subsequent paints use cache
        self.assertEqual(call_count[0], 0,
                         "Cache should prevent recomputation on subsequent paints")
        scope.deleteLater()


class TestSkinLineHalf(unittest.TestCase):
    """Hautton-Linie: Pixel in Gegenrichtung (unten rechts, halber Radius) ist Hintergrund."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_opposite_direction_is_background(self):
        scope = Vectorscope()
        scope.resize(300, 300)
        scope.set_data([])
        scope.show()
        self.app.processEvents()

        img = scope.grab().toImage()
        w, h = img.width(), img.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2

        # Skin line goes to upper-left (123°). Opposite is lower-right.
        # Angle 123° in math coords → screen: upper-left
        # Opposite: 123° - 180° = -57° → lower-right
        angle_opp = math.radians(123.0 - 180.0)
        # In screen coords: x = cx + cos(angle)*d, y = cy - sin(angle)*d
        # For the opposite direction:
        dx = math.cos(math.radians(-57.0)) * radius * 0.5
        dy = -math.sin(math.radians(-57.0)) * radius * 0.5
        # This gives lower-right: dx > 0, dy > 0 (screen y is down)
        x = int(cx + dx)
        y = int(cy + dy)

        # Sample a small area
        max_brightness = 0
        for dy2 in range(-3, 4):
            for dx2 in range(-3, 4):
                xx, yy = x + dx2, y + dy2
                if 0 <= xx < w and 0 <= yy < h:
                    c = QColor(img.pixel(xx, yy))
                    b = c.red() + c.green() + c.blue()
                    max_brightness = max(max_brightness, b)

        # Background is (20,20,20) = 60 total. Allow some tolerance for
        # the crosshair lines (40,40,40 = 120) and anti-aliasing.
        # The skin line should NOT be here, so brightness should be low.
        self.assertLess(max_brightness, 130,
                        f"Opposite direction has brightness {max_brightness} — "
                        f"skin line extends too far (should be center-outward only)")
        scope.deleteLater()


if __name__ == "__main__":
    unittest.main()
