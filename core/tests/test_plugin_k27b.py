"""Tests für Paket K27b: Vectorscope fein, keine Beschriftung, Regenbogen-Kontur."""

from __future__ import annotations

import inspect
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

from PyQt6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from refgrade.scope import vectorscope  # noqa: E402
from ref_grade.widgets import Vectorscope  # noqa: E402


class TestDefaultBins(unittest.TestCase):
    """scope.vectorscope(img) liefert 192 Zeilen à 192 Werte (Default)."""

    def test_default_192(self):
        x = torch.rand(1, 16, 16, 3)
        result = vectorscope(x)
        self.assertEqual(len(result), 192)
        self.assertEqual(len(result[0]), 192)


class TestNoDrawText(unittest.TestCase):
    """Kein drawText-Aufruf mehr in Vectorscope.paintEvent (Quelltext-Test)."""

    def test_no_drawtext_in_paintevent(self):
        src = inspect.getsource(Vectorscope.paintEvent)
        self.assertNotIn("drawText", src,
                         "paintEvent must not call drawText")

    def test_no_drawtext_in_scope_circle(self):
        src = inspect.getsource(Vectorscope._draw_scope_circle)
        self.assertNotIn("drawText", src,
                         "_draw_scope_circle must not call drawText")


class TestRainbowContour(unittest.TestCase):
    """Kontur: Pixel auf dem äußeren Kreis in Richtung Rot/Blau."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _render(self, size: int = 300) -> QImage:
        scope = Vectorscope()
        scope.resize(size, size)
        # Empty data: no cloud, just the contour
        scope.set_data([])
        scope.show()
        self.app.processEvents()
        img = scope.grab().toImage()
        scope.deleteLater()
        return img

    def test_red_direction(self):
        """Pixel auf dem äußeren Kreis in Richtung Rot: R ≥ 180, G,B ≤ 90."""
        img = self._render(300)
        w, h = img.width(), img.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 2  # slightly inside the edge

        # Red is at the top in Cb/Cr space (Cr > 0, Cb ≈ 0)
        # In screen: top = (cx, cy - radius)
        x, y = cx, cy - radius
        # Sample a small area to account for anti-aliasing
        best_r, best_g, best_b = 0, 0, 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                c = QColor(img.pixel(x + dx, y + dy))
                if c.red() > best_r:
                    best_r, best_g, best_b = c.red(), c.green(), c.blue()
        self.assertGreaterEqual(best_r, 180,
                                f"Red direction: R={best_r} < 180")
        self.assertLessEqual(best_g, 90,
                             f"Red direction: G={best_g} > 90")
        self.assertLessEqual(best_b, 90,
                             f"Red direction: B={best_b} > 90")

    def test_blue_direction(self):
        """Pixel auf dem äußeren Kreis in Richtung Blau: B ≥ 180, R,G ≤ 130."""
        img = self._render(300)
        w, h = img.width(), img.height()
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 2

        # Blue is at the right in Cb/Cr space (Cb > 0, Cr ≈ 0)
        # In screen: right = (cx + radius, cy)
        x, y = cx + radius, cy
        best_b, best_r, best_g = 0, 0, 0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                c = QColor(img.pixel(x + dx, y + dy))
                if c.blue() > best_b:
                    best_b, best_r, best_g = c.blue(), c.red(), c.green()
        self.assertGreaterEqual(best_b, 180,
                                f"Blue direction: B={best_b} < 180")
        self.assertLessEqual(best_r, 130,
                             f"Blue direction: R={best_r} > 130")
        self.assertLessEqual(best_g, 130,
                             f"Blue direction: G={best_g} > 130")


class TestFineDots(unittest.TestCase):
    """Ein einzelner Datenpunkt füllt höchstens 3 × Zellgröße im Durchmesser."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_single_point_diameter(self):
        scope = Vectorscope()
        scope.resize(300, 300)

        n = 192
        data = [[0.0] * n for _ in range(n)]
        # Single point at center
        data[n // 2][n // 2] = 1.0
        scope.set_data(data)
        scope.show()
        self.app.processEvents()

        img = scope.grab().toImage()
        w, h = img.width(), img.height()
        cx, cy = w // 2, h // 2
        cell_size = min(w, h) / n

        # Scan diagonally (avoids the crosshair lines)
        # The visible extent (including glow) should be ≤ 3 * cell_size diameter
        max_extent = 0
        for angle_deg in (45, 135, 225, 315):
            a = math.radians(angle_deg)
            dx, dy = math.cos(a), math.sin(a)
            dist = 0
            while dist < 50:
                x = int(cx + dx * dist)
                y = int(cy + dy * dist)
                if x < 0 or x >= w or y < 0 or y >= h:
                    break
                c = QColor(img.pixel(x, y))
                if c.red() + c.green() + c.blue() > 80:
                    max_extent = max(max_extent, dist)
                else:
                    break
                dist += 1

        # Diameter = 2 * max_extent.
        # K27g: image-based rendering with 1/4-scale glow blur.
        # Main image: bilinear falloff ~2×cell. Glow (1/4 blur): ~4× main.
        # Total visible: ~10×cell. Limit: 12×cell.
        diameter = 2 * max_extent
        limit = 12 * cell_size
        self.assertLessEqual(diameter, limit + 2,
                             f"Blob diameter {diameter:.1f} > 12×cell ({limit:.1f}+2)")


if __name__ == "__main__":
    unittest.main()
