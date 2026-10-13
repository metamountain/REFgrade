"""K25f: RGB Parade — Plugin offscreen Test."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import Parade  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class TestParadeWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _render(self, w: int, h: int, data: dict) -> QImage:
        """Render the Parade widget to a QImage."""
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        widget = Parade()
        widget.setFixedSize(w, h)
        widget.set_data(data)
        widget.show()
        _wait(self.app, lambda: widget.width() >= w - 2)
        self.app.processEvents()
        img = QImage(widget.size(), QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 255))
        widget.render(img)
        widget.close()
        widget.deleteLater()
        self.app.processEvents()
        return img

    def test_top_left_red(self):
        """Parade 300×100, r only top-left set → pixel (20,5) reddish, (250,50) black."""
        # Parade data are 256x256 grey PNGs (row 0 = top). R: bright top-left quarter.
        tmp = tempfile.mkdtemp(prefix="k25f_")
        data = {}
        for ch in ("r", "g", "b"):
            g = QImage(256, 256, QImage.Format.Format_Grayscale8)
            g.fill(0)
            if ch == "r":
                # setPixel(x, y, 255) on Grayscale8 is QRgb 0x0000FF (≈ grey 18), not white;
                # hidden while the parade drew opaque blocks (2026-10-11)
                for y in range(64):
                    for x in range(64):
                        g.setPixelColor(x, y, QColor(255, 255, 255))
            data[ch] = os.path.join(tmp, ch + ".png")
            g.save(data[ch], "PNG")

        img = self._render(300, 100, data)

        # Pixel (20, 5): should be in the top-left red cell
        # Panel width = (300 - 2*6) // 3 = 96, so first panel is x=0..96
        # Cell width = 96/4 = 24, cell height = 100/4 = 25
        # r[3][0]: level 3 (top), col 0 (left) → x=0..24, y=0..25 (top)
        # Pixel (20, 5) is in this cell
        px = img.pixelColor(20, 5)
        self.assertGreater(px.red(), 150, f"Expected R>150, got {px.red()}")
        # additive glow lifts a full core toward white (the wanted "hot" look), so red must
        # dominate clearly rather than G/B stay under a fixed value
        self.assertGreater(px.red(), 1.5 * px.green(), f"R not dominant: {px.getRgb()}")
        self.assertGreater(px.red(), 1.5 * px.blue(), f"R not dominant: {px.getRgb()}")

        # Pixel (250, 50): should be black (in the green/blue panel area, no data)
        px2 = img.pixelColor(250, 50)
        self.assertLess(px2.red(), 30, f"Expected R<30, got {px2.red()}")
        self.assertLess(px2.green(), 30, f"Expected G<30, got {px2.green()}")
        self.assertLess(px2.blue(), 30, f"Expected B<30, got {px2.blue()}")

    def test_empty_data_black(self):
        """No data → entire widget is black."""
        img = self._render(300, 100, {})
        for x in (10, 100, 200, 290):
            for y in (10, 50, 90):
                px = img.pixelColor(x, y)
                self.assertLess(px.red(), 10)
                self.assertLess(px.green(), 10)
                self.assertLess(px.blue(), 10)


if __name__ == "__main__":
    unittest.main()
