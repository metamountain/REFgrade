"""Preview stays stable when the image resolution differs from the logical size.

Old code scaled both axes with the width ratio, so a 640×426 image in a 1280×853
logical space was stretched vertically by 0.5 instead of 426/853 ≈ 0.4994,
causing a ~1 px jitter in the transition row.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules.setdefault("ref_grade", _pkg)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QSizeF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import ImageView  # noqa: E402

W, H = 800, 533
LOGICAL_W, LOGICAL_H = 1280, 853


def _image_a() -> QImage:
    """1280×853: black for y < 400, white for y >= 400."""
    im = QImage(LOGICAL_W, LOGICAL_H, QImage.Format.Format_RGB32)
    im.fill(QColor(0, 0, 0))
    for y in range(400, LOGICAL_H):
        for x in range(LOGICAL_W):
            im.setPixel(x, y, QColor(255, 255, 255).rgb())
    return im


def _image_b() -> QImage:
    """A scaled to 640×426 (Smooth)."""
    return _image_a().scaled(640, 426, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)


def _first_bright_row(img: QImage, x: int) -> int:
    """First row at column x where brightness > 128."""
    for y in range(img.height()):
        c = QColor(img.pixel(x, y))
        b = (c.red() + c.green() + c.blue()) / 3
        if b > 128:
            return y
    return -1


def _mean_abs_dev(img1: QImage, img2: QImage) -> float:
    """Mean absolute brightness deviation between two same-size images."""
    total = 0.0
    count = 0
    for y in range(img1.height()):
        for x in range(0, img1.width(), 4):  # sample every 4th pixel for speed
            c1 = QColor(img1.pixel(x, y))
            c2 = QColor(img2.pixel(x, y))
            total += abs((c1.red() + c1.green() + c1.blue()) / 3
                         - (c2.red() + c2.green() + c2.blue()) / 3)
            count += 1
    return total / count


class TestPreviewStable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _grab(self, img: QImage, logical: QSizeF) -> QImage:
        v = ImageView()
        v.resize(W, H)
        v.set_image(img, logical)
        v.show()
        self.app.processEvents()
        shot = v.grab().toImage()
        v.close()
        return shot

    def test_transition_row_identical(self):
        """The first bright row must be at the same screen y for both resolutions."""
        logical = QSizeF(LOGICAL_W, LOGICAL_H)
        shot_a = self._grab(_image_a(), logical)
        shot_b = self._grab(_image_b(), logical)
        mid_x = W // 2
        row_a = _first_bright_row(shot_a, mid_x)
        row_b = _first_bright_row(shot_b, mid_x)
        self.assertGreaterEqual(row_a, 0, "image A has a bright row")
        self.assertGreaterEqual(row_b, 0, "image B has a bright row")
        self.assertEqual(row_a, row_b,
                         f"transition row differs: A={row_a} B={row_b}")

    def test_mean_deviation_bounded(self):
        """Mean absolute deviation between the two captures ≤ 4/255."""
        logical = QSizeF(LOGICAL_W, LOGICAL_H)
        shot_a = self._grab(_image_a(), logical)
        shot_b = self._grab(_image_b(), logical)
        dev = _mean_abs_dev(shot_a, shot_b)
        self.assertLessEqual(dev, 4.0,
                             f"mean abs deviation {dev:.2f} > 4.0")


if __name__ == "__main__":
    unittest.main()
