"""Split line stays visible: 1 px white, 1 px black shadow right of it (user 2026-10-12).

It disappeared because the clip rectangle of the image drawing was still active and the line
sits exactly on its edge. Checked at fractional split positions, with and without overlay.
"""

from __future__ import annotations

import math
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

from PyQt6.QtCore import QRectF  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import ImageView  # noqa: E402

W, H = 600, 400


def _img(color: QColor) -> QImage:
    im = QImage(W, H, QImage.Format.Format_RGB32)
    im.fill(color)
    return im


class TestSplitLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _render(self, pos: float, overlay: bool) -> tuple[QImage, int]:
        v = ImageView()
        v.resize(W, H)
        v.set_image(_img(QColor(120, 120, 120)))
        v.set_original(_img(QColor(120, 120, 120)))
        v.split_mode = True
        v.split_pos = pos
        if overlay:
            v.show_overlay = True
            v.set_overlay(_img(QColor(120, 120, 120)), QRectF(0, 0, W, H))
        v.show()
        self.app.processEvents()
        shot = v.grab().toImage()
        x = math.floor(v._split_x())
        v.close()
        return shot, x

    def test_line_white_and_shadow_black(self):
        for pos in (0.5, 0.3337, 0.71, 0.123):
            for overlay in (False, True):
                with self.subTest(pos=pos, overlay=overlay):
                    shot, x = self._render(pos, overlay)
                    y = H // 3
                    self.assertEqual(QColor(shot.pixel(x, y)).getRgb()[:3], (255, 255, 255),
                                     "white line at the split")
                    self.assertEqual(QColor(shot.pixel(x + 1, y)).getRgb()[:3], (0, 0, 0),
                                     "black 1 px shadow right of it")
                    self.assertEqual(QColor(shot.pixel(x + 3, y)).getRgb()[:3], (120, 120, 120),
                                     "only 1 px shadow")

    def test_grab_far_from_the_line(self):
        """User 2026-10-12: 6 px was too fiddly — 20 px beside the line still grabs it,
        and hovering there shows the resize cursor."""
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest
        v = ImageView()
        v.resize(W, H)
        v.set_image(_img(QColor(120, 120, 120)))
        v.set_original(_img(QColor(120, 120, 120)))
        v.split_mode = True
        v.split_pos = 0.5
        v.show()
        self.app.processEvents()
        QTest.qWaitForWindowExposed(v)   # QWindow-Overload braucht ein exposed Window
        self.assertGreaterEqual(v._split_grab_half(), 20)
        y = H // 3
        start = QPoint(W // 2 + 20, y)
        # QWindow-Overload statt QWidget-Overload: der QWidget-Pfad nutzt QCursor::setPos
        # (asynchroner WM-Roundtrip; Timing/Placement WM-abhaengig) und wird im
        # Shared-Process-Runner unzuverlaessig. Die QWindow-Variante liefert das Event
        # sofort und synchron. (Qt-Wiki "Writing good tests", 2026-10-18)
        win = v.window()
        QTest.mouseMove(win, start)
        self.app.processEvents()
        self.assertEqual(v.cursor().shape(), Qt.CursorShape.SizeHorCursor, "hover cursor")
        QTest.mousePress(win, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(win, QPoint(W // 4, y))
        QTest.mouseRelease(win, Qt.MouseButton.LeftButton, pos=QPoint(W // 4, y))
        self.app.processEvents()
        self.assertAlmostEqual(v.split_pos, 0.25, delta=0.01)
        far = QPoint(W - 60, y)
        QTest.mouseMove(win, far)
        self.app.processEvents()
        self.assertNotEqual(v.cursor().shape(), Qt.CursorShape.SizeHorCursor, "no cursor far away")
        v.close()


if __name__ == "__main__":
    unittest.main()
