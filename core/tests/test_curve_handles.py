"""Krita-style curve handles (user 2026-10-08: "curves benötigt richtig gute Handles").

Select by click, drag (y always 0..1), drag a point out of the widget to delete it,
Del deletes the selected point, arrows nudge it.
"""

import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", pkg)

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from ref_grade.curves_ui import CurvesEditor as CurveEditor  # noqa: E402

L = Qt.MouseButton.LeftButton
NO = Qt.KeyboardModifier.NoModifier


class TestCurveHandles(unittest.TestCase):

    def setUp(self):
        self.ed = CurveEditor()
        self.ed.resize(256, 256)
        self.ed.show()
        app.processEvents()
        self.emitted = []
        self.ed.changed.connect(lambda ch, pts: self.emitted.append([list(p) for p in pts]))

    def _xy(self, x, y):
        pt = self.ed._to_screen(x, y)
        return QPoint(round(pt.x()), round(pt.y()))

    def _add_mid_point(self):
        QTest.mouseClick(self.ed, L, NO, self._xy(0.5, 0.5))
        app.processEvents()
        return self.ed.curves["rgb"]

    def test_click_adds_and_selects(self):
        pts = self._add_mid_point()
        self.assertEqual(len(pts), 3)
        self.assertEqual(self.ed._sel_idx, 1)

    def test_drag_far_above_clamps_y(self):
        self._add_mid_point()
        QTest.mousePress(self.ed, L, NO, self._xy(0.5, 0.5))
        QTest.mouseMove(self.ed, QPoint(self._xy(0.5, 0.5).x(), 5))   # near top, inside
        QTest.mouseRelease(self.ed, L, NO, QPoint(self._xy(0.5, 0.5).x(), 5))
        y = self.ed.curves["rgb"][1][1]
        self.assertTrue(0.9 < y <= 1.0, y)
        for pts in self.emitted:                      # never an invalid curve to the worker
            self.assertTrue(all(0.0 <= p[1] <= 1.0 for p in pts))

    def test_drag_out_deletes(self):
        self._add_mid_point()
        QTest.mousePress(self.ed, L, NO, self._xy(0.5, 0.5))
        QTest.mouseMove(self.ed, QPoint(128, 300))                     # 44 px below the widget
        QTest.mouseRelease(self.ed, L, NO, QPoint(128, 300))
        self.assertEqual(len(self.ed.curves["rgb"]), 2)

    def test_endpoint_is_not_deleted_by_drag_out(self):
        QTest.mousePress(self.ed, L, NO, self._xy(0.0, 0.0))
        QTest.mouseMove(self.ed, QPoint(-40, 300))
        QTest.mouseRelease(self.ed, L, NO, QPoint(-40, 300))
        self.assertEqual(len(self.ed.curves["rgb"]), 2)

    def test_delete_key_and_arrows(self):
        self._add_mid_point()
        y0 = self.ed.curves["rgb"][1][1]
        QTest.keyClick(self.ed, Qt.Key.Key_Up)
        self.assertAlmostEqual(self.ed.curves["rgb"][1][1], min(1.0, y0 + 1 / 255), places=6)
        QTest.keyClick(self.ed, Qt.Key.Key_Delete)
        self.assertEqual(len(self.ed.curves["rgb"]), 2)

    def test_hue_curve_order_kept(self):
        self.ed.active_channel = "hue_hue"
        QTest.mouseClick(self.ed, L, NO, self._xy(0.3, 0.5))
        QTest.mouseClick(self.ed, L, NO, self._xy(0.6, 0.5))
        pts = self.ed.curves["hue_hue"]
        i = next(k for k, p in enumerate(pts) if abs(p[0] - 0.3) < 0.02)
        QTest.mousePress(self.ed, L, NO, self._xy(0.3, 0.5))
        QTest.mouseMove(self.ed, self._xy(0.8, 0.7))                  # past its neighbour
        QTest.mouseRelease(self.ed, L, NO, self._xy(0.8, 0.7))
        xs = [p[0] for p in self.ed.curves["hue_hue"]]
        self.assertEqual(xs, sorted(xs))


if __name__ == "__main__":
    unittest.main()
