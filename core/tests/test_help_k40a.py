"""K40a: per-value "?" help.

Checks (spec 103):
1. Every SliderRow name in the dialog that has a HELP entry has HELP[name]["what"],
   20–320 chars, no "=", no "*".
2. Every such SliderRow has exactly one "?" button, left of the value label in the same row;
   rows without a HELP entry have none.
3. Clicking "?" on shadows opens a visible popup whose text contains HELP["shadows"]["what"].
4. Curves: click on the "?" glyph (position from _help_rect) pins the help panel
   (_help_pinned True), second click unpins (False); clicking ↺ still resets only the
   active channel.
"""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Callable

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
if "ref_grade" not in sys.modules:
    _pkg = types.ModuleType("ref_grade")
    _pkg.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from refgrade.help import HELP  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.curves_ui import CurvesEditor  # noqa: E402
from ref_grade.widgets import SliderRow  # noqa: E402

W, H = 2560, 1392


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K40aTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 2560×1392, all sections expanded."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()
        for key in list(self.dlg.sections.keys()):
            self.dlg._set_section_collapsed(key, False)
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

    def _all_rows(self) -> list[SliderRow]:
        return list(self.dlg.findChildren(SliderRow))


class TestWhatTexts(K40aTestBase):
    """1. Every SliderRow name in the dialog that has a HELP entry has a valid 'what'."""

    def test_all_rows_have_what(self):
        checked = 0
        for row in self._all_rows():
            entry = HELP.get(row.name)
            if entry is None:
                continue
            checked += 1   # outside the subTest: a subTest failure skips the rest of the block
            with self.subTest(slider=row.name):
                self.assertIn("what", entry, f"HELP[{row.name!r}] missing 'what'")
                what = entry["what"]
                self.assertGreaterEqual(len(what), 20,
                                        f"HELP[{row.name!r}]['what'] is {len(what)} chars, need ≥ 20")
                self.assertLessEqual(len(what), 320,
                                     f"HELP[{row.name!r}]['what'] is {len(what)} chars, need ≤ 320")
                self.assertNotIn("=", what, f"HELP[{row.name!r}]['what'] contains '='")
                self.assertNotIn("*", what, f"HELP[{row.name!r}]['what'] contains '*'")
        self.assertGreaterEqual(checked, 50, f"Only {checked} rows checked, expected ≥ 50")


class TestHelpButtons(K40aTestBase):
    """2. Exactly one '?' button per row with a HELP entry, left of the value label."""

    def test_one_button_left_of_value(self):
        checked = 0
        for row in self._all_rows():
            if row.name in HELP:
                checked += 1   # outside the subTest: a subTest failure skips the rest of the block
            with self.subTest(slider=row.name):
                btns = [b for b in row.findChildren(QPushButton)
                        if b.objectName() == f"help_{row.name}"]
                if row.name in HELP:
                    self.assertEqual(len(btns), 1,
                                     f"Row {row.name!r}: expected 1 '?' button, found {len(btns)}")
                    btn = btns[0]
                    self.assertEqual(btn.text(), "?")
                    self.assertTrue(btn.toolTip().strip(),
                                    f"Row {row.name!r}: '?' button has no tooltip")
                    # geometry only for visible rows (hidden HSL band rows have x=y=0)
                    if row.isVisibleTo(self.dlg):
                        self.assertLess(btn.x(), row.value_label.x(),
                                        f"Row {row.name!r}: '?' button x={btn.x()} not left of value x={row.value_label.x()}")
                        self.assertEqual(btn.y(), row.value_label.y(),
                                         f"Row {row.name!r}: '?' button not in the same row as the value label")
                else:
                    self.assertEqual(len(btns), 0,
                                     f"Row {row.name!r} has no HELP entry but a '?' button")
        self.assertGreaterEqual(checked, 50, f"Only {checked} rows checked, expected ≥ 50")


class TestHelpPopup(K40aTestBase):
    """3. Click '?' on shadows → visible popup containing HELP['shadows']['what']."""

    def test_shadows_popup(self):
        row = next((r for r in self._all_rows() if r.name == "shadows"), None)
        self.assertIsNotNone(row, "No 'shadows' SliderRow in the dialog")
        self.assertIsNotNone(row.help_btn, "shadows row has no '?' button")
        self.assertTrue(row.help_btn.isVisibleTo(self.dlg), "shadows '?' button not visible")
        QTest.mouseClick(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertIsNotNone(row.help_popup, "Popup was not created")
        self.assertTrue(row.help_popup.isVisible(), "Popup is not visible after click")
        text = row.help_popup.text()
        self.assertIn(HELP["shadows"]["what"], text,
                      "Popup text does not contain HELP['shadows']['what']")
        self.assertIn(HELP["shadows"]["short"], text,
                      "Popup text does not contain HELP['shadows']['short']")


class TestCurvesHelpGlyph(unittest.TestCase):
    """4. Curves: '?' glyph pins/unpins the help panel; ↺ still resets the active channel."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.ed = CurvesEditor()
        self.ed.resize(400, 400)
        self.ed.show()
        self.app.processEvents()

    def tearDown(self):
        self.ed.close()
        self.ed.deleteLater()
        self.app.processEvents()

    def _click_at(self, rect) -> None:
        QTest.mouseClick(self.ed, Qt.MouseButton.LeftButton,
                         pos=QPoint(int(rect.center().x()), int(rect.center().y())))
        self.app.processEvents()

    def test_help_glyph_toggles_pinned(self):
        self.assertFalse(self.ed._help_pinned)
        self._click_at(self.ed._help_rect(self.ed._plot_rect()))
        self.assertTrue(self.ed._help_pinned, "First click on '?' should pin the help panel")
        self._click_at(self.ed._help_rect(self.ed._plot_rect()))
        self.assertFalse(self.ed._help_pinned, "Second click on '?' should unpin the panel")

    def test_help_glyph_click_does_not_reset(self):
        self.ed.set_curve("rgb", [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]])
        self._click_at(self.ed._help_rect(self.ed._plot_rect()))
        self.assertEqual(self.ed.get_curve("rgb"), [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
                         "Clicking '?' must not change the curve")

    def test_reset_glyph_resets_active_channel_only(self):
        self.ed.set_curve("rgb", [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]])
        self.ed.set_curve("r", [[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]])
        self._click_at(self.ed._reset_rect(self.ed._plot_rect()))
        self.assertEqual(self.ed.get_curve("rgb"), [[0.0, 0.0], [1.0, 1.0]],
                         "↺ should reset the active channel (rgb)")
        self.assertEqual(self.ed.get_curve("r"), [[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]],
                         "↺ should not touch the other channels")


if __name__ == "__main__":
    unittest.main()
