"""Frozen-help fix (2026-10-17): the '?' click must close the help popup.

A Qt.Popup help window is auto-closed by Qt when its own '?' button is clicked (the
button's window activates, the popup loses activation) — during the press, before the
release's clicked() runs. The open state is therefore tracked in a flag that the
auto-close does NOT clear; the toggle reads the flag (not isVisible()), and a genuine
click-away press clears the flag via the app-level _HelpPressFilter.

Checks:
1. SliderRow '?': click opens; the auto-close (hide) does not clear the flag; the
   release (clicked) then closes the popup. Click-away clears the flag so the next
   click opens again.
2. Section help (general '?'): same toggle; the flag survives the auto-close and the
   second click closes.
3. A press on the toggle button or inside the open popup does NOT clear the flag.
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
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
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


class _Base(unittest.TestCase):
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

    def _row(self, name: str) -> SliderRow:
        row = next((r for r in self.dlg.findChildren(SliderRow) if r.name == name), None)
        self.assertIsNotNone(row, f"No SliderRow {name!r}")
        self.assertIsNotNone(row.help_btn, f"{name!r} has no '?' button")
        return row

    def _press_at(self, widget, pos: QPoint) -> None:
        """Post a real left-button press (so the app-level filter sees it) and release."""
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()


class TestSliderRowToggle(_Base):
    """1. SliderRow '?' toggle across the auto-close race."""

    def test_second_click_closes(self):
        row = self._row("shadows")
        QTest.mouseClick(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(row.help_popup is not None and row.help_popup.isVisible(),
                        "popup not open after first click")
        self.assertTrue(row._help_tracker.is_open, "tracker flag not set after open")

        # Simulate the Windows order of the 2nd click: the press auto-closes the popup
        # (hide) BEFORE the release's clicked() runs. The auto-close must not clear the flag.
        row.help_popup.hide()
        self.app.processEvents()
        self.assertTrue(row._help_tracker.is_open,
                        "auto-close (hide) cleared the flag — the release would re-open")

        # The release's clicked() then runs and must close (flag was still 'open').
        row.help_btn.click()
        self.app.processEvents()
        self.assertFalse(row.help_popup is not None and row.help_popup.isVisible(),
                         "popup still visible after second click (frozen help)")
        self.assertFalse(row._help_tracker.is_open, "flag not cleared after close")

    def test_click_away_clears_flag_then_reopens(self):
        row = self._row("shadows")
        QTest.mouseClick(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(row._help_tracker.is_open, "flag not set after open")

        # A press outside the popup and not on the '?' button is a click-away: it clears
        # the flag (the popup auto-closes on the real platform).
        self._press_at(self.dlg.view, QPoint(10, 10))
        self.assertFalse(row._help_tracker.is_open, "click-away did not clear the flag")

        # The next '?' click is then read as an open again.
        QTest.mouseClick(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(row._help_tracker.is_open, "flag not set after re-open")
        self.assertTrue(row.help_popup is not None and row.help_popup.isVisible(),
                        "popup not visible after re-open")

    def test_press_on_button_keeps_flag(self):
        row = self._row("shadows")
        QTest.mouseClick(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(row._help_tracker.is_open, "flag not set after open")

        # A press on the '?' button itself must not clear the flag (the release toggles).
        QTest.mousePress(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(row._help_tracker.is_open,
                        "press on the '?' button cleared the flag")
        QTest.mouseRelease(row.help_btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()


class TestSectionHelpToggle(_Base):
    """2. Section help (general '?') toggle across the auto-close race."""

    def test_second_click_closes(self):
        btn = self.dlg._general_help_btn
        btn.click()
        self.app.processEvents()
        self.assertTrue(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                        "popup not open after first click")
        self.assertEqual(self.dlg._help_tracker.open_value(), "general",
                         "tracker flag not set to 'general' after open")

        # Simulate the Windows order: the press auto-closes the popup (close -> hideEvent
        # -> _on_help_closed) before the release's clicked() runs. The flag must survive.
        self.dlg._help_popup.close()
        self.app.processEvents()
        self.assertEqual(self.dlg._help_tracker.open_value(), "general",
                         "auto-close cleared the flag — the release would re-open")

        # The release's clicked() then runs and must close (flag was still 'general').
        btn.click()
        self.app.processEvents()
        self.assertFalse(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                         "popup still visible after second click (frozen help)")
        self.assertIsNone(self.dlg._help_tracker.open_value(), "flag not cleared after close")

    def test_press_inside_popup_keeps_flag(self):
        btn = self.dlg._general_help_btn
        btn.click()
        self.app.processEvents()
        popup = self.dlg._help_popup
        self.assertIsNotNone(popup)
        self.assertEqual(self.dlg._help_tracker.open_value(), "general")

        # A press inside the open popup (e.g. scrolling) must not clear the flag.
        centre = popup.rect().center()
        QTest.mousePress(popup, Qt.MouseButton.LeftButton, pos=QPoint(int(centre.x()), int(centre.y())))
        self.app.processEvents()
        self.assertEqual(self.dlg._help_tracker.open_value(), "general",
                         "press inside the popup cleared the flag")
        QTest.mouseRelease(popup, Qt.MouseButton.LeftButton, pos=QPoint(int(centre.x()), int(centre.y())))
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
