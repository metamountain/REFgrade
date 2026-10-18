"""Repro (user 2026-10-18): clicking two help '?' buttons shortly after each other
freezes the UI. Simulates the Windows order: the press on the second '?' auto-closes
the first popup (hide/close) BEFORE the release's clicked() runs."""

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

from PyQt6.QtCore import Qt  # noqa: E402
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

    def _click(self, btn) -> None:
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
        self.app.processEvents()


class TestTwoSliderHelps(_Base):
    """Sub-module help: open slider A's '?', then shortly after slider B's '?'."""

    def test_second_help_opens_first_closes(self):
        a = self._row("wheel_gain")
        b = self._row("wheel_falloff")
        self._click(a.help_btn)
        self.assertTrue(a.help_popup is not None and a.help_popup.isVisible(),
                        "A not open after click")
        self.assertTrue(a._help_tracker.is_open)

        # Windows order: the press on B's button auto-closes A's popup BEFORE B's
        # release's clicked() runs.
        a.help_popup.hide()
        self.app.processEvents()

        self._click(b.help_btn)
        self.assertTrue(b.help_popup is not None and b.help_popup.isVisible(),
                        "B not open after click (frozen?)")
        self.assertFalse(a.help_popup is not None and a.help_popup.isVisible(),
                         "A still open")
        self.assertFalse(a._help_tracker.is_open, "A flag not cleared")
        self.assertTrue(b._help_tracker.is_open, "B flag not set")

    def test_rapid_alternation_stays_consistent(self):
        a = self._row("wheel_gain")
        b = self._row("wheel_falloff")
        # A -> B -> A -> B in quick succession; each open auto-closes the previous.
        self._click(a.help_btn)
        a.help_popup.hide(); self.app.processEvents()
        self._click(b.help_btn)
        b.help_popup.hide(); self.app.processEvents()
        self._click(a.help_btn)
        self.assertTrue(a.help_popup is not None and a.help_popup.isVisible(),
                        "A not re-opened after alternation")
        self.assertFalse(b.help_popup is not None and b.help_popup.isVisible(),
                         "B still open after A re-opened")
        self._click(b.help_btn)
        self.assertTrue(b.help_popup is not None and b.help_popup.isVisible(),
                        "B not re-opened after alternation")


class TestTwoSectionHelps(_Base):
    """Chapter help: open section A's '?', then shortly after section B's '?'."""

    def _sec_btn(self, key: str):
        return self.dlg.sections[key][0].help_btn

    def test_second_section_opens_first_closes(self):
        keys = [k for k in self.dlg.sections if self.dlg.sections[k][0].help_btn is not None]
        self.assertGreaterEqual(len(keys), 2, "need two sections with '?' buttons")
        ka, kb = keys[0], keys[1]
        self._click(self._sec_btn(ka))
        self.assertTrue(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                        "A not open after click")
        self.assertEqual(self.dlg._help_tracker.open_value(), ka)

        # Windows order: the press on B's button auto-closes A's popup (close ->
        # hideEvent -> closed -> _on_help_closed) BEFORE B's release's clicked() runs.
        self.dlg._help_popup.close()
        self.app.processEvents()

        self._click(self._sec_btn(kb))
        self.assertTrue(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                        "B not open after click (frozen?)")
        self.assertEqual(self.dlg._help_tracker.open_value(), kb,
                         "tracker not switched to B")

    def test_section_then_slider_then_section(self):
        keys = [k for k in self.dlg.sections if self.dlg.sections[k][0].help_btn is not None]
        ka = keys[0]
        row = self._row("wheel_gain")
        # section -> slider -> section
        self._click(self._sec_btn(ka))
        self.dlg._help_popup.close(); self.app.processEvents()
        self._click(row.help_btn)
        self.assertTrue(row.help_popup is not None and row.help_popup.isVisible(),
                        "slider help not open after section help")
        row.help_popup.hide(); self.app.processEvents()
        self._click(self._sec_btn(ka))
        self.assertTrue(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                        "section help not re-opened")
        self.assertEqual(self.dlg._help_tracker.open_value(), ka)


if __name__ == "__main__":
    unittest.main()
