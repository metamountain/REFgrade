"""K25c: ON/OFF switch only dims the content (no collapse, no revert).
Triangle click collapses/expands independently of the ON state."""

import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", pkg)

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from ref_grade.dialog import RefGradeDialog  # noqa: E402


class TestOpenCloseSwitch(unittest.TestCase):

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        self.d = RefGradeDialog(QImage(str(ROOT / "bench" / "pool" / "07.jpg")))
        self.d.resize(1920, 1080)
        self.d.show()
        app.processEvents()
        # K28f: at 1920x1080 TONE may start collapsed for space (K25c "soweit Platz");
        # these tests are about ON/OFF, not the start layout, so open it first.
        self.d._set_section_collapsed("tone", False)
        app.processEvents()

    def tearDown(self):
        self.d.close()

    def _toggle(self, key):
        self.d.sections[key][0].toggle()
        app.processEvents()

    def test_open_switches_on(self):
        """2026-10-11: tone starts OFF; toggle → ON; toggle → OFF again."""
        self.assertFalse(self.d.sections["tone"][0].is_on())
        self._toggle("tone")                              # ON
        self.assertTrue(self.d.sections["tone"][0].is_on())
        self._toggle("tone")                              # OFF again
        self.assertFalse(self.d.sections["tone"][0].is_on())

    def test_close_unchanged_switches_off(self):
        """K25c/K27d: toggle OFF → content visible, dimmed, still operable."""
        self._toggle("tone")                              # ON (starts OFF since 2026-10-11)
        self.assertTrue(self.d.sections["tone"][0].is_on())
        self._toggle("tone")                              # OFF
        self.assertFalse(self.d.sections["tone"][0].is_on())
        content = self.d._contents["tone"]
        self.assertTrue(content.isVisible())
        self.assertTrue(content.isEnabled())          # K27d: OFF stays operable
        self.assertTrue(content.property("dim") is True)

    def test_off_does_not_collapse(self):
        """K25c: turning OFF does not collapse the content."""
        self._toggle("tone")                              # ON (starts OFF since 2026-10-11)
        self._toggle("tone")                              # OFF
        content = self.d._contents["tone"]
        self.assertTrue(content.isVisible(), "OFF must not hide the content")


if __name__ == "__main__":
    unittest.main()
