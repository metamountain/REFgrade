"""Every visible QAbstractButton in the RefGrade dialog has a tooltip (≥ 12 chars, ≠ text)."""

from __future__ import annotations

import os
import sys
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
from PyQt6.QtWidgets import QApplication, QAbstractButton  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

W, H = 2560, 1400


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class TestTooltipsEverywhere(unittest.TestCase):
    """Every visible QAbstractButton in the dialog has toolTip().strip() ≥ 12 chars and ≠ text()."""

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
        # Expand all sections so all buttons are visible
        for key in list(self.dlg.sections.keys()):
            self.dlg._set_section_collapsed(key, False)
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

    def test_all_buttons_have_tooltips(self):
        """Every visible QAbstractButton has toolTip().strip() >= 12 chars and != text()."""
        missing = []
        for btn in self.dlg.findChildren(QAbstractButton):
            if not btn.isVisibleTo(self.dlg):
                continue
            tip = btn.toolTip().strip()
            name = btn.objectName() or btn.metaObject().className()
            text = btn.text().strip()
            if len(tip) < 12:
                missing.append(f"  {name}: text='{text}' tip='{tip}' (len={len(tip)})")
            elif tip == text:
                missing.append(f"  {name}: text='{text}' tip='{tip}' (same as text)")
        self.assertEqual(
            missing, [],
            f"{len(missing)} button(s) missing a proper tooltip:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
