"""Hilfe über dem Bild (Nutzer 2026-10-11): schwarzes Quadrat mittig über dem Bild, für jeden ?-Knopf."""
from __future__ import annotations

import os
import sys
import time
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402


class TestHelpOverPicture(unittest.TestCase):
    """Hilfe = schwarzes Quadrat mittig über dem Bild (Nutzer 2026-10-11), für jeden ?-Knopf."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_square_centred_over_view(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img)
        dlg.resize(1920, 1080)
        dlg.show()
        t = time.time()
        while time.time() - t < 1.0:
            self.app.processEvents()
        try:
            for key in ("tone", "wheels", "general"):
                dlg._on_help_requested(key)
                self.app.processEvents()
                pop = dlg._help_popup
                self.assertIsNotNone(pop, key)
                self.assertEqual(pop.width(), pop.height(), f"{key}: Quadrat")
                vc = dlg.view.mapToGlobal(dlg.view.rect().center())
                pc = pop.mapToGlobal(pop.rect().center())
                self.assertLessEqual(abs(vc.x() - pc.x()), 2, key)
                self.assertLessEqual(abs(vc.y() - pc.y()), 2, key)
                pop.close()
                self.app.processEvents()
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
