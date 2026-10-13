"""K25e: RGB Master geht nicht verloren, Zonen ehrlich beschriftet."""

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
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K25eTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self, W: int, H: int) -> RefGradeDialog:
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        dlg = RefGradeDialog(img)
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        return dlg

    def tearDown(self):
        self.app.processEvents()


class TestZoneLabel(K25eTestBase):
    """Beschriftung 'Channels act on' vorhanden."""

    def test_channels_act_on_label(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            rgb_w = dlg._rgb_h
            labels = rgb_w.findChildren(QLabel)
            texts = [lbl.text() for lbl in labels]
            self.assertIn("Channels act on", texts,
                          msg=f"Label 'Channels act on' not found. Labels: {texts}")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


class TestZonePercentDisplay(K25eTestBase):
    """Zonenwert zeigt '100 %' (Default 1.0 → 100 %)."""

    def test_zone_default_100_percent(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            zone_names = ("rgb_use_shadows", "rgb_use_midtones", "rgb_use_highlights")
            for n in zone_names:
                r = next((r for r in dlg._rgb_rows if r.name == n), None)
                self.assertIsNotNone(r, f"Zone slider {n} not found")
                text = r.value_label.text()
                self.assertEqual(text, "100 %",
                                 msg=f"Zone {n}: value '{text}' not '100 %'")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


class TestZoneTooltip(K25eTestBase):
    """Tooltip auf 'Channels act on' vorhanden."""

    def test_tooltip_on_zone_label(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            rgb_w = dlg._rgb_h
            labels = rgb_w.findChildren(QLabel)
            zone_lbl = next((lbl for lbl in labels if lbl.text() == "Channels act on"), None)
            self.assertIsNotNone(zone_lbl, "Label 'Channels act on' not found")
            tip = zone_lbl.toolTip()
            self.assertIn("Master always acts everywhere", tip,
                          msg=f"Tooltip missing 'Master always acts everywhere': '{tip}'")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
