"""K29e: Curves nie flacher als 2:1 (Breite:Höhe)."""

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
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402

W, H = 2560, 1392
OTHER_SECTIONS = ("color", "rgb", "tone", "hsl", "lut", "reference")


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class CurvesRatioBase(unittest.TestCase):
    """Offscreen dialog at 2560×1392; parametrize image shape."""

    img_w: int = 400
    img_h: int = 300

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(self.img_w, self.img_h, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestCurvesRatioPortrait(CurvesRatioBase):
    """Hochformat: Bild 300×400."""

    img_w, img_h = 300, 400


class TestCurvesRatioLandscape(CurvesRatioBase):
    """Querformat: Bild 400×300."""

    img_w, img_h = 400, 300


class _RatioMixin:
    """Shared test logic for both orientations."""

    def test_curves_ratio_on_open(self):
        """After opening CURVES, editor height >= width//2 - 2."""
        dlg = self.dlg
        dlg._section_collapsed_changed("curves", False)
        self.app.processEvents()
        editor = dlg.curves_editor
        if editor.isVisible():
            self.assertGreaterEqual(
                editor.height(), editor.width() // 2 - 2,
                f"Curves squashed: h={editor.height()} w={editor.width()}",
            )

    def test_curves_ratio_with_other_sections_open(self):
        """While CURVES is visible, expanding other sections keeps 2:1 and no overflow."""
        dlg = self.dlg
        dlg._section_collapsed_changed("curves", False)
        self.app.processEvents()
        editor = dlg.curves_editor
        if not editor.isVisible():
            self.skipTest("CURVES collapsed (window too small)")
        for key in OTHER_SECTIONS:
            dlg._section_collapsed_changed(key, False)
            self.app.processEvents()
            if not editor.isVisible():
                break  # CURVES was auto-collapsed; stop checking
            self.assertGreaterEqual(
                editor.height(), editor.width() // 2 - 2,
                f"Curves squashed after opening {key}: "
                f"h={editor.height()} w={editor.width()}",
            )
            min_h = dlg._panel_min_height(dlg.panel.layout())
            self.assertLessEqual(
                min_h, dlg.panel.height(),
                f"Panel overflow after opening {key}: "
                f"min={min_h} panel={dlg.panel.height()}",
            )


# Mix in the shared tests
TestCurvesRatioPortrait.test_curves_ratio_on_open = _RatioMixin.test_curves_ratio_on_open
TestCurvesRatioPortrait.test_curves_ratio_with_other_sections_open = _RatioMixin.test_curves_ratio_with_other_sections_open
TestCurvesRatioLandscape.test_curves_ratio_on_open = _RatioMixin.test_curves_ratio_on_open
TestCurvesRatioLandscape.test_curves_ratio_with_other_sections_open = _RatioMixin.test_curves_ratio_with_other_sections_open
