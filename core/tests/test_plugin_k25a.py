"""Tests für Paket K25a: Eine Einheit s, Schrift und Höhen wachsen mit dem Bildschirm."""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ref_grade import scale  # noqa: E402


# ── Pure function tests (no Qt needed) ───────────────────────────────────────

class TestScaleFunctions(unittest.TestCase):
    """Sollwerte aus der Spec, exakt."""

    def test_unit(self):
        self.assertEqual(scale.unit(1920, 720), 144.0)
        self.assertEqual(scale.unit(1920, 1032), 206.4)
        self.assertEqual(scale.unit(1920, 1152), 230.4)
        self.assertEqual(scale.unit(2560, 1392), 278.4)

    def test_unit_portrait_limited(self):
        # Portrait: W/6 limits (1080/6=180 < 1920/5=384)
        self.assertEqual(scale.unit(1080, 1920), 180.0)

    def test_base_px(self):
        self.assertEqual(scale.base_px(scale.unit(1920, 720)), 9)
        self.assertEqual(scale.base_px(scale.unit(1920, 1032)), 10)
        self.assertEqual(scale.base_px(scale.unit(1920, 1152)), 11)
        self.assertEqual(scale.base_px(scale.unit(2560, 1392)), 13)

    def test_base_px_clamp_low(self):
        # Very small screen: clamp to 10
        self.assertEqual(scale.base_px(scale.unit(600, 500)), 9)

    def test_base_px_clamp_high(self):
        # Very large screen: clamp to 18
        self.assertEqual(scale.base_px(scale.unit(5000, 5000)), 16)

    def test_role_px_small(self):
        self.assertEqual(scale.role_px("small", scale.unit(1920, 1032)), 9)

    def test_role_px_base(self):
        self.assertEqual(scale.role_px("base", scale.unit(1920, 1032)), 10)

    def test_role_px_title(self):
        self.assertEqual(scale.role_px("title", scale.unit(2560, 1392)), 14)

    def test_role_px_big(self):
        self.assertEqual(scale.role_px("big", scale.unit(1920, 1032)), 20)

    def test_row_h(self):
        self.assertEqual(scale.row_h(scale.unit(1920, 1032)), 17)
        self.assertEqual(scale.row_h(scale.unit(2560, 1392)), 22)

    def test_button_h(self):
        self.assertEqual(scale.button_h(scale.unit(1920, 1032)), 19)
        self.assertEqual(scale.button_h(scale.unit(2560, 1392)), 25)

    def test_value_w(self):
        self.assertEqual(scale.value_w(scale.unit(1920, 1032)), 34)
        self.assertEqual(scale.value_w(scale.unit(2560, 1392)), 44)


# ── Source code grep tests ───────────────────────────────────────────────────

class TestNoNumericFont(unittest.TestCase):
    """Kein Aufruf font(<Zahl>) mehr in dialog.py, widgets.py, curves_ui.py."""

    def _check_file(self, name: str):
        path = _PLUGIN_DIR / name
        text = path.read_text(encoding="utf-8")
        # Match style.font( followed by a digit
        matches = re.findall(r"style\.font\(\s*\d", text)
        self.assertEqual(matches, [], f"{name}: numeric font() calls remain: {matches}")

    def test_dialog(self):
        self._check_file("dialog.py")

    def test_widgets(self):
        self._check_file("widgets.py")

    def test_curves_ui(self):
        self._check_file("curves_ui.py")


class TestNoNumericFixedHeight(unittest.TestCase):
    """Kein setFixedHeight(<Zahl>) außer (0) und (1)."""

    def _check_file(self, name: str):
        path = _PLUGIN_DIR / name
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"setFixedHeight\((\d+)\)", text):
            val = int(m.group(1))
            self.assertIn(val, (0, 1),
                          f"{name}: setFixedHeight({val}) found, only 0 and 1 allowed")

    def test_dialog(self):
        self._check_file("dialog.py")

    def test_widgets(self):
        self._check_file("widgets.py")


# ── Offscreen dialog test ────────────────────────────────────────────────────

class TestDialogResponsiveFont(unittest.TestCase):
    """Dialog offscreen: Pixelgröße der Schrift wächst mit der Fensterhöhe."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self, w: int, h: int):
        from PyQt6.QtGui import QImage, QColor
        from ref_grade import style
        from ref_grade.dialog import RefGradeDialog
        # Reset scale to default before creating
        style.set_scale(scale.unit(w, h))
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        d = RefGradeDialog(image=img)
        d.resize(w, h)
        d.show()
        self.app.processEvents()
        return d

    def test_font_grows_with_height(self):
        from ref_grade import style
        from PyQt6.QtWidgets import QLabel

        # At 1920×1032
        d1 = self._make_dialog(1920, 1032)
        s1 = scale.unit(1920, 1032)
        base1 = scale.base_px(s1)
        # Find a SliderRow label
        labels = d1.findChildren(QLabel)
        slider_label = None
        for lab in labels:
            if lab.property("font_role") == "base" and lab.text() in (
                    "Light", "Color", "Exposure", "Contrast"):
                slider_label = lab
                break
        self.assertIsNotNone(slider_label, "No slider label with font_role='base' found")
        px1 = slider_label.font().pixelSize()
        self.assertEqual(px1, base1)
        d1.close()

        # At 2560×1392
        d2 = self._make_dialog(2560, 1392)
        s2 = scale.unit(2560, 1392)
        base2 = scale.base_px(s2)
        labels2 = d2.findChildren(QLabel)
        slider_label2 = None
        for lab in labels2:
            if lab.property("font_role") == "base" and lab.text() in (
                    "Light", "Color", "Exposure", "Contrast"):
                slider_label2 = lab
                break
        self.assertIsNotNone(slider_label2, "No slider label with font_role='base' found (2560)")
        px2 = slider_label2.font().pixelSize()
        self.assertEqual(px2, base2)
        d2.close()

        # 2560×1392 must be larger than 1920×1032
        self.assertGreater(px2, px1)


if __name__ == "__main__":
    unittest.main()
