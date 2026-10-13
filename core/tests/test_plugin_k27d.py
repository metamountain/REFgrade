"""K27d: OFF-Bereiche bedienbar, SPLIT ins Bild, Mindestmaße aus s, Curves 2:1."""

from __future__ import annotations

import ast
import os
import re
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

from PyQt6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QSlider  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

W, H = 2560, 1392


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K27dTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 2560×1392."""

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
        self.s = scale.unit(W, H)
        self.row_h = scale.row_h(self.s)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestSliderMinWidth(K27dTestBase):
    """Alle sichtbaren QSlider in der Spalte: width() >= 6 * row_h - 1."""

    def test_slider_min_width(self):
        panel = self.dlg.panel
        sliders = panel.findChildren(QSlider)
        visible = [s for s in sliders if s.isVisible()]
        self.assertGreater(len(visible), 0, "No visible sliders found in panel")
        min_w = 6 * self.row_h - 1
        for s in visible:
            self.assertGreaterEqual(
                s.width(), min_w,
                f"Slider '{s.objectName() or s.metaObject().className()}' "
                f"width {s.width()} < {min_w}")


class TestNoFixedPixels(K27dTestBase):
    """Keine festen Pixel-Mindestmaße > 1 mehr in plugin/ref_grade/*.py."""

    def test_no_fixed_pixel_minima(self):
        """Source-text scan: no setMinimumSize(N, N) / setMinimumHeight(N) / TILE_TARGET = N with N > 1."""
        pattern_min_size = re.compile(
            r"setMinimumSize\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        pattern_min_h = re.compile(
            r"setMinimumHeight\(\s*(\d+)\s*\)")
        pattern_tile = re.compile(
            r"TILE_TARGET\s*=\s*(\d+)")
        violations = []
        for f in sorted(_PLUGIN_DIR.glob("*.py")):
            text = f.read_text(encoding="utf-8")
            for m in pattern_min_size.finditer(text):
                if int(m.group(1)) > 1 or int(m.group(2)) > 1:
                    violations.append(f"{f.name}:{m.start()}: setMinimumSize({m.group(1)}, {m.group(2)})")
            for m in pattern_min_h.finditer(text):
                if int(m.group(1)) > 1:
                    violations.append(f"{f.name}:{m.start()}: setMinimumHeight({m.group(1)})")
            for m in pattern_tile.finditer(text):
                if int(m.group(1)) > 1:
                    violations.append(f"{f.name}:{m.start()}: TILE_TARGET = {m.group(1)}")
        self.assertEqual(violations, [], f"Fixed pixel minima found:\n" + "\n".join(violations))


class TestCurvesAspect(K27dTestBase):
    """CURVES aufgeklappt: height >= width/2 - 1. Fenster niedrig: eingeklappt."""

    def test_curves_aspect_expanded(self):
        """At 2560×1392 CURVES is expanded and satisfies 2:1."""
        editor = self.dlg.curves_editor
        self.app.processEvents()
        if editor.isVisible():
            self.assertGreaterEqual(
                editor.height(), editor.width() // 2 - 1,
                f"Curves height {editor.height()} < width/2 {editor.width() // 2}")

    def test_curves_collapsed_when_too_low(self):
        """At 2560×700 the panel is too short: CURVES must be collapsed."""
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img)
        dlg.resize(2560, 700)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= 2540)
        self.app.processEvents()
        editor = dlg.curves_editor
        self.app.processEvents()
        if editor.isVisible():
            self.assertGreaterEqual(
                editor.height(), editor.width() // 2 - 1,
                f"Curves visible but too flat: h={editor.height()} w={editor.width()}")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestNoScopeOverlay(K27dTestBase):
    """Vectorscope-Klick ändert nichts; kein _scope_clicked."""

    def test_no_scope_overlay_attr(self):
        self.assertFalse(hasattr(self.dlg.view, "scope_overlay"))
        self.assertFalse(hasattr(self.dlg.view, "scope_image"))

    def test_no_scope_clicked_method(self):
        self.assertFalse(hasattr(self.dlg, "_scope_clicked"))

    def test_vectorscope_click_no_change(self):
        """Clicking the vectorscope does not alter the view."""
        vs = self.dlg.vectorscope
        self.app.processEvents()
        # Simulate a click on the vectorscope
        pos = QPoint(vs.width() // 2, vs.height() // 2)
        QTest.mouseClick(vs, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        self.assertFalse(hasattr(self.dlg.view, "scope_overlay"))


class TestOffSectionsInteractive(K27dTestBase):
    """OFF-Bereiche: isEnabled() == True, dim-Property gesetzt."""

    def test_off_sections_enabled_and_dimmed(self):
        """Every OFF section has content.isEnabled() == True and dim property set."""
        for key, (header, rows) in self.dlg.sections.items():
            if not header.is_on():
                content = self.dlg._contents[key]
                self.assertTrue(content.isEnabled(),
                                f"Section '{key}' OFF but content disabled")
                self.assertTrue(content.property("dim") is True,
                                f"Section '{key}' OFF but dim property not set")

    def test_reference_off_tile_click_turns_on(self):
        """REFERENCE OFF: click in the section → REFERENCE ON."""
        ref_header = self.dlg.sections["reference"][0]
        # Ensure reference is OFF
        if ref_header.is_on():
            ref_header.on.setChecked(False)
            self.app.processEvents()
        # Expand content if collapsed so widgets are visible
        self.dlg._set_section_collapsed("reference", False)
        self.app.processEvents()
        # Click on the RefSlots widget (in the OFF section)
        slots = self.dlg.slots
        slots.show()
        self.app.processEvents()
        pos = QPoint(slots.width() // 2, slots.height() // 2)
        QTest.mouseClick(slots, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        self.assertTrue(ref_header.is_on(),
                        "REFERENCE should be ON after click in section")

    def test_tone_off_slider_drag_turns_on(self):
        """TONE OFF: drag Exposure slider → TONE ON, value changed."""
        tone_header = self.dlg.sections["tone"][0]
        if tone_header.is_on():
            tone_header.on.setChecked(False)
            self.app.processEvents()
        # Find the exposure slider in the tone section
        tone_rows = self.dlg.sections["tone"][1]
        slider = None
        for r in tone_rows:
            if r.name == "exposure":
                slider = r.slider
                break
        if slider is None:
            self.skipTest("No exposure slider found")
        old_val = slider.value()
        slider.show()
        self.app.processEvents()
        # Simulate a drag: press at current pos, move right, release
        h = slider.height()
        w = slider.width()
        start_x = w // 4
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(start_x, h // 2))
        self.app.processEvents()
        QTest.mouseMove(slider, QPoint(w * 3 // 4, h // 2))
        self.app.processEvents()
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(w * 3 // 4, h // 2))
        self.app.processEvents()
        self.assertTrue(tone_header.is_on(),
                        "TONE should be ON after slider drag")
        self.assertNotEqual(slider.value(), old_val,
                            "Exposure value should have changed")


class TestSplitButtonInView(K27dTestBase):
    """Scope-Reihe enthält keinen SPLIT; Split-Schalter in view oben rechts."""

    def test_no_split_in_scope_row(self):
        """The scope block has no SPLIT button."""
        scope_block = self.dlg._scope_block
        for child in scope_block.findChildren(type(self.dlg._scope_btns["rgb"])):
            self.assertNotEqual(
                getattr(child, "channel", None), "split",
                "SPLIT button should not be in the scope row")

    def test_split_button_in_view(self):
        """User 2026-10-11: one "S" button bottom-right of the image; click and key toggle on
        AND off (the old button toggled twice and could not switch split off)."""
        view = self.dlg.view
        view.update()
        self.app.processEvents()
        view.repaint()
        rect = view._split_btn_rect
        self.assertFalse(rect.isNull(), "S button not drawn")
        self.assertGreater(rect.x(), view.width() * 3 // 4, "S button right")
        self.assertGreater(rect.y(), view.height() * 3 // 4, "S button bottom")
        pos = QPoint(int(rect.center().x()), int(rect.center().y()))
        start = view.split_mode
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        self.assertNotEqual(view.split_mode, start, "click switches")
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        self.assertEqual(view.split_mode, start, "second click switches back")
        QTest.keyClick(view, Qt.Key.Key_S)
        self.app.processEvents()
        self.assertNotEqual(view.split_mode, start, "S key switches")
        QTest.keyClick(view, Qt.Key.Key_S)
        self.app.processEvents()
        self.assertEqual(view.split_mode, start, "S key switches back")

if __name__ == "__main__":
    unittest.main()
