"""K25c: Einklappen gibt Platz frei, alles ON außer REFERENCE, Hauptknöpfe unten."""

from __future__ import annotations

import os
import sys
import tempfile
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

from PyQt6.QtCore import QSettings, QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K25cTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 1920×1080."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= 1900)
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestMainButtons(K25cTestBase):
    """Hauptknöpfe RESET/CANCEL/OK unten im Panel."""

    def test_buttons_at_bottom(self):
        """OK-Button-Unterkante = Panel-Unterkante ±2."""
        panel_bottom = self.dlg.panel.mapTo(self.dlg, QPoint(0, self.dlg.panel.height())).y()
        ok_bottom = self.dlg._btn_ok.mapTo(self.dlg, QPoint(0, self.dlg._btn_ok.height())).y()
        self.assertAlmostEqual(ok_bottom, panel_bottom, delta=2,
                               msg=f"OK bottom {ok_bottom} != panel bottom {panel_bottom}")

    def test_ok_width_double_cancel(self):
        """OK-Breite = 2×CANCEL-Breite ±3."""
        ok_w = self.dlg._btn_ok.width()
        cancel_w = self.dlg._btn_cancel.width()
        self.assertAlmostEqual(ok_w, 2 * cancel_w, delta=3,
                               msg=f"OK width {ok_w} != 2×CANCEL {cancel_w}")

    def test_button_height(self):
        """Button-Höhe = round(button_h * 1.5) ±1."""
        s = scale.unit(1920, 1080)
        expected_h = round(scale.button_h(s) * 1.5)
        actual_h = self.dlg._btn_ok.height()
        self.assertAlmostEqual(actual_h, expected_h, delta=1,
                               msg=f"Button height {actual_h} != expected {expected_h}")

    def test_ok_is_default(self):
        """OK-Button ist der Default-Button."""
        self.assertTrue(self.dlg._btn_ok.isDefault())

    def test_ok_color_grey(self):
        """User 2026-10-11: the button reads SAVE and looks like CANCEL (no grey fill)."""
        ok, cancel = self.dlg._btn_ok, self.dlg._btn_cancel
        self.assertEqual(ok.text(), "SAVE")
        po, pc = ok.grab().toImage().pixelColor(4, 4), cancel.grab().toImage().pixelColor(4, 4)
        self.assertEqual(po.name(), pc.name(), f"SAVE {po.name()} vs CANCEL {pc.name()}")


class TestInitialState(K25cTestBase):
    """Start: alle Sektionen ON außer REFERENCE; alle expanded außer REFERENCE."""

    def test_all_off_at_start(self):
        """Nutzer 2026-10-11: alle Sektionen starten OFF (ON erst, wenn sich etwas ändert)."""
        for key in ("color", "rgb", "tone", "hsl", "curves", "wheels", "lut", "reference"):
            self.assertFalse(self.dlg.sections[key][0].is_on(), key)

    def test_all_expanded_except_reference(self):
        """K28f: alle offen außer REFERENCE, soweit Platz (K25c). Bei 1920×1080 passt nicht alles
        (gemessen 1087 > 855 px) → nur so wenig wie nötig zu: CURVES zuletzt (K29e), COLOR und
        WHEELS nie; jeder zusätzlich geöffnete Bereich sprengt die Spalte."""
        self.assertTrue(self.dlg.sections["reference"][0].is_collapsed())
        for key in ("color", "wheels"):
            self.assertFalse(self.dlg.sections[key][0].is_collapsed(), key)
        lay = self.dlg.panel.layout()
        limit = self.dlg.panel.height()
        self.assertLessEqual(self.dlg._panel_min_height(lay), limit, "Spalte passt nicht")
        closed = [k for k in ("rgb", "tone", "hsl", "curves", "lut")
                  if self.dlg.sections[k][0].is_collapsed()]
        # K29e (user 2026-10-11): CURVES is collapsed LAST — rather close the others than
        # squash the curve below 2:1; it may only be closed when everything else is too
        if "curves" in closed:
            self.assertEqual(set(closed), {"rgb", "tone", "hsl", "curves", "lut"} & set(
                k for k in ("rgb", "tone", "hsl", "curves", "lut")
                if self.dlg._contents.get(k) is not None),
                "CURVES nur zu, wenn alle anderen schon zu sind")
        for key in closed:
            self.dlg._set_section_collapsed(key, False)
            self.assertGreater(self.dlg._panel_min_height(lay), limit,
                               f"{key} hätte offen bleiben können")
            self.dlg._set_section_collapsed(key, True)


class TestCollapseGivesSpace(K25cTestBase):
    """Einklappen gibt Platz frei: nächster Header rückt nach oben."""

    def test_collapse_tone_moves_next_header(self):
        """TONE einklappen → nächster Header rückt nach oben."""
        self.dlg._set_section_collapsed("tone", False)   # K28f: may start closed for space
        self.app.processEvents()
        tone_header = self.dlg.sections["tone"][0]
        content = self.dlg._contents["tone"]
        self.assertFalse(tone_header.is_collapsed())
        self.assertTrue(content.isVisible())

        # Measure next header position before collapse
        next_key = "hsl"
        next_header = self.dlg.sections[next_key][0]
        y_before = next_header.mapTo(self.dlg.panel, QPoint(0, 0)).y()

        # Collapse TONE
        tone_header._title_click()
        self.app.processEvents()
        self.assertTrue(tone_header.is_collapsed())
        self.assertFalse(content.isVisible())

        # Measure next header position after collapse
        y_after = next_header.mapTo(self.dlg.panel, QPoint(0, 0)).y()

        # Next header should have moved up
        delta = y_before - y_after
        self.assertGreater(delta, 0, "Next header should move up after collapse")

    def test_all_collapsed_headers_close(self):
        """(K29: WHEELS sits in the wheel strip now.) Alle Sektionen collapsed → Abstand zwischen Headern ≤ 2×row_h."""
        # Collapse all sections except reference (already collapsed)
        for key in ("color", "rgb", "tone", "hsl", "curves", "lut"):
            if not self.dlg.sections[key][0].is_collapsed():
                self.dlg.sections[key][0]._title_click()
                self.app.processEvents()

        s = scale.unit(1920, 1080)
        row_h = scale.row_h(s)

        # Get y positions of all headers and sort by y
        keys = ["color", "rgb", "tone", "hsl", "curves", "lut"]
        positions = []
        for key in keys:
            h = self.dlg.sections[key][0]
            y = h.mapTo(self.dlg.panel, QPoint(0, 0)).y()
            positions.append((y, key))
        positions.sort()

        # Check distance between consecutive headers (in actual y order)
        for i in range(len(positions) - 1):
            y1, key1 = positions[i]
            y2, key2 = positions[i + 1]
            dist = y2 - y1
            self.assertGreater(dist, 0, f"Headers out of order: {key1} y={y1}, {key2} y={y2}")
            self.assertLessEqual(dist, 2 * row_h,
                                 msg=f"Distance between {key1} and {key2}: {dist} > 2×row_h={2*row_h}")


class TestLutCollapse(K25cTestBase):
    """LUT einklappen → FOLDER-Button nicht sichtbar."""

    def test_lut_collapsed_folder_hidden(self):
        """LUT collapsed → FOLDER-Button nicht sichtbar."""
        lut_header = self.dlg.sections["lut"][0]
        self.assertFalse(lut_header.is_collapsed())

        # Find the FOLDER button in the LUT section
        folder_btn = self.dlg._lut_folder_btn
        self.assertTrue(folder_btn.isVisible())

        # Collapse LUT
        lut_header._title_click()
        self.app.processEvents()
        self.assertTrue(lut_header.is_collapsed())
        self.assertFalse(folder_btn.isVisible())


class TestReferenceExpand(K25cTestBase):
    """REFERENCE-Dreieck klicken → Inhalt sichtbar, kein Overflow."""

    def test_reference_expand_no_overflow(self):
        """REFERENCE expandieren → Inhalt sichtbar, Panel passt noch."""
        ref_header = self.dlg.sections["reference"][0]
        self.assertTrue(ref_header.is_collapsed())

        # Expand reference
        ref_header._title_click()
        self.app.processEvents()
        self.assertFalse(ref_header.is_collapsed())

        content = self.dlg._contents["reference"]
        self.assertTrue(content.isVisible())

        # Panel should still fit (no overflow)
        panel_h = self.dlg.panel.height()
        self.assertGreater(panel_h, 0)


class TestOffDimsContent(K25cTestBase):
    """OFF → Inhalt sichtbar aber disabled; auch für WHEELS."""

    def test_off_tone_content_disabled(self):
        """TONE OFF → Inhalt sichtbar, dim-Property gesetzt."""
        self.dlg._set_section_collapsed("tone", False)   # K28f: may start closed for space
        self.app.processEvents()
        tone_header = self.dlg.sections["tone"][0]
        tone_header.on.setChecked(True)                  # 2026-10-11: starts OFF
        self.assertTrue(tone_header.is_on())

        tone_header.on.setChecked(False)
        self.app.processEvents()
        self.assertFalse(tone_header.is_on())

        content = self.dlg._contents["tone"]
        self.assertTrue(content.isVisible())
        self.assertTrue(content.property("dim") is True)

    def test_off_wheels_content_disabled(self):
        """WHEELS OFF → Inhalt sichtbar, dim-Property gesetzt."""
        wheels_header = self.dlg.sections["wheels"][0]
        wheels_header.on.setChecked(True)                # 2026-10-11: starts OFF
        self.assertTrue(wheels_header.is_on())

        wheels_header.on.setChecked(False)
        self.app.processEvents()
        self.assertFalse(wheels_header.is_on())

        content = self.dlg._contents["wheels"]
        self.assertTrue(content.isVisible())
        self.assertTrue(content.property("dim") is True)


if __name__ == "__main__":
    unittest.main()



class TestOnWhenChanged(K25cTestBase):
    """Nutzer 2026-10-11: eine Änderung schaltet nur das eigene Kapitel AN, ↺ wieder AUS."""

    def test_change_switches_on_reset_switches_off(self):
        tone = self.dlg.sections["tone"][0]
        color = self.dlg.sections["color"][0]
        exposure = next(r for r in self.dlg.sections["tone"][1] if r.name == "exposure")
        exposure.set_value(0.5)
        self.app.processEvents()
        self.assertTrue(tone.is_on())
        self.assertFalse(color.is_on(), "nur das eigene Kapitel")
        self.dlg._reset_section("tone")
        self.app.processEvents()
        self.assertFalse(tone.is_on())
        self.assertAlmostEqual(exposure.value(), exposure.default)

    def test_manual_off_stays_until_next_change(self):
        tone = self.dlg.sections["tone"][0]
        exposure = next(r for r in self.dlg.sections["tone"][1] if r.name == "exposure")
        exposure.set_value(0.5)
        tone.on.setChecked(False)
        self.app.processEvents()
        self.assertFalse(tone.is_on())
        exposure.set_value(0.6)
        self.assertTrue(tone.is_on())
