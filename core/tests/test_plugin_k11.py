"""Tests für K11 (Fenster): Gain/Falloff im Bereich Wheels (2026-10-18: war Blending/Balance)."""

from __future__ import annotations

import unittest

try:
    import test_plugin_k9b as k9b    # Fake-Client, Dialog-Aufbau (setzt auch das ref_grade-Paket)
except ModuleNotFoundError:   # als tests.test_plugin_k11 geladen (tests/ nicht auf sys.path)
    import tests.test_plugin_k9b as k9b


class TestWheelsGainFalloff(k9b.K9bTestBase):
    """Gain/Falloff: Zeilen, /preview, Reset, „Unverändert“."""

    def _row(self, name: str):
        return next(r for r in self.dialog.sections["wheels"][1] if r.name == name)

    def _wheels_on(self) -> None:
        self.dialog.sections["wheels"][0].on.setChecked(True)
        self.app.processEvents()

    def test_rows_exist_in_wheels(self):
        self.assertEqual(self._row("wheel_gain").default, 0.0)        # 2026-10-18: Kerngröße, Default 0
        self.assertEqual(self._row("wheel_falloff").default, 0.5)     # 2026-10-18: Blur-Breite, Default 0.5

    def test_no_intensity_slider(self):
        """2026-10-18: wheel_intensity entfernt (doppelt mit Amount, das bis 200 % geht)."""
        names = [r.name for r in self.dialog.sections["wheels"][1]]
        self.assertNotIn("wheel_intensity", names)
        # Amount reicht jetzt bis 200 % (2.0): slider-Schritte = (2.0-0.0)/0.01
        self.assertEqual(self._row("amount_wheels").slider.maximum(), 200)

    def test_gain_in_preview(self):
        self._wheels_on()
        initial = self.fake.count("/preview")
        self._row("wheel_gain").set_value(0.3)
        self.app.processEvents()
        self.dialog._schedule_preview()
        k9b._wait(self.app, lambda: self.fake.count("/preview") > initial, 2000)
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertAlmostEqual(payload["values"]["wheel_gain"], 0.3, places=4)
        self.assertIn("wheel_falloff", payload["values"])

    def test_section_reset_restores_both(self):
        self._wheels_on()
        self._row("wheel_gain").set_value(0.3)
        self._row("wheel_falloff").set_value(0.2)
        self.dialog._reset_section("wheels")
        self.assertAlmostEqual(self._row("wheel_gain").value(), 0.0)
        self.assertAlmostEqual(self._row("wheel_falloff").value(), 0.5)


if __name__ == "__main__":
    unittest.main()
