"""Tests für K11 (Fenster): Blending/Balance im Bereich Wheels."""

from __future__ import annotations

import unittest

import test_plugin_k9b as k9b    # Fake-Client, Dialog-Aufbau (setzt auch das ref_grade-Paket)


class TestWheelsBlendingBalance(k9b.K9bTestBase):
    """Blending/Balance: Zeilen, /preview, Reset, „Unverändert“."""

    def _row(self, name: str):
        return next(r for r in self.dialog.sections["wheels"][1] if r.name == name)

    def _wheels_on(self) -> None:
        self.dialog.sections["wheels"][0].on.setChecked(True)
        self.app.processEvents()

    def test_rows_exist_in_wheels(self):
        self.assertEqual(self._row("wheel_blending").default, 0.5)   # Lightroom middle (2026-10-11)
        self.assertEqual(self._row("wheel_intensity").default, 1.0)
        self.assertEqual(self._row("wheel_balance").default, -0.3)   # user 2026-10-12: highlights less dominant

    def test_blending_in_preview(self):
        self._wheels_on()
        initial = self.fake.count("/preview")
        self._row("wheel_blending").set_value(0.3)
        self.app.processEvents()
        self.dialog._schedule_preview()
        k9b._wait(self.app, lambda: self.fake.count("/preview") > initial, 2000)
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertAlmostEqual(payload["values"]["wheel_blending"], 0.3, places=4)
        self.assertIn("wheel_balance", payload["values"])

    def test_section_reset_restores_both(self):
        self._wheels_on()
        self._row("wheel_blending").set_value(0.3)
        self._row("wheel_balance").set_value(0.6)
        self.dialog._reset_section("wheels")
        self.assertAlmostEqual(self._row("wheel_blending").value(), 0.5)
        self.assertAlmostEqual(self._row("wheel_balance").value(), -0.3)


if __name__ == "__main__":
    unittest.main()
