"""WAVE: Luma-Waveform im Neon-Look (Nutzer 2026-10-11, Vorbild Kdenlive „Paint mode: White“)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.scope import WAVE_COLOR, luma_waveform, parade_neon_all  # noqa: E402

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
from ref_grade.widgets import Waveform  # noqa: E402


class TestLumaWaveform(unittest.TestCase):
    def test_left_dark_right_bright(self):
        """Links schwarz, rechts weiß → Spalte 0 hat ihre Dichte bei Stufe 0, Spalte 255 bei 255."""
        x = torch.zeros(1, 32, 512, 3)     # ≥ 256 px breit: jede der 256 Spalten belegt
        x[:, :, 256:] = 1.0
        wf = luma_waveform(x)
        self.assertEqual(tuple(wf.shape), (256, 256))
        self.assertEqual(int(wf[:, 0].argmax()), 0)
        self.assertEqual(int(wf[:, 255].argmax()), 255)

    def test_luma_weights(self):
        """Reines Grün (Y = 0,7152) landet auf Stufe 183 (= int(0,7152·256))."""
        x = torch.zeros(1, 8, 8, 3)
        x[..., 1] = 1.0
        self.assertEqual(int(luma_waveform(x)[:, 0].argmax()), 183)

    def test_neon_white_on_dense(self):
        """Dichte Spur glüht fast weiß (alle Kanäle hoch), leere Fläche bleibt schwarz."""
        x = torch.full((1, 64, 64, 3), 0.5)
        rgb = parade_neon_all({"l": luma_waveform(x)}, {"l": WAVE_COLOR})["l"]
        dense = rgb[255 - 128, 100]
        self.assertGreater(float(dense.min()), 0.7)
        self.assertLess(float(rgb[20, 100].max()), 0.02)


class TestWaveView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_widget_draws_image(self):
        tmp = tempfile.mkdtemp(prefix="wave_")
        img = QImage(256, 256, QImage.Format.Format_RGB32)
        img.fill(QColor(200, 200, 200))
        path = os.path.join(tmp, "w.bmp")
        img.save(path, "BMP")
        w = Waveform()
        w.resize(400, 150)
        w.set_data(path)
        w.show()
        self.app.processEvents()
        px = w.grab().toImage().pixelColor(200, 30)    # keine Gitterlinie
        self.assertGreater(px.red(), 150)

    def test_wave_button_switches_view(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img)
        dlg.resize(1920, 1080)
        dlg.show()
        deadline = time.time() + 3
        while time.time() < deadline:
            self.app.processEvents()
        try:
            dlg._btn_wave.click()
            self.app.processEvents()
            self.assertTrue(dlg._wave.isVisible())
            self.assertFalse(dlg._parade.isVisible())
            self.assertFalse(dlg.histogram.isVisible())
            self.assertTrue(dlg._btn_wave.isChecked())
            # Knopf neben PARADE, ohne Überlappung
            self.assertGreaterEqual(dlg._btn_wave.x(), dlg._btn_parade.geometry().right())
            self.assertGreater(dlg._btn_wave.width(), 0)
            dlg._btn_hist.click()
            self.app.processEvents()
            self.assertFalse(dlg._wave.isVisible())
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


class TestLumaSwitch(unittest.TestCase):
    """L ist ein eigener Schalter: Luminanz als Ghost hinter den gewählten Kanälen (2026-10-11)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_l_toggles_ghost_and_keeps_channels(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img)
        dlg.resize(1920, 1080)
        dlg.show()
        self.app.processEvents()
        try:
            self.assertTrue(dlg.histogram.show_luma)              # Standard: an
            dlg._scope_btn_clicked("r")
            dlg._scope_btn_clicked("l")
            self.assertFalse(dlg.histogram.show_luma)
            self.assertEqual(dlg.histogram.visible_channels, {"r"})   # R bleibt gewählt
            self.assertFalse(dlg._scope_btns["l"].is_active())
            self.assertTrue(dlg._scope_btns["r"].is_active())
            dlg._scope_btn_clicked("l")
            self.assertTrue(dlg.histogram.show_luma)
            self.assertEqual(dlg.histogram.visible_channels, {"r"})
            dlg._scope_btn_clicked("l")                            # aus → gespeichert
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()
        self.assertFalse(QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).value(
            "scope_luma", True, type=bool))


class TestHistogramWaveAndPeak(unittest.TestCase):
    """Wave hinter dem Histogramm (Stufe B, 0,35) nur bei L an; Clipping-Spitze setzt nicht die Höhe."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _hist(self, show_luma: bool):
        from ref_grade.widgets import Histogram
        tmp = tempfile.mkdtemp(prefix="hw_")
        img = QImage(256, 256, QImage.Format.Format_RGB32)
        img.fill(QColor(255, 255, 255))
        path = os.path.join(tmp, "w.bmp")
        img.save(path, "BMP")
        h = Histogram()
        h.resize(400, 150)
        h.show_luma = show_luma
        h.set_rgb_data({"result": {c: [0.0] * 128 for c in "rgbl"}})
        h.set_wave(path)
        h.show()
        self.app.processEvents()
        return h, h.grab().toImage().pixelColor(170, 20)    # not on the 50 % grid line

    def test_wave_behind_only_with_l(self):
        from ref_grade.widgets import Histogram
        self.assertAlmostEqual(Histogram.WAVE_OPACITY, 0.35)
        _, on = self._hist(True)
        _, off = self._hist(False)
        self.assertGreater(on.red(), off.red() + 60)       # weiße Wave sichtbar, gedämpft
        self.assertLess(on.red(), 140)                      # 0,35 deckt nicht voll

    def test_clip_spike_does_not_set_scale(self):
        from ref_grade.widgets import Histogram
        h = Histogram()
        vals = [0.01] * 128
        vals[64] = 0.04
        vals[-2] = 0.5                                       # Clipping-Spitze nahe Weiß
        h.set_rgb_data({"result": {"r": vals, "g": vals, "b": vals}})
        self.assertAlmostEqual(h._peak(), 0.04 * 1.05, places=6)


if __name__ == "__main__":
    unittest.main()
