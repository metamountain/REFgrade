"""K25d: RGB-Regler breit (QGridLayout), Histogramm-Rampe fix."""

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

from PyQt6.QtCore import QSettings, QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QFontMetrics  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.widgets import Histogram, SliderRow  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K25dTestBase(unittest.TestCase):
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


class TestRgbSliderWidth(K25dTestBase):
    """Jeder RGB-Regler ≥ 0.13 × Spaltenbreite (1920×1032), ≥ 0.11 × (1366×720)."""

    def test_slider_width_1920(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            s = scale.unit(1920, 1032)
            # Find grid SliderRows (label_width=0 → no visible label)
            grid_rows = [r for r in dlg._rgb_rows
                         if r.name.startswith("rgb_") and r.name != "amount_rgb"]
            self.assertGreater(len(grid_rows), 0, "No RGB grid sliders found")
            for r in grid_rows:
                col_w = r.width()
                slider_w = r.slider.width()
                self.assertGreaterEqual(
                    slider_w, 0.13 * col_w,
                    msg=f"Slider {r.name}: {slider_w} < 0.13×{col_w}")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()

    def test_slider_width_1366(self):
        dlg = self._make_dialog(1366, 720)
        try:
            grid_rows = [r for r in dlg._rgb_rows
                         if r.name.startswith("rgb_") and r.name != "amount_rgb"]
            self.assertGreater(len(grid_rows), 0)
            for r in grid_rows:
                col_w = r.width()
                slider_w = r.slider.width()
                self.assertGreaterEqual(
                    slider_w, 0.11 * col_w,
                    msg=f"Slider {r.name}: {slider_w} < 0.11×{col_w}")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


class TestRgbLabels(K25dTestBase):
    """Kein Label-Text breiter als das Label; Kopfzeile Master/Red/Green/Blue y-gleich."""

    def test_no_label_truncated(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            # Find all QLabels in the RGB section
            rgb_w = dlg._rgb_h
            labels = rgb_w.findChildren(QLabel)
            for lbl in labels:
                text = lbl.text()
                if not text:
                    continue
                fm = QFontMetrics(lbl.font())
                text_w = fm.horizontalAdvance(text)
                self.assertLessEqual(
                    text_w, lbl.width(),
                    msg=f"Label '{text}': text {text_w}px > label {lbl.width()}px")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()

    def test_header_row(self):
        dlg = self._make_dialog(1920, 1032)
        try:
            rgb_w = dlg._rgb_h
            labels = rgb_w.findChildren(QLabel)
            headers = {}
            for lbl in labels:
                t = lbl.text()
                if t in ("Master", "Red", "Green", "Blue"):
                    headers[t] = lbl
            self.assertEqual(set(headers.keys()), {"Master", "Red", "Green", "Blue"},
                             msg=f"Headers found: {list(headers.keys())}")
            # All four at same y
            ys = [lbl.mapTo(rgb_w, QPoint(0, 0)).y() for lbl in headers.values()]
            self.assertLessEqual(max(ys) - min(ys), 1,
                                 msg=f"Header y positions differ: {ys}")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


class TestZoneValues(K25dTestBase):
    """K25e: Zonenwert zeigt Prozent (100 %)."""

    def test_zone_percent(self):
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


class TestHistogramPixel(K25dTestBase):
    """Histogramm 400×100 mit Bins [0.5]*32+[0.0]*31+[1.0]: Pixel (300,30) ist Hintergrund."""

    def test_no_ramp_at_300_30(self):
        hist = Histogram()
        hist.setFixedSize(400, 100)
        hist.show()
        self.app.processEvents()
        bins = [0.5] * 32 + [0.0] * 31 + [1.0]
        hist.set_rgb_data({
            "source": {"r": bins},
            "result": {"r": bins},
        })
        self.app.processEvents()
        pm = hist.grab().toImage()
        pixel = pm.pixelColor(300, 30)
        # Background is QColor(20, 20, 20); allow ±10 for antialiasing
        self.assertLess(pixel.red(), 40,
                        msg=f"Pixel (300,30) R={pixel.red()} — not background")
        self.assertLess(pixel.green(), 40,
                        msg=f"Pixel (300,30) G={pixel.green()} — not background")
        self.assertLess(pixel.blue(), 40,
                        msg=f"Pixel (300,30) B={pixel.blue()} — not background")
        hist.close()
        hist.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
