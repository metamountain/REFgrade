"""Tests für Paket K19: Feinere Regler — relatives Ziehen, Gain, Shift, Mausrad."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import SHORT_TRACK_PX, SliderRow, _RelativeSlider  # noqa: E402

_ALIVE: list = []


def _app() -> QApplication:
    a = QApplication.instance()
    if a is None:
        a = QApplication(sys.argv)
    return a


class K19Base(unittest.TestCase):
    """Base: offscreen QApplication, keeps widgets alive."""

    def setUp(self):
        self.app = _app()

    def tearDown(self):
        for w in _ALIVE:
            try:
                w.close()
                w.deleteLater()
            except RuntimeError:
                pass
        _ALIVE.clear()
        self.app.processEvents()

    def make_slider(self, lo: float, hi: float, step: float, width: int,
                    height: int = 18) -> tuple[SliderRow, _RelativeSlider]:
        row = SliderRow("test", lo, hi, (lo + hi) / 2, step, label_width=0)
        _ALIVE.append(row)
        row.show()
        self.app.processEvents()
        slider = row.slider
        slider.setFixedSize(width, height)
        self.app.processEvents()
        return row, slider

    def drag(self, slider: _RelativeSlider, start_x: float, end_x: float,
             shift: bool = False) -> None:
        """Simulate press at start_x, move to end_x, release. Shift held during move."""
        mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, mods, QPoint(int(start_x), 9))
        self.app.processEvents()
        steps = 5
        for i in range(1, steps + 1):
            x = int(start_x + (end_x - start_x) * i / steps)
            ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(x, 9), QPointF(x, 9),
                             Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, mods)
            self.app.sendEvent(slider, ev)
            self.app.processEvents()
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, mods, QPoint(int(end_x), 9))
        self.app.processEvents()

    def wheel(self, slider: _RelativeSlider, delta: int = 120, shift: bool = False) -> None:
        """Simulate a wheel event on the slider."""
        mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        center = QPointF(slider.width() / 2, slider.height() / 2)
        ev = QWheelEvent(center, center, QPoint(0, delta), QPoint(0, delta),
                         Qt.MouseButton.NoButton, mods,
                         Qt.ScrollPhase.NoScrollPhase, False)
        slider.wheelEvent(ev)
        self.app.processEvents()


class TestLongSlider(K19Base):
    """Langer Regler (Spur >= 200 px): Ziehen um 50 % der Spurbreite → 50 % ± 2 % des Bereichs."""

    def test_drag_50_percent(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        track_w = slider._track_width()
        self.assertGreaterEqual(track_w, 200, f"Track width {track_w} < 200")

        slider.setValue(round(0.5 * (1.0 / 0.001)))
        start_val = row.value()

        drag_dist = track_w * 0.5
        self.drag(slider, track_w * 0.25, track_w * 0.25 + drag_dist)

        end_val = row.value()
        delta = end_val - start_val
        expected = 0.5 * 1.0
        self.assertAlmostEqual(delta, expected, delta=0.02,
                               msg=f"Δ={delta:.4f}, expected {expected:.4f} ± 0.02")


class TestShortSlider(K19Base):
    """Kurzer Regler (Spur < 120 px): gleiche Ziehstrecke → halbe Änderung."""

    def test_drag_half_gain(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=100)
        track_w = slider._track_width()
        self.assertLess(track_w, SHORT_TRACK_PX, f"Track width {track_w} >= {SHORT_TRACK_PX}")

        slider.setValue(round(0.5 * (1.0 / 0.001)))
        start_val = row.value()

        drag_dist = track_w * 0.5
        self.drag(slider, track_w * 0.25, track_w * 0.25 + drag_dist)

        end_val = row.value()
        delta = end_val - start_val
        expected = 0.5 * 1.0 * 0.5
        self.assertAlmostEqual(delta, expected, delta=0.02,
                               msg=f"Δ={delta:.4f}, expected {expected:.4f} ± 0.02")


class TestShiftFineMode(K19Base):
    """Mit Shift ist die Änderung ¼ davon."""

    def test_shift_quarter(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        track_w = slider._track_width()

        slider.setValue(round(0.5 * (1.0 / 0.001)))
        start_val = row.value()

        drag_dist = track_w * 0.5
        self.drag(slider, track_w * 0.25, track_w * 0.25 + drag_dist, shift=True)

        end_val = row.value()
        delta = end_val - start_val
        expected = 0.5 * 1.0 * 0.25
        self.assertAlmostEqual(delta, expected, delta=0.02,
                               msg=f"Δ={delta:.4f}, expected {expected:.4f} ± 0.02")


class TestClickNoChange(K19Base):
    """Klick auf die Spur ohne Ziehen ändert den Wert nicht."""

    def test_click_track_no_change(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        slider.setValue(round(0.5 * (1.0 / 0.001)))
        start_val = row.value()

        pos = QPoint(int(slider._track_width() * 0.8), 9)
        QTest.mousePress(slider, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, pos)
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier, pos)
        self.app.processEvents()

        self.assertEqual(row.value(), start_val,
                         f"Value changed on click without drag: {start_val} → {row.value()}")


class TestWheel(K19Base):
    """Mausrad: +1 Raste = +step, mit Shift = +step/10. Spalte scrollt nicht."""

    def test_wheel_one_step(self):
        row, slider = self.make_slider(0.0, 1.0, 0.01, width=300)
        slider.setValue(50)
        start_val = row.value()

        self.wheel(slider, delta=120)

        end_val = row.value()
        self.assertAlmostEqual(end_val - start_val, 0.01, places=4,
                               msg=f"Wheel: Δ={end_val - start_val}, expected 0.01")

    def test_wheel_shift_tenth_step(self):
        row, slider = self.make_slider(0.0, 1.0, 0.01, width=300)
        slider.setValue(50)
        start_val = row.value()

        for _ in range(10):
            self.wheel(slider, delta=120, shift=True)

        end_val = row.value()
        self.assertAlmostEqual(end_val - start_val, 0.01, places=4,
                               msg=f"Wheel Shift ×10: Δ={end_val - start_val}, expected 0.01")

    def test_wheel_down(self):
        row, slider = self.make_slider(0.0, 1.0, 0.01, width=300)
        slider.setValue(50)
        start_val = row.value()

        self.wheel(slider, delta=-120)

        end_val = row.value()
        self.assertAlmostEqual(start_val - end_val, 0.01, places=4,
                               msg=f"Wheel down: Δ={start_val - end_val}, expected 0.01")


class TestClamping(K19Base):
    """Klemmung: Ziehen weit über das Ende hinaus ergibt genau das Maximum."""

    def test_clamp_max(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        slider.setValue(round(0.9 * (1.0 / 0.001)))

        track_w = slider._track_width()
        self.drag(slider, 0, track_w * 2)

        self.assertAlmostEqual(row.value(), 1.0, places=4,
                               msg=f"Value not clamped to max: {row.value()}")

    def test_clamp_min(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        slider.setValue(round(0.1 * (1.0 / 0.001)))

        track_w = slider._track_width()
        self.drag(slider, track_w, 0)

        self.assertAlmostEqual(row.value(), 0.0, places=4,
                               msg=f"Value not clamped to min: {row.value()}")


class TestGainBoundary(K19Base):
    """Gain-Schwelle: bei SHORT_TRACK_PX ist gain = 1.0, darunter 0.5."""

    def test_gain_long(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=300)
        self.assertGreaterEqual(slider._track_width(), SHORT_TRACK_PX)
        self.assertEqual(slider._gain(), 1.0)

    def test_gain_short(self):
        row, slider = self.make_slider(0.0, 1.0, 0.001, width=100)
        self.assertLess(slider._track_width(), SHORT_TRACK_PX)
        self.assertEqual(slider._gain(), 0.5)


if __name__ == "__main__":
    unittest.main()
