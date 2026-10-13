"""K29 (user 2026-10-11): WHEELS chapter beside the wheels, wheel names above the master
sliders flush with the slider, vectorscope as large as a wheel disc, general ? opens help."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
ROOT = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", _pkg)

from PyQt6.QtCore import QPoint, QSettings  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)
from ref_grade.dialog import RefGradeDialog  # noqa: E402

POOL = ROOT / "bench" / "pool"
KINDS = ("gain", "gamma", "lift")


def _dialog(image: str) -> RefGradeDialog:
    QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
    d = RefGradeDialog(QImage(str(POOL / image)))
    d.resize(2560, 1392)
    d.show()
    app.processEvents()
    d.resize(2560, 1392)
    app.processEvents()
    return d


class TestWheelStrip(unittest.TestCase):

    def _check(self, d: RefGradeDialog) -> None:
        strip = d._wheel_strip
        sec = d._wheel_section
        self.assertTrue(sec.isVisible())
        self.assertFalse(d.panel.isAncestorOf(sec), "WHEELS chapter left the control column")
        sec_r = sec.geometry().translated(-strip.x(), -strip.y())   # strip coordinates
        self.assertGreaterEqual(sec_r.x(), 0, "chapter starts inside the strip")
        self.assertLessEqual(sec_r.bottom(), strip.height(), "chapter ends inside the strip")
        self.assertFalse(sec.geometry().intersects(d.panel.geometry()), "chapter clear of column")
        for kind in KINDS:
            m = d.masters[kind]
            name = d._wheel_names[kind]
            labels = [w for w in m.findChildren(QLabel) if w is not m.value_label]
            self.assertEqual(labels, [], f"{kind}: no Master/H/M/L label in the row")
            slider_x = m.slider.mapTo(strip, QPoint(0, 0)).x()
            self.assertEqual(name.geometry().x(), slider_x, f"{kind}: name flush with slider")
            self.assertEqual(name.geometry().bottom() + 1, m.geometry().y(),
                             f"{kind}: name directly above the slider row")
            c, r = d.wheels[kind]._geometry()
            self.assertGreaterEqual(name.geometry().y(), d.wheels[kind].geometry().y() + c.y() + r,
                                    f"{kind}: name below the wheel disc, not on it")
            self.assertFalse(sec_r.intersects(d.wheels[kind].geometry()), kind)
            self.assertFalse(sec_r.intersects(m.geometry()), kind)
        # each slider is nearer its own disc than the next wheel's disc (user 2026-10-11)
        for own, nxt in (("gain", "gamma"), ("gamma", "lift")):
            m = d.masters[own].geometry()
            co, ro = d.wheels[own]._geometry()
            cn, rn = d.wheels[nxt]._geometry()
            own_bottom = d.wheels[own].geometry().y() + co.y() + ro
            next_top = d.wheels[nxt].geometry().y() + cn.y() - rn
            if d.wheels[nxt].geometry().y() > m.y():          # V: next wheel below
                self.assertLess(m.y() - own_bottom, next_top - m.bottom(), own)
        _c, r = d.wheels["gain"]._geometry()
        self.assertAlmostEqual(d.vectorscope.width(), 2 * r, delta=1,
                               msg="vectorscope circle as large as a wheel disc")
        self.assertEqual(d.vectorscope.width(), d.vectorscope.height())
        self.assertEqual([d._wheel_names[k].text() for k in KINDS],
                         ["Highlights", "Midtones", "Shadows"])

    def test_portrait_v(self):
        d = _dialog("10.jpg")
        self.assertEqual(d._effective_mode, "V")
        self._check(d)
        sec_top = d._wheel_section.geometry().y() - d._wheel_strip.y()
        for kind in KINDS:
            self.assertGreaterEqual(sec_top, d.masters[kind].geometry().bottom(),
                                    "V: chapter under the wheels")
        d.close()

    def test_landscape_h(self):
        d = _dialog("07.jpg")
        self.assertEqual(d._effective_mode, "H")
        self._check(d)
        self.assertLessEqual(d._wheel_section.geometry().right() - d._wheel_strip.x(),
                             d.vectorscope.geometry().x(), "H: chapter left of the vectorscope")
        d.close()


class TestWheelScroll(unittest.TestCase):
    """Mouse wheel on a colour wheel moves the point along its direction, gently."""

    def _scroll(self, w, notches: int, shift: bool = False) -> None:
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QWheelEvent
        mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
        pos = QPointF(w.width() / 2, w.height() / 2)
        ev = QWheelEvent(pos, w.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120 * notches),
                         Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(w, ev)

    def test_scroll_scales_strength_keeps_hue(self):
        import math
        d = _dialog("07.jpg")
        w = d.wheels["gain"]
        got = []
        w.changed.connect(lambda k, u, v: got.append((u, v)))
        w.set_point(0.3, 0.4)                                  # r = 0.5
        self._scroll(w, 1)
        self.assertAlmostEqual(math.hypot(w.u, w.v), 0.52, places=6)
        self.assertAlmostEqual(math.atan2(w.v, w.u), math.atan2(0.4, 0.3), places=6)
        self._scroll(w, -2, shift=True)
        self.assertAlmostEqual(math.hypot(w.u, w.v), 0.51, places=6)
        self.assertEqual(len(got), 2, "each notch emits a change (→ preview)")
        w.set_point(0.0, 0.0)
        self._scroll(w, 3)
        self.assertEqual((w.u, w.v), (0.0, 0.0), "no direction at the centre: nothing moves")
        w.set_point(0.6, 0.8)
        self._scroll(w, 5)
        self.assertAlmostEqual(math.hypot(w.u, w.v), 1.0, places=6, msg="clamped at the rim")
        d.close()


class TestGeneralHelp(unittest.TestCase):

    def test_question_mark_beside_master_opens_and_closes_help(self):
        d = _dialog("07.jpg")
        d._general_help_btn.click()
        app.processEvents()
        self.assertIsNotNone(d._help_popup)
        self.assertTrue(d._help_popup.isVisible())
        d._general_help_btn.click()
        app.processEvents()
        self.assertIsNone(d._help_popup)
        d.close()


if __name__ == "__main__":
    unittest.main()
