"""Nutzerwünsche 2026-10-16: Regler-Griffe nie verdeckt, TONE nie auto-eingeklappt,
Vorschau in Anzeigegröße (kein bilineares Herunterskalieren in Qt = Aliasing)."""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings, QSizeF  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.widgets import SliderRow  # noqa: E402


def _settle(app: QApplication, ms: int = 150) -> None:
    end = time.time() + ms / 1000
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


class _Base(unittest.TestCase):
    SIZE = (2560, 1392)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(*self.SIZE)
        self.dlg.show()
        _settle(self.app)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestSliderNoOverlap(_Base):
    """In every visible SliderRow the slider ends before the "?" / value label (else the
    handle hides under the "?" at max)."""

    def _check(self):
        for key in list(self.dlg._contents):
            self.dlg._set_section_collapsed(key, False)
        _settle(self.app)
        n = 0
        for r in self.dlg.findChildren(SliderRow):
            if not r.isVisible():
                continue
            n += 1
            right = r.slider.geometry().right()
            for w in (r.help_btn, r.value_label):
                if w is not None and w.isVisible():
                    self.assertLess(right, w.geometry().x(),
                                    f"{r.name}: slider ends at {right}, next widget at {w.x()}")
        self.assertGreater(n, 40)

    def test_layout_h(self):
        self.dlg._set_layout_mode("H")
        self._check()

    def test_layout_v(self):
        self.dlg._set_layout_mode("V")
        self._check()

    def test_small_screen(self):
        self.dlg.resize(1366, 720)
        _settle(self.app)
        self._check()


class TestToneStaysOpen(_Base):
    def test_tone_open_at_start(self):
        self.assertTrue(self.dlg._contents["tone"].isVisible())

    def test_tone_open_small_screen(self):
        self.dlg.resize(1366, 720)
        _settle(self.app)
        self.assertTrue(self.dlg._contents["tone"].isVisible())


class TestPreviewDisplaySize(_Base):
    def test_max_side_is_fitted_long_side(self):
        view = self.dlg.view
        r = view.devicePixelRatioF()
        for lw, lh in ((4000, 6000), (6000, 4000), (3000, 3000)):
            view.set_image(QImage(lw // 10, lh // 10, QImage.Format.Format_RGB32), QSizeF(lw, lh))
            self.app.processEvents()
            shown = max(lw, lh) * view.fit_zoom() * r
            self.assertLessEqual(abs(self.dlg._max_side() - shown), 1.0, f"{lw}x{lh}")


class TestBrightBarHandle(unittest.TestCase):
    """COLOR REPLACE luminance bar: round handle like the sliders, ends reachable (2026-10-17)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_handle_and_ends(self):
        from ref_grade.dlg_color_replace import BrightBar
        b = BrightBar("cr_0_src_bright", 0.5)
        b.resize(200, b.height())
        self.assertGreaterEqual(b.height(), BrightBar.HANDLE)
        b._set_from_x(0)
        self.assertEqual(b.value(), 0.0)
        b._set_from_x(b.width())
        self.assertEqual(b.value(), 1.0)
        b._set_from_x(b.width() / 2)
        self.assertAlmostEqual(b.value(), 0.5, places=2)
        img = b.grab().toImage()
        c = img.pixelColor(round(b.HANDLE / 2 + 0.5 * (b.width() - b.HANDLE)), b.height() // 2)
        self.assertGreater(c.lightness(), 180, "handle not drawn at the value")


if __name__ == "__main__":
    unittest.main()
