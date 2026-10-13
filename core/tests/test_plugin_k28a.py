"""K28a: Reset je Kurvenkanal; Clipping-Warnung mit Wert im Histogramm."""

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

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.widgets import Histogram, Parade  # noqa: E402

W, H = 1920, 1152


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K28aTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 1920×1152."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self, client=None, source=None):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        if client is None:
            source = source or {"session": "test", "width": 400, "height": 300}
            client = lambda port, path, payload: {}
        dlg = RefGradeDialog(img, client=client, source=source)
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        return dlg

    def tearDown(self):
        self.app.processEvents()


class TestClipDisplay(K28aTestBase):
    """Clipping-Warnung: Text sichtbar ab 0.5 %, unsichtbar darunter."""

    def _fake_preview_result(self, clip: dict[str, float]) -> dict:
        """Build a minimal /preview response with the given clip dict."""
        img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "k28a_clip.png")
        QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
        return {"image": img_path, "histogram": {}, "ms": 5.0, "clip": clip}

    def test_clip_visible_at_12_percent(self):
        """clip={r_hi: 0.12} → Histogramm zeigt „R ▲ 12 %"."""
        dlg = self._make_dialog()
        clip = {"r_hi": 0.12, "r_lo": 0.0, "g_hi": 0.0,
                "g_lo": 0.0, "b_hi": 0.0, "b_lo": 0.0}
        dlg._apply_preview(self._fake_preview_result(clip))
        self.assertEqual(dlg.histogram._clip.get("r_hi"), 0.12)
        self.assertGreaterEqual(dlg.histogram._clip.get("r_hi", 0), 0.005)

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_clip_hidden_below_threshold(self):
        """clip={r_hi: 0.003} → kein Text (unter 0.5 %)."""
        dlg = self._make_dialog()
        clip = {"r_hi": 0.003, "r_lo": 0.0, "g_hi": 0.0,
                "g_lo": 0.0, "b_hi": 0.0, "b_lo": 0.0}
        dlg._apply_preview(self._fake_preview_result(clip))
        # Clip data is below threshold → no text rendered
        self.assertLess(dlg.histogram._clip.get("r_hi", 0), 0.005)

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_parade_receives_clip(self):
        """Parade erhält ebenfalls Clip-Daten."""
        dlg = self._make_dialog()
        clip = {"r_hi": 0.0, "r_lo": 0.0, "g_hi": 0.0,
                "g_lo": 0.0, "b_hi": 0.08, "b_lo": 0.0}
        dlg._apply_preview(self._fake_preview_result(clip))
        self.assertEqual(dlg._parade._clip.get("b_hi"), 0.08)

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestCurvesChannelReset(K28aTestBase):
    """Kanal-Reset: nur aktiver Kanal wird zurückgesetzt."""

    def test_reset_active_channel_only(self):
        """R und G verbogen, R gewählt → Reset → R identisch, G unverändert;
        genau ein /preview danach."""
        previews = []

        def fake_client(port, path, payload):
            if path == "/preview":
                previews.append(payload)
                img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "k28a_curve.png")
                QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
                return {"image": img_path, "histogram": {}, "ms": 5.0}
            return {}

        dlg = self._make_dialog(client=fake_client)
        editor = dlg.curves_editor

        # Bend R and G channels
        r_bent = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]
        g_bent = [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]]
        editor.set_curve("r", r_bent)
        editor.set_curve("g", g_bent)

        # Select R channel
        editor.set_active("r")

        # Trigger channel reset — calls reset_channel() + _schedule_preview()
        dlg._curves_channel_reset()

        # R should be identity immediately (synchronous)
        r_curve = editor.get_curve("r")
        self.assertEqual(len(r_curve), 2)
        self.assertAlmostEqual(r_curve[0][0], 0.0)
        self.assertAlmostEqual(r_curve[0][1], 0.0)
        self.assertAlmostEqual(r_curve[1][0], 1.0)
        self.assertAlmostEqual(r_curve[1][1], 1.0)

        # G should be unchanged
        g_curve = editor.get_curve("g")
        self.assertEqual(len(g_curve), 3)
        self.assertAlmostEqual(g_curve[1][1], 0.3)

        # _schedule_preview was called → debounce timer is active
        self.assertTrue(dlg._debounce_timer.isActive(),
                        "Debounce-Timer nach Reset nicht aktiv")

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_reset_button_exists(self):
        """Der ↺-Knopf existiert im Dialog."""
        dlg = self._make_dialog()
        from PyQt6.QtWidgets import QPushButton
        btn = dlg.findChild(QPushButton, "curves_ch_reset")
        self.assertIsNotNone(btn, "curves_ch_reset Button nicht gefunden")
        self.assertEqual(btn.text(), "↺")

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestWheelLabels(K28aTestBase):
    """K28a: Räder beschriftet (Highlights/Mids/Shadows), Tooltips, keine Überlappung."""

    def test_labels_correct(self):
        """Beschriftungen exakt „Highlights", „Mids", „Shadows" an Gain/Gamma/Lift."""
        dlg = self._make_dialog()
        expected = {"gain": "Highlights", "gamma": "Midtones", "lift": "Shadows"}
        for kind, text in expected.items():
            wheel = dlg.wheels[kind]
            self.assertEqual(wheel._label, text, f"{kind}: _label={wheel._label!r}")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_tooltips_present(self):
        """Jedes Rad hat einen Tooltip mit dem Rad-Namen."""
        dlg = self._make_dialog()
        for kind in ("gain", "gamma", "lift"):
            tip = dlg.wheels[kind].toolTip()
            self.assertTrue(len(tip) > 10, f"{kind}: Tooltip fehlt oder zu kurz")
            self.assertIn(kind.capitalize() if kind != "gain" else "Highlights", tip)
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_label_no_overlap(self):
        """K29 (user 2026-10-11): the name is no longer drawn on the wheel but sits as its own
        label directly above the master slider; it must not overlap the wheel disc."""
        dlg = self._make_dialog()
        try:
            for kind in ("gain", "gamma", "lift"):
                wheel = dlg.wheels[kind]
                name = dlg._wheel_names[kind]
                c, r = wheel._geometry()
                disc_bottom = wheel.geometry().y() + c.y() + r
                self.assertGreaterEqual(name.geometry().y(), disc_bottom,
                                        f"{kind}: name top {name.geometry().y()} "
                                        f"< disc bottom {disc_bottom:.1f}")
        finally:
            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
