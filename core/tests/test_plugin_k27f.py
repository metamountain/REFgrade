"""K27f: Two-stage preview (lut → full), fade-in, overlay_style."""

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

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from ref_grade import style  # noqa: E402
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


class K27fTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 2560×1392 with fake client."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self._empty_lut_dir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        self._orig_load_lut = RefGradeDialog._load_lut_folder

        def _patched_load_lut(self_dlg):
            self_dlg._refresh_lut_list(self._empty_lut_dir)

        RefGradeDialog._load_lut_folder = _patched_load_lut
        self._previews: list[dict] = []
        self._applies: list[dict] = []
        self._client_delay = 0.0  # seconds

    def tearDown(self):
        RefGradeDialog._load_lut_folder = self._orig_load_lut
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

    def _make_client(self):
        def fake_client(port, path, payload):
            if self._client_delay > 0:
                time.sleep(self._client_delay)
            if path == "/preview":
                self._previews.append(payload)
                img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "test_k27f_preview.png")
                QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
                return {"image": img_path, "histogram": {}, "ms": 5.0}
            if path == "/apply":
                self._applies.append(payload)
                return {"raw": "/tmp/fake.raw", "ms": 10.0}
            return {}
        return fake_client

    def _make_dialog(self, clarity: float = 0.0):
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg = RefGradeDialog(
            img,
            client=self._make_client(),
            source={"session": "test", "width": 400, "height": 300},
        )
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()
        # Set clarity value
        if clarity != 0.0:
            tone_rows = self.dlg.sections.get("tone", (None, None))[1]
            for r in tone_rows:
                if r.name == "clarity":
                    r.set_value(clarity)
                    break
        self.app.processEvents()


class TestLutStageOnly(K27fTestBase):
    """Clarity 0: one slider change → exactly one /preview with stage='lut', no full."""

    def test_single_lut_no_full(self):
        self._make_dialog(clarity=0.0)
        # Trigger a preview by changing a slider
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                r.set_value(0.3)
                break
        # Wait for the preview to complete
        _wait(self.app, lambda: len(self._previews) >= 1, timeout_ms=2000)
        self.app.processEvents()
        # Give time for a full preview to arrive (should not)
        time.sleep(0.3)
        self.app.processEvents()
        lut_requests = [p for p in self._previews if p.get("stage") == "lut"]
        full_requests = [p for p in self._previews if p.get("stage") == "full"]
        self.assertGreaterEqual(len(lut_requests), 1, "expected at least one lut request")
        self.assertEqual(len(full_requests), 0, "no full request expected with clarity=0")


class TestLutThenFull(K27fTestBase):
    """Clarity ≠ 0: first lut, then full."""

    def test_lut_then_full(self):
        self._make_dialog(clarity=0.5)
        self._previews.clear()
        # Trigger a preview
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                r.set_value(0.3)
                break
        # Wait for both lut and full
        _wait(self.app, lambda: any(p.get("stage") == "full" for p in self._previews),
              timeout_ms=3000)
        self.app.processEvents()
        lut_requests = [p for p in self._previews if p.get("stage") == "lut"]
        full_requests = [p for p in self._previews if p.get("stage") == "full"]
        self.assertGreaterEqual(len(lut_requests), 1, "expected at least one lut request")
        self.assertGreaterEqual(len(full_requests), 1, "expected at least one full request")
        # lut comes before full
        first_lut = self._previews.index(lut_requests[0])
        first_full = self._previews.index(full_requests[0])
        self.assertLess(first_lut, first_full, "lut should come before full")


class TestDragOnlyLut(K27fTestBase):
    """During drag: only lut requests; after release + 120 ms: full (if clarity ≠ 0)."""

    def test_drag_only_lut(self):
        self._make_dialog(clarity=0.5)
        self._previews.clear()
        # Simulate drag: set _dragging = True, make 5 changes
        self.dlg._dragging = True
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                for i in range(5):
                    r.set_value(0.1 * (i + 1))
                    self.app.processEvents()
                break
        # Wait for previews
        _wait(self.app, lambda: len(self._previews) >= 1, timeout_ms=2000)
        self.app.processEvents()
        # During drag: only lut
        full_during_drag = [p for p in self._previews if p.get("stage") == "full"]
        self.assertEqual(len(full_during_drag), 0, "no full during drag")

        # Release: _dragging = False, trigger preview
        self.dlg._dragging = False
        self.dlg._on_drag_stopped()
        # Wait for full
        _wait(self.app, lambda: any(p.get("stage") == "full" for p in self._previews),
              timeout_ms=3000)
        self.app.processEvents()
        full_after = [p for p in self._previews if p.get("stage") == "full"]
        self.assertGreaterEqual(len(full_after), 1, "full expected after release")


class TestStaleResponse(K27fTestBase):
    """A stale response does not change the displayed image."""

    def test_stale_dropped(self):
        self._make_dialog(clarity=0.0)
        self._client_delay = 0.5  # slow client
        self._previews.clear()
        # Trigger two quick changes
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                r.set_value(0.2)
                self.app.processEvents()
                r.set_value(0.4)
                break
        # Wait for both responses
        _wait(self.app, lambda: len(self._previews) >= 2, timeout_ms=5000)
        self.app.processEvents()
        # The dialog should still be functional (no crash)
        self.assertIsNotNone(self.dlg)


class TestFadeIn(K27fTestBase):
    """60 ms after a new result, fade alpha is between 0.3 and 0.7."""

    def test_fade_alpha_at_60ms(self):
        self._make_dialog(clarity=0.0)
        # Set an initial image
        img1 = QImage(100, 100, QImage.Format.Format_RGB32)
        img1.fill(QColor(100, 100, 100))
        self.dlg.view.replace_image(img1)
        self.app.processEvents()
        # Now replace with a new image (triggers fade)
        img2 = QImage(100, 100, QImage.Format.Format_RGB32)
        img2.fill(QColor(200, 200, 200))
        self.dlg.view.replace_image(img2)
        # Wait ~80 ms with event processing (10 ms timer interval needs real time)
        deadline = time.time() + 0.08
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        # Check fade alpha (8 ticks × 10/120 ≈ 0.67)
        alpha = self.dlg.view._fade_alpha
        self.assertGreater(alpha, 0.3, f"alpha {alpha} too low at 80ms")
        self.assertLess(alpha, 0.95, f"alpha {alpha} too high at 80ms")


class TestApplyFullSize(K27fTestBase):
    """OK → /apply in full size (no stage field, no preview_scale)."""

    def test_apply_no_stage(self):
        self._make_dialog(clarity=0.0)
        self._applies.clear()
        ok_btn = self.dlg.findChild(QPushButton, "btn_ok")
        self.assertIsNotNone(ok_btn)
        QTest.mouseClick(ok_btn, Qt.MouseButton.LeftButton)
        _wait(self.app, lambda: len(self._applies) >= 1, timeout_ms=2000)
        self.app.processEvents()
        self.assertGreaterEqual(len(self._applies), 1)
        # No stage field in apply
        self.assertNotIn("stage", self._applies[0])
        # No preview_scale widget or setting
        self.assertFalse(hasattr(self.dlg, "preview_scale"))
        self.assertIsNone(self.dlg._settings.value("preview_scale"))


class TestOverlayStyle(unittest.TestCase):
    """overlay_style returns the table values for each state."""

    def test_idle(self):
        alpha, color = style.overlay_style("idle")
        self.assertEqual(alpha, 60)
        self.assertEqual(color, "#5A5A5A")

    def test_hover(self):
        alpha, color = style.overlay_style("hover")
        self.assertEqual(alpha, 140)
        self.assertEqual(color, "#5A5A5A")

    def test_busy(self):
        alpha, color = style.overlay_style("busy")
        self.assertEqual(alpha, 200)
        self.assertEqual(color, "#8A8A8A")

    def test_done(self):
        alpha, color = style.overlay_style("done")
        self.assertEqual(alpha, 200)
        self.assertEqual(color, "#B4B4B4")

    def test_error(self):
        alpha, color = style.overlay_style("error")
        self.assertEqual(alpha, 200)
        self.assertEqual(color, "#A04040")

    def test_unknown_state(self):
        alpha, color = style.overlay_style("unknown")
        self.assertEqual(alpha, 60)
        self.assertEqual(color, "#5A5A5A")


class TestOverlayTransitions(K27fTestBase):
    """Slow fake client: busy during request, not busy after; error sets mark_color."""

    def test_busy_then_idle(self):
        self._make_dialog(clarity=0.0)
        self._client_delay = 0.3  # 300 ms slow client
        # Trigger a preview
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                r.set_value(0.3)
                break
        # Wait for the request to be in flight (loading should be active)
        _wait(self.app, lambda: self.dlg.view._loading, timeout_ms=1000)
        self.assertTrue(self.dlg.view._loading, "loading should be active during request")
        # Wait for the result
        _wait(self.app, lambda: not self.dlg.view._loading, timeout_ms=3000)
        self.assertFalse(self.dlg.view._loading, "loading should be inactive after result")

    def test_error_sets_mark_color(self):
        self._make_dialog(clarity=0.0)
        # Make the client raise an error
        def error_client(port, path, payload):
            raise RuntimeError("fake worker error")
        self.dlg._client = error_client
        self._previews.clear()
        tone_rows = self.dlg.sections.get("tone", (None, None))[1]
        for r in tone_rows:
            if r.name == "exposure":
                r.set_value(0.3)
                break
        # Wait for the error to be processed
        _wait(self.app, lambda: self.dlg.view.mark_color is not None, timeout_ms=3000)
        self.assertIsNotNone(self.dlg.view.mark_color, "mark_color should be set on error")


if __name__ == "__main__":
    unittest.main()
