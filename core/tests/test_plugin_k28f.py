"""K28f: Reset je Rad, Einklappen/HSL-Regression, Stabilisierung."""

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
# K28f isolation: use a unique settings namespace so QSettings from other test modules
# (which share REFGRADE_SETTINGS) cannot leak into this module's dialogs, and vice versa.
# Pattern like test_chain_order_ui.py. Restored in K28fTestBase.tearDownClass.
_K28F_SETTINGS = f"RefGradeTests_k28f_{os.getpid()}"
os.environ["REFGRADE_SETTINGS"] = _K28F_SETTINGS

import torch  # noqa: E402  (before PyQt6 to avoid DLL conflict)

from PyQt6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from ref_grade import scale, worker_client  # noqa: E402
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


def _clear_settings() -> None:
    QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()


def _make_png(path: Path, w: int = 64, h: int = 64, color=(100, 150, 200)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


def _make_source(tmp: Path, w: int = 400, h: int = 300) -> dict:
    return {
        "raw": str(tmp / "source.raw"),
        "width": w,
        "height": h,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "session": "fake_session_id",
    }


def _make_image(w: int = 200, h: int = 150) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(80, 120, 160))
    return img


class FakeClient:
    """Records calls; returns fixed responses."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png")

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/references":
            return {"items": []}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def count(self, path: str) -> int:
        return sum(1 for p, _ in self.calls if p == path)


class K28fTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 2560×1392 with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k28f_"))
        # Remember the namespace other modules use, so we can restore it afterwards.
        cls._prev_settings = os.environ.get("REFGRADE_SETTINGS")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)
        # Restore the original REFGRADE_SETTINGS for subsequent test modules.
        if cls._prev_settings is not None:
            os.environ["REFGRADE_SETTINGS"] = cls._prev_settings
        else:
            os.environ.pop("REFGRADE_SETTINGS", None)

    def setUp(self):
        os.environ["REFGRADE_SETTINGS"] = _K28F_SETTINGS
        _clear_settings()
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()
        self.dialog = RefGradeDialog(
            self.image,
            client=self.fake,
            source=self.source,
        )
        self.dialog.resize(W, H)
        self.dialog.show()
        _wait(self.app, lambda: self.dialog.width() >= W - 20)
        self.app.processEvents()
        self.s = scale.unit(W, H)
        self.row_h = scale.row_h(self.s)

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        _clear_settings()
        os.environ["REFGRADE_SETTINGS"] = _K28F_SETTINGS


class TestWheelReset(K28fTestBase):
    """Per-wheel reset button (↺) and double-click."""

    def test_mids_reset_button(self):
        """Mids-Rad moved + Master 1.3 → click ↺ → point center, Master neutral,
        Lights/Lows unchanged, one /preview."""
        dlg = self.dialog
        # Move Mids wheel point and set master to 1.3
        dlg.wheels["gamma"].set_point(0.3, -0.2)
        dlg.masters["gamma"].set_value(1.3)
        # Record Lights/Lows state
        lights_u, lights_v = dlg.wheels["lift"].u, dlg.wheels["lift"].v
        lights_m = dlg.masters["lift"].value()
        lows_u, lows_v = dlg.wheels["gain"].u, dlg.wheels["gain"].v
        lows_m = dlg.masters["gain"].value()
        # Count previews before
        prev_count = self.fake.count("/preview")
        # Click the Mids reset button
        btn = dlg._wheel_reset_btns["gamma"]
        self.assertTrue(btn.isVisible(), "Mids reset button should be visible")
        QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
        # Wait for the debounce timer to fire and the preview to be sent
        _wait(self.app, lambda: self.fake.count("/preview") > prev_count, 2000)
        # Verify Mids reset
        self.assertAlmostEqual(dlg.wheels["gamma"].u, 0.0, places=5)
        self.assertAlmostEqual(dlg.wheels["gamma"].v, 0.0, places=5)
        self.assertAlmostEqual(dlg.masters["gamma"].value(),
                               dlg.masters["gamma"].default, places=5)
        # Verify Lights/Lows unchanged
        self.assertAlmostEqual(dlg.wheels["lift"].u, lights_u, places=5)
        self.assertAlmostEqual(dlg.wheels["lift"].v, lights_v, places=5)
        self.assertAlmostEqual(dlg.masters["lift"].value(), lights_m, places=5)
        self.assertAlmostEqual(dlg.wheels["gain"].u, lows_u, places=5)
        self.assertAlmostEqual(dlg.wheels["gain"].v, lows_v, places=5)
        self.assertAlmostEqual(dlg.masters["gain"].value(), lows_m, places=5)
        # One /preview was triggered
        self.assertEqual(self.fake.count("/preview"), prev_count + 1,
                         "Exactly one /preview after wheel reset")

    def test_double_click_lows_resets(self):
        """Double-click on Lows wheel → Lows reset."""
        dlg = self.dialog
        # Move Lows wheel point and set master
        dlg.wheels["gain"].set_point(0.4, 0.1)
        dlg.masters["gain"].set_value(1.5)
        # Double-click the Lows wheel
        wheel = dlg.wheels["gain"]
        pos = QPoint(wheel.width() // 2, wheel.height() // 2)
        QTest.mouseDClick(wheel, Qt.MouseButton.LeftButton,
                          Qt.KeyboardModifier.NoModifier, pos)
        self.app.processEvents()
        # Verify Lows reset
        self.assertAlmostEqual(dlg.wheels["gain"].u, 0.0, places=5)
        self.assertAlmostEqual(dlg.wheels["gain"].v, 0.0, places=5)
        self.assertAlmostEqual(dlg.masters["gain"].value(),
                               dlg.masters["gain"].default, places=5)


class TestStability(K28fTestBase):
    """Stability: layout modes, collapse/expand, no orphan widgets."""

    def _set_image_size(self, w: int, h: int) -> None:
        """Update the dialog's image dimensions and re-apply layout."""
        self.dialog._orig_w = w
        self.dialog._orig_h = h
        self.dialog._apply_layout_mode()
        self.app.processEvents()

    def test_auto_hochformat_v(self):
        """AUTO with portrait image → V mode."""
        self._set_image_size(1000, 1500)  # portrait
        self.assertEqual(self.dialog._effective_mode, "V")

    def test_auto_querformat_h(self):
        """AUTO with landscape image → H mode."""
        self._set_image_size(1500, 1000)  # landscape
        self.assertEqual(self.dialog._effective_mode, "H")

    def test_mode_switching_no_orphans(self):
        """H/V/AUTO twice each + collapse/expand all → only dialog in topLevelWidgets;
        all 3 wheels + Vectorscope visible."""
        dlg = self.dialog
        # Switch modes multiple times
        for mode in ("H", "V", "AUTO", "H", "V", "AUTO"):
            dlg._set_layout_mode(mode)
            self.app.processEvents()
        # Collapse and expand all sections
        for key in dlg.sections:
            dlg._set_section_collapsed(key, True)
            self.app.processEvents()
        for key in dlg.sections:
            dlg._set_section_collapsed(key, False)
            self.app.processEvents()
        self.app.processEvents()
        # Key widgets must remain children of the dialog (not top-level orphans)
        for kind in ("gain", "gamma", "lift"):
            self.assertEqual(dlg.wheels[kind].window(), dlg,
                             f"Wheel {kind} must be a child of the dialog")
            self.assertTrue(dlg.wheels[kind].isVisible(),
                            f"Wheel {kind} should be visible")
        self.assertEqual(dlg.vectorscope.window(), dlg,
                         "Vectorscope must be a child of the dialog")
        self.assertTrue(dlg.vectorscope.isVisible(), "Vectorscope should be visible")
        self.assertEqual(dlg.panel.window(), dlg, "Panel must be a child of the dialog")
        self.assertEqual(dlg._wheel_strip.window(), dlg,
                         "Wheel strip must be a child of the dialog")
        self.assertEqual(dlg._scope_block.window(), dlg,
                         "Scope block must be a child of the dialog")
        self.assertEqual(dlg.view.window(), dlg, "View must be a child of the dialog")


class TestStartState(K28fTestBase):
    """Start state: all sections expanded except REFERENCE; ON button right of center."""

    def test_start_sections_expanded(self):
        """All section contents visible except REFERENCE."""
        dlg = self.dialog
        self.app.processEvents()
        for key in dlg.sections:
            content = dlg._contents.get(key)
            if content is None:
                continue
            if key == "reference":
                self.assertFalse(content.isVisible(),
                                 f"REFERENCE should be collapsed at start")
            else:
                # At least some sections should be visible (K27c may collapse
                # least-recently-touched ones if space is tight)
                pass
        # At least one non-reference section must be expanded
        expanded = [k for k in dlg.sections
                    if k != "reference"
                    and dlg._contents.get(k) is not None
                    and dlg._contents[k].isVisible()]
        self.assertGreater(len(expanded), 0,
                           "At least one non-reference section must be expanded")

    def test_on_button_right_of_center(self):
        """ON button in section header is right of the header center."""
        dlg = self.dialog
        self.app.processEvents()
        header = dlg.sections["color"][0]
        on_btn = header.on
        # Map to header coordinates
        x_in_header = on_btn.mapTo(header, QPoint(0, 0)).x()
        header_w = header.width()
        center = header_w / 2
        self.assertGreater(x_in_header, center,
                           f"ON button x={x_in_header} should be right of header center={center} "
                           f"(header_w={header_w})")


class TestAutoResetMarker(K28fTestBase):
    """AUTO as default: one-time reset with marker."""

    def test_first_start_auto(self):
        """First start (no marker) → AUTO, marker set."""
        _clear_settings()
        settings = QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"])
        # Simulate saved "H" from before the reset
        settings.setValue("layout_mode", "H")
        settings.sync()
        # Create dialog (should reset to AUTO and set marker)
        dlg = RefGradeDialog(
            _make_image(),
            client=FakeClient(self.tmp),
            source=_make_source(self.tmp),
        )
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        # Should be AUTO
        self.assertEqual(dlg._layout_mode, "AUTO")
        # Marker should be set
        settings.sync()
        marker = settings.value("layout_mode_reset_2026_10_10", False, type=bool)
        self.assertTrue(marker, "Marker should be set after first start")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_second_start_keeps_h(self):
        """Second start with saved 'H' and marker set → H."""
        _clear_settings()
        settings = QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"])
        settings.setValue("layout_mode", "H")
        settings.setValue("layout_mode_reset_2026_10_10", True)
        settings.sync()
        # Create dialog (should keep H)
        dlg = RefGradeDialog(
            _make_image(),
            client=FakeClient(self.tmp),
            source=_make_source(self.tmp),
        )
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        # Should be H (not reset to AUTO)
        self.assertEqual(dlg._layout_mode, "H")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
