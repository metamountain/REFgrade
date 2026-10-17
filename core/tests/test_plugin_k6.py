"""Tests für Paket K6: Fenster-Verdrahtung (dialog.py mit Fake-Client)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade import pixel_rules  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    """Create a small valid PNG file."""
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


class FakeClient:
    """Records all calls; returns fixed responses with small PNGs."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))
        self._png2 = _make_png(tmp / "resp2.png", 64, 64, (50, 200, 100))
        self._raise_error = False
        self._fit_values = {
            "exposure": 0.3, "contrast": 0.2, "highlight_rolloff": 0.1,
            "temperature": 0.1, "tint": -0.1, "saturation": 0.5,
            "lift_m": 0.5, "lift_r": 0.05, "lift_g": 0.0, "lift_b": -0.05,
            "gamma_m": 0.5, "gamma_r": 1.0, "gamma_g": 0.95, "gamma_b": 1.05,
            "gain_m": 0.5, "gain_r": 1.1, "gain_g": 1.0, "gain_b": 0.9,
        }

    def set_error(self, msg: str = "GPU OOM") -> None:
        self._raise_error = True

    def clear_error(self) -> None:
        self._raise_error = False

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if self._raise_error:
            raise worker_client.WorkerError(500, self._error_msg)
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        elif path == "/crop":
            return {"image": self._png, "scale": 1.0}
        elif path == "/fit":
            return {"values": self._fit_values}
        elif path == "/apply":
            raw = self._tmp / "result.raw"
            raw.write_bytes(b"\x00" * 1024)
            return {"raw": str(raw), "ms": 1.0}
        elif path == "/cube":
            p = self._tmp / "test.cube"
            p.write_text("TITLE \"test\"\n")
            return {"path": str(p)}
        elif path == "/references":
            items = []
            for i in range(6):
                tp = _make_png(self._tmp / f"ref{i}.png", 32, 32, (i * 40, 100, 200))
                items.append({"path": f"/pool/ref{i}.jpg", "name": f"ref{i}",
                              "score": 0.9 - i * 0.05, "thumb": tp})
            return {"items": items}
        elif path == "/prerender":
            return {"image": self._png2}
        elif path == "/thumb":
            return {"thumb": self._png}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    @property
    def _error_msg(self) -> str:
        return "GPU OOM"

    def count(self, path: str) -> int:
        return sum(1 for p, pl in self.calls if p == path and (pl or {}).get("purpose") != "original")

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path and (payload or {}).get("purpose") != "original":
                return payload
        return None


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    """Process events until condition() is True or timeout."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _make_source(tmp: Path) -> dict:
    """A fake source dict (as if from krita_bridge.read_active + /load)."""
    return {
        "raw": str(tmp / "source.raw"),
        "width": 100,
        "height": 80,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "session": "fake_session_id",
    }


def _make_image(w: int = 200, h: int = 150) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(80, 120, 160))
    return img


class K6TestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k6_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()
        self.ok_calls: list[tuple[str, str]] = []
        self.lut_calls: list[tuple[str, str]] = []

        def on_ok(raw_path: str, layer_name: str):
            self.ok_calls.append((raw_path, layer_name))

        def save_lut(folder: str, filename: str) -> str | None:
            self.lut_calls.append((folder, filename))
            p = Path(folder) / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("TITLE \"test\"\n")
            return str(p)

        self.dlg = RefGradeDialog(
            self.image, None,
            client=self.fake, source=self.source,
            save_lut=save_lut, on_ok=on_ok,
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class GeometryTest(K6TestBase):
    """K25b: Panel width = 3s - m, where s = min(H/5, W/6)."""

    def test_column_width_1080p(self):
        """1920×1080: s = 216, panel = 3*216 - 9 = 639."""
        self.dlg.resize(1920, 1080)
        _wait(self.app, lambda: self.dlg.width() == 1920)
        self.app.processEvents()
        from ref_grade import scale
        s = scale.unit(1920, 1080)
        m = round(s * 0.04)
        expected = round(3 * s) - m
        self.assertAlmostEqual(self.dlg.panel.width(), expected, delta=2)

    def test_column_width_1440p(self):
        """2560×1440: s = 288, panel = 3*288 - 12 = 852."""
        self.dlg.resize(2560, 1440)
        _wait(self.app, lambda: self.dlg.width() == 2560)
        self.app.processEvents()
        from ref_grade import scale
        s = scale.unit(2560, 1440)
        m = round(s * 0.04)
        expected = round(3 * s) - m
        self.assertAlmostEqual(self.dlg.panel.width(), expected, delta=2)

    def test_histogram_in_scope_block(self):
        """K25b: Histogram is in the scope block (bottom), not the panel."""
        self.dlg.resize(1920, 1080)
        _wait(self.app, lambda: self.dlg.width() == 1920)
        self.app.processEvents()
        self.assertTrue(self.dlg._scope_block.isAncestorOf(self.dlg.histogram))


class DebounceTest(K6TestBase):
    """3 slider changes within 10 ms → exactly 1 /preview after 100 ms."""

    def test_debounce(self):
        # Wait for initial preview to settle
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial = self.fake.count("/preview")

        # 3 changes within 10 ms
        sections = self.dlg.sections
        tone_rows = sections["tone"][1]
        tone_rows[0].set_value(tone_rows[0].value() + 0.1)
        tone_rows[1].set_value(tone_rows[1].value() + 0.1)
        tone_rows[2].set_value(tone_rows[2].value() + 0.1)

        # After 100 ms: exactly one new /preview
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=1000)
        time.sleep(0.05)
        self.app.processEvents()
        total = self.fake.count("/preview")
        self.assertEqual(total, initial + 1)

        # The preview payload contains the new values and mix
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertIn("values", payload)
        self.assertIn("mix", payload)
        self.assertEqual(payload["mix"], self.dlg.mode())


class WheelTest(K6TestBase):
    """Wheel to (1, 0) → /preview.values has lift_r > 0 and lift_g < 0."""

    def test_wheel(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial = self.fake.count("/preview")

        self.dlg.wheels["lift"].set_point(1.0, 0.0)
        self.dlg.wheels["lift"].changed.emit("lift", 1.0, 0.0)

        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=1000)
        time.sleep(0.05)
        self.app.processEvents()

        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        vals = payload["values"]
        self.assertGreater(vals["lift_r"], 0)
        self.assertLess(vals["lift_g"], 0)


class OnOffTest(K6TestBase):
    """ON/OFF per section (K25c): all sections except reference ON at start;
    switching OFF dims the content and removes it from mix.weights, ON undims."""

    def test_default_on_sections(self):
        """At start: all sections except reference are ON (K25c)."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertEqual(set(payload["mix"]["weights"]), set())  # 2026-10-11: sections start OFF, ON once something changes
        for key in ("color", "rgb", "tone", "hsl", "curves", "wheels", "lut", "reference"):
            self.assertFalse(self.dlg.sections[key][0].is_on(), key)
        self.assertFalse(self.dlg.sections["reference"][0].is_on())

    def test_switch_tone_off(self):
        """K25c: tone starts ON; toggle OFF → removed from weights, content dimmed."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial = self.fake.count("/preview")
        self.dlg.sections["tone"][0].on.setChecked(False)
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=1000)
        time.sleep(0.05)
        self.app.processEvents()
        payload = self.fake.last("/preview")
        self.assertNotIn("tone", payload["mix"]["weights"])
        self.assertTrue(self.dlg._contents["tone"].property("dim") is True)

    def test_weight_zero(self):
        """K25c: color starts ON; amount=0 → removed from weights."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial = self.fake.count("/preview")
        self.dlg.sections["color"][0].amount_slider.set_value(0.0)
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=1000)
        time.sleep(0.05)
        self.app.processEvents()
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertNotIn("color", payload["mix"]["weights"])


class FitTest(K6TestBase):
    """FIT → 1 /fit, then 1 /preview; reference weight = 0."""

    def test_fit(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial_preview = self.fake.count("/preview")

        # K25c: reference section starts OFF and collapsed; expand and turn ON first
        self.dlg.sections["reference"][0].on.setChecked(True)
        self.dlg._set_section_collapsed("reference", False)
        self.app.processEvents()

        self.dlg.fit_btn.click()

        # Wait for /fit
        _wait(self.app, lambda: self.fake.count("/fit") >= 1, timeout_ms=2000)
        self.assertEqual(self.fake.count("/fit"), 1)

        # Then exactly one new /preview
        _wait(self.app, lambda: self.fake.count("/preview") >= initial_preview + 1,
              timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()
        self.assertEqual(self.fake.count("/preview"), initial_preview + 1)

        # Reference weight is 0
        ref_weight = self.dlg.sections["reference"][0].amount_slider.value()
        self.assertEqual(ref_weight, 0.0)


class HoverTest(K6TestBase):
    """Hover over tile 3 → /prerender with that path."""

    def test_hover(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        # Wait for /references to fill tiles
        _wait(self.app, lambda: self.fake.count("/references") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: self.dlg.slots.tiles[3].path is not None, timeout_ms=2000)
        if self.dlg.slots.tiles[3].path is None:
            self.skipTest("Tile 3 not filled")

        tile_path = self.dlg.slots.tiles[3].path
        initial = self.fake.count("/prerender")

        # Simulate hover
        from PyQt6.QtCore import QEvent
        enter = QEvent(QEvent.Type.Enter)
        self.dlg.eventFilter(self.dlg.slots.tiles[3], enter)

        _wait(self.app, lambda: self.fake.count("/prerender") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        payload = self.fake.last("/prerender")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["reference"], tile_path)


class OkTest(K6TestBase):
    """OK → /apply, then on_ok exactly once with layer name 'REF · <name>'."""

    def test_ok(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: self.fake.count("/references") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: any(t.active for t in self.dlg.slots.tiles), timeout_ms=2000)

        self.dlg._do_ok()

        _wait(self.app, lambda: self.fake.count("/apply") >= 1, timeout_ms=2000)
        self.assertEqual(self.fake.count("/apply"), 1)

        _wait(self.app, lambda: len(self.ok_calls) >= 1, timeout_ms=2000)
        self.assertEqual(len(self.ok_calls), 1)
        raw_path, layer_name = self.ok_calls[0]
        # Layer name should be "REF · <stem of first active ref>"
        first_ref = self.dlg._references()[0]
        expected_name = pixel_rules.layer_name(first_ref)
        self.assertEqual(layer_name, expected_name)
        self.assertTrue(layer_name.startswith("REF · "))


class LutTest(K6TestBase):
    """LUT… → save_lut gets .cube suggestion; then 1 /cube."""

    def test_lut(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: self.fake.count("/references") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: any(t.active for t in self.dlg.slots.tiles), timeout_ms=2000)

        self.dlg._do_lut()

        # save_lut was called with a .cube filename
        self.assertEqual(len(self.lut_calls), 1)
        folder, filename = self.lut_calls[0]
        self.assertTrue(filename.endswith(".cube"))

        # Then exactly one /cube
        _wait(self.app, lambda: self.fake.count("/cube") >= 1, timeout_ms=2000)
        self.assertEqual(self.fake.count("/cube"), 1)


class ZoomCropTest(K6TestBase):
    """Zoom to 100 % → after 200 ms exactly 1 /crop; rect covers visible area."""

    def test_crop(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: self.fake.count("/references") >= 1, timeout_ms=2000)
        # Wait for preview to be applied
        _wait(self.app, lambda: self.dlg._last_preview is not None, timeout_ms=2000)
        if self.dlg._last_preview is None:
            self.skipTest("No preview applied")

        # Zoom to 100 % (which is above preview scale for a 64px preview of 100px source)
        # preview_scale = 64/100 = 0.64, so zoom 1.0 > 0.64
        self.dlg.view.zoom_at(1.0 / self.dlg.view.zoom, QPointF(
            self.dlg.view.width() / 2, self.dlg.view.height() / 2))
        # Ensure zoom is at 100%
        self.dlg.view.zoom = 1.0
        self.dlg.view._fitted = False
        self.dlg.view._clamp()
        self.dlg.view.update()

        # Trigger crop scheduling
        self.dlg._maybe_schedule_crop()

        _wait(self.app, lambda: self.fake.count("/crop") >= 1, timeout_ms=2000)
        self.assertEqual(self.fake.count("/crop"), 1)

        payload = self.fake.last("/crop")
        self.assertIsNotNone(payload)
        rect = payload["rect"]
        self.assertEqual(len(rect), 4)
        # Rect should cover the visible area
        self.assertGreater(rect[2], 0)  # width > 0
        self.assertGreater(rect[3], 0)  # height > 0


class ErrorTest(K6TestBase):
    """Worker error → mark red, dialog stays open."""

    def test_error(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        initial = self.fake.count("/preview")

        self.fake.set_error()
        # Trigger a new preview
        tone_rows = self.dlg.sections["tone"][1]
        tone_rows[0].set_value(tone_rows[0].value() + 0.1)

        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # Mark should show error (grey accent, K27e: RED → grey)
        self.assertIn("ERROR", self.dlg.view.mark)
        self.assertIsNotNone(self.dlg.view.mark_color)
        self.assertEqual(self.dlg.view.mark_color.name(), "#b4b4b4")
        # Dialog is still open
        self.assertTrue(self.dlg.isVisible())

        self.fake.clear_error()


class ScreenshotTest(K6TestBase):
    """Screenshots 1920×1080 (landscape and portrait)."""

    def test_screenshot_landscape(self):
        self.dlg.resize(1920, 1080)
        _wait(self.app, lambda: self.dlg.width() == 1920)
        _wait(self.app, lambda: self.dlg._last_preview is not None, timeout_ms=2000)
        self.app.processEvents()
        png = REPORT_DIR / "K6_landscape.png"
        self.dlg.grab().save(str(png))
        self.assertTrue(png.exists())
        self.assertGreater(png.stat().st_size, 1000)

    def test_screenshot_portrait(self):
        self.dlg.resize(1080, 1920)
        _wait(self.app, lambda: self.dlg.width() == 1080)
        _wait(self.app, lambda: self.dlg._last_preview is not None, timeout_ms=2000)
        self.app.processEvents()
        png = REPORT_DIR / "K6_portrait.png"
        self.dlg.grab().save(str(png))
        self.assertTrue(png.exists())
        self.assertGreater(png.stat().st_size, 1000)


class DummyModeTest(unittest.TestCase):
    """Without client: the dummy still works (offscreen)."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_dummy_offscreen(self):
        dlg = RefGradeDialog()
        self.assertTrue(dlg.windowTitle().startswith("REFgrade"))
        # K25i: status strip shows short text; full text is the tooltip
        self.assertEqual(dlg.view.mark, "LAYOUT DUMMY")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
