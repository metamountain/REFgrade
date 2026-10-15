"""Tests für Paket K10b: Hue-Kurven-Tabs, Pipette, Vectorscope, Layout."""

from __future__ import annotations

import math
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

# torch vor PyQt6 importieren, um DLL-Konflikt zu vermeiden
import torch  # noqa: E402

from PyQt6.QtCore import QPoint, QSettings, QPointF, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QMouseEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog, _srgb_hue  # noqa: E402
from ref_grade.pchip import pchip_cyclic_eval, pchip_cyclic_curve  # noqa: E402
from ref_grade.curves_ui import HUE_CHANNELS  # noqa: E402

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


def _clear_settings() -> None:
    QSettings("RefGrade", os.environ.get("REFGRADE_SETTINGS", "RefGradeTests")).clear()


class FakeClient:
    """Records all calls; returns fixed responses with small PNGs + vectorscope data."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))
        # 96×96 vectorscope: one cell at (48, 48) = 1.0, rest 0
        self._scope = [[0.0] * 96 for _ in range(96)]
        self._scope[48][48] = 1.0
        self._scope_src = [[0.0] * 96 for _ in range(96)]

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png,
                    "histogram": {"source": [0.015625] * 64, "result": [0.02] * 64},
                    "vectorscope": self._scope,
                    "vectorscope_source": self._scope_src,
                    "ms": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/references":
            return {"items": []}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def count(self, path: str) -> int:
        return sum(1 for p, pl in self.calls if p == path and (pl or {}).get("purpose") != "original")

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path and (payload or {}).get("purpose") != "original":
                return payload
        return None


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _make_source(tmp: Path) -> dict:
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


class K10bTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k10b_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        _clear_settings()
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()
        self.ok_calls: list[tuple[str, str]] = []
        self.dialog = RefGradeDialog(
            self.image,
            client=self.fake,
            source=self.source,
            on_ok=lambda raw, name: self.ok_calls.append((raw, name)),
        )
        self.dialog.resize(1920, 1080)
        self.dialog.show()
        _wait(self.app, lambda: self.dialog.isVisible())

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()


class TestHueTabs(K10bTestBase):
    """Hue-Tabs: Punkt setzen → /preview.mix.curves.hue_hue; RGB bleibt unverändert."""

    def test_hvh_point_in_preview(self):
        """HvH aktiv, Punkt setzen → /preview enthält hue_hue mit dem Punkt."""
        d = self.dialog
        # Activate HvH tab
        d.curves_editor.set_active("hue_hue")
        self.app.processEvents()
        # Set a point (simulate click at x=0.3, y=0.7)
        r = d.curves_editor._plot_rect()
        pos = QPointF(r.x() + 0.3 * r.width(), r.y() + (1.0 - 0.7) * r.height())
        ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, pos,
                         Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier)
        d.curves_editor.mousePressEvent(ev)
        self.app.processEvents()
        # Wait for preview to be scheduled and sent
        _wait(self.app, lambda: self.fake.count("/preview") > 0, 2000)
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        curves = payload.get("mix", {}).get("curves", {})
        self.assertIn("hue_hue", curves)
        # The point at x≈0.3 should be in the curve
        xs = [p[0] for p in curves["hue_hue"]]
        self.assertTrue(any(abs(x - 0.3) < 0.05 for x in xs),
                        f"Expected point near x=0.3, got xs={xs}")

    def test_rgb_unchanged_after_hue(self):
        """RGB-Kurve bleibt unverändert, wenn nur Hue-Tabs bearbeitet werden."""
        d = self.dialog
        # Activate HvH and set a point
        d.curves_editor.set_active("hue_hue")
        d.curves_editor.add_point_at(0.4, 0.6)
        self.app.processEvents()
        # RGB curve should still be identity
        rgb_pts = d.curves_editor.get_curve("rgb")
        self.assertEqual(rgb_pts, [[0.0, 0.0], [1.0, 1.0]])


class TestCyclicPCHIP(unittest.TestCase):
    """pchip.py zyklisch = Kern (huecurves._cyclic_pchip), max 1e-6 auf 1001 Stellen."""

    def test_cyclic_matches_core(self):
        """Plugin pchip_cyclic_eval matches core _cyclic_pchip on 1001 points."""
        from refgrade.huecurves import _cyclic_pchip
        # Test points (strictly increasing, x in [0,1])
        pts = [[0.0, 0.5], [0.25, 0.6], [0.5, 0.4], [0.75, 0.7], [1.0, 0.5]]
        x_pts = [p[0] for p in pts]
        y_pts = [p[1] for p in pts]

        # Core: evaluate on 1001 points (same x values as plugin)
        n = 1001
        x_vals = [i / (n - 1) for i in range(n)]
        x_tensor = torch.tensor(x_vals, dtype=torch.float64).unsqueeze(-1)
        core_vals = _cyclic_pchip(x_tensor, pts).flatten().tolist()

        # Plugin: evaluate on same 1001 points
        plugin_vals = [pchip_cyclic_eval(x_pts, y_pts, x) for x in x_vals]

        max_diff = max(abs(a - b) for a, b in zip(core_vals, plugin_vals))
        self.assertLess(max_diff, 1e-6,
                        f"Max diff {max_diff:.2e} exceeds 1e-6")

    def test_cyclic_curve_matches_core(self):
        """pchip_cyclic_curve matches core on 256 points (same x values)."""
        from refgrade.huecurves import _cyclic_pchip
        pts = [[0.0, 0.5], [0.3, 0.7], [0.6, 0.3], [1.0, 0.5]]
        x_pts = [p[0] for p in pts]
        y_pts = [p[1] for p in pts]

        n = 256
        x_vals = [i / (n - 1) for i in range(n)]
        x_tensor = torch.tensor(x_vals, dtype=torch.float64).unsqueeze(-1)
        core_vals = _cyclic_pchip(x_tensor, pts).flatten().tolist()
        plugin_vals = pchip_cyclic_curve(x_pts, y_pts, n)

        max_diff = max(abs(a - b) for a, b in zip(core_vals, plugin_vals))
        self.assertLess(max_diff, 1e-6,
                        f"Max diff {max_diff:.2e} exceeds 1e-6")


class TestPipette(K10bTestBase):
    """Pipette: Strg+Klick mit reinem Rot → Punkt bei Hue(Rot) ± 0.01."""

    def test_pipette_red(self):
        """Ctrl+Click on pure red pixel → point at x ≈ Hue(Red) in HvH tab."""
        d = self.dialog
        # Set a pure red preview image
        red_img = QImage(100, 100, QImage.Format.Format_RGB32)
        red_img.fill(QColor(255, 0, 0))
        d._last_preview = red_img
        d.view.replace_image(red_img)
        # Activate HvH tab
        d.curves_editor.set_active("hue_hue")
        self.app.processEvents()

        # Expected hue of pure red: atan2(b, a) for R=1, G=0, B=0
        # linear: R=1, G=0, B=0
        # X=0.4124564, Y=0.2126729, Z=0.0193339
        # Lab: a≈500*(f(X/Xn)-f(Y/Yn)), b≈200*(f(Y/Yn)-f(Z/Zn))
        # For pure red: a > 0, b > 0 → hue in first quadrant
        expected_hue = _srgb_hue(1.0, 0.0, 0.0)

        # Click at center of the view
        cx = d.view.width() // 2
        cy = d.view.height() // 2
        ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(cx, cy),
                         Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.ControlModifier)
        # Use eventFilter to handle it
        d.eventFilter(d.view, ev)
        self.app.processEvents()

        # Check that a point was added near the expected hue
        pts = d.curves_editor.get_curve("hue_hue")
        # Should have at least 3 points (identity 2 + new 1)
        self.assertGreaterEqual(len(pts), 3)
        # Find the new point (not the identity endpoints)
        new_pts = [p for p in pts if not (p[0] in (0.0, 1.0) and abs(p[1] - 0.5) < 1e-6)]
        self.assertTrue(len(new_pts) >= 1, f"No new point found, pts={pts}")
        # The new point's x should be near expected_hue
        found = any(abs(p[0] - expected_hue) < 0.01 for p in new_pts)
        self.assertTrue(found,
                        f"Expected point near x={expected_hue:.4f}, got {new_pts}")


class TestVectorscope(K10bTestBase):
    """Vectorscope: Pixel-Test, Overlay-Toggle."""

    def test_scope_pixel(self):
        """Fake /preview mit 96×96 (Zelle 48,48=1) → Widget zeigt helle Region."""
        d = self.dialog
        # Trigger a preview to get scope data
        d._schedule_preview()
        _wait(self.app, lambda: d.vectorscope._result is not None, 2000)
        self.app.processEvents()

        # Grab the widget and check that the center area is bright
        img = d.vectorscope.grab().toImage()
        w, h = img.width(), img.height()
        # The cell (48, 48) in a 96×96 grid maps to roughly the center
        cx, cy = w // 2, h // 2
        c = QColor(img.pixel(cx, cy))
        # The center should be brighter than the corners
        corner = QColor(img.pixel(5, 5))
        self.assertGreater(c.lightness(), corner.lightness(),
                           f"Center ({c.lightness()}) should be brighter than corner ({corner.lightness()})")


class TestLayout(K10bTestBase):
    """K25b: Vectorscope ist ein Quadrat im Wheel-Strip; Histogramm im Scope-Block (bottom)."""

    def test_scope_square(self):
        """Vectorscope ist quadratisch (±1px) — ein Quadrat von 4 im Strip."""
        d = self.dialog
        self.app.processEvents()
        w = d.vectorscope.width()
        h = d.vectorscope.height()
        self.assertGreater(w, 0)
        self.assertAlmostEqual(w, h, delta=1,
                               msg=f"Scope not square: {w}×{h}")

    def test_scope_in_strip(self):
        """K25b: Vectorscope ist im Wheel-Strip, nicht im Panel."""
        d = self.dialog
        self.app.processEvents()
        self.assertTrue(d._wheel_strip.isAncestorOf(d.vectorscope),
                        "Vectorscope should be in the wheel strip")
        self.assertFalse(d.panel.isAncestorOf(d.vectorscope),
                         "Vectorscope should NOT be in the panel")

    def test_histogram_in_scope_block(self):
        """K25b: Histogramm ist im Scope-Block (bottom), nicht im Panel."""
        d = self.dialog
        self.app.processEvents()
        self.assertTrue(d._scope_block.isAncestorOf(d.histogram),
                        "Histogram should be in the scope block")
        self.assertFalse(d.panel.isAncestorOf(d.histogram),
                         "Histogram should NOT be in the panel")

    def test_rgb_histogram_draws_channels(self):
        """Mit RGB-Daten zeichnet das Histogramm R, G und B (visible_channels)."""
        d = self.dialog
        self.app.processEvents()
        # Simulate worker response with RGB histogram data
        rgb_data = {
            "source": {"r": [0.1] * 128, "g": [0.2] * 128, "b": [0.3] * 128, "l": [0.15] * 128},
            "result": {"r": [0.12] * 128, "g": [0.22] * 128, "b": [0.32] * 128, "l": [0.17] * 128},
        }
        d.histogram.set_rgb_data(rgb_data)
        self.assertIn("r", d.histogram.visible_channels)
        self.assertIn("g", d.histogram.visible_channels)
        self.assertIn("b", d.histogram.visible_channels)


class TestSrgbHue(unittest.TestCase):
    """_srgb_hue: bekannte Farben liefern erwartete LCh-Hue-Werte."""

    def test_pure_red(self):
        """Reines Rot: LCh-Hue ≈ 0.111 (a>0, b>0, erster Quadrant)."""
        h = _srgb_hue(1.0, 0.0, 0.0)
        self.assertAlmostEqual(h, 0.1111, delta=0.001,
                               msg=f"Red LCh hue should be ~0.111, got {h}")

    def test_pure_green(self):
        """Reines Grün: LCh-Hue ≈ 0.378 (a<0, b<0, dritter Quadrant)."""
        h = _srgb_hue(0.0, 1.0, 0.0)
        self.assertAlmostEqual(h, 0.3778, delta=0.001,
                               msg=f"Green LCh hue should be ~0.378, got {h}")

    def test_pure_blue(self):
        """Reines Blau: LCh-Hue ≈ 0.851 (a>0, b<0, vierter Quadrant)."""
        h = _srgb_hue(0.0, 0.0, 1.0)
        self.assertAlmostEqual(h, 0.8508, delta=0.001,
                               msg=f"Blue LCh hue should be ~0.851, got {h}")

    def test_gray(self):
        """Grau: Chroma=0, Hue ist undefiniert (aber kein NaN/Inf)."""
        h = _srgb_hue(0.5, 0.5, 0.5)
        self.assertTrue(0.0 <= h < 1.0, f"Gray hue should be in [0,1), got {h}")
        self.assertFalse(math.isnan(h))

    def test_hue_range(self):
        """Alle Hue-Werte liegen in [0, 1)."""
        for r in (0.0, 0.5, 1.0):
            for g in (0.0, 0.5, 1.0):
                for b in (0.0, 0.5, 1.0):
                    h = _srgb_hue(r, g, b)
                    self.assertTrue(0.0 <= h < 1.0,
                                    f"RGB({r},{g},{b}) hue {h} out of range")


def _screenshot_hue_tab():
    """Screenshot: Hue-Tab aktiv."""
    app = QApplication.instance() or QApplication(sys.argv)
    tmp = Path(tempfile.mkdtemp(prefix="refgrade_k10b_ss_"))
    fake = FakeClient(tmp)
    source = _make_source(tmp)
    image = _make_image()
    d = RefGradeDialog(image, client=fake, source=source, on_ok=lambda r, n: None)
    d.resize(1920, 1080)
    d.show()
    _wait(app, lambda: d.isVisible())
    d.curves_editor.set_active("hue_hue")
    d.curves_editor.add_point_at(0.3, 0.7)
    d.curves_editor.add_point_at(0.7, 0.3)
    d._schedule_preview()
    _wait(app, lambda: fake.count("/preview") > 0, 2000)
    app.processEvents()
    img = d.grab().toImage()
    out = REPORT_DIR / "K10_hue_tab.png"
    img.save(str(out), "PNG")
    d.close()
    d.deleteLater()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return out


class TestScreenshots(unittest.TestCase):
    """Generiert Screenshots für den Bericht."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_screenshots(self):
        """Erzeugt K10_hue_tab.png."""
        p1 = _screenshot_hue_tab()
        self.assertTrue(p1.exists())
        self.assertGreater(p1.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
