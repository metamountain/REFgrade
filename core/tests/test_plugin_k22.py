"""Tests für Paket K22: Arbeitsreihenfolge, Räder unten fixiert, Scope-Block."""

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

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


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


class FakeClient:
    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png,
                    "histogram": {"source": [0.015625] * 64, "result": [0.02] * 64},
                    "ms": 1.0}
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


class K22TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k22_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()
        self.dlg = RefGradeDialog(
            self.image,
            client=self.fake,
            source=self.source,
            on_ok=lambda raw, name: None,
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestSectionOrder(K22TestBase):
    """Reihenfolge der Bereichsköpfe (y aufsteigend): color, tone, hsl, curves, lut, reference."""

    def test_order_by_y(self):
        d = self.dlg
        self.app.processEvents()
        expected = ["color", "tone", "hsl", "curves", "lut", "reference"]
        ys = {}
        for key in expected:
            header = d.sections[key][0]
            y = header.mapTo(d, QPoint(0, 0)).y()
            ys[key] = y
        ordered = sorted(ys, key=ys.get)
        self.assertEqual(ordered, expected,
                         f"Section order wrong: {ordered} (ys={ys})")

    def test_wheels_between_curves_and_lut(self):
        """K29 (user 2026-10-11, reverses K25b): WHEELS chapter sits with the wheels in the
        wheel strip, no longer between CURVES and LUT in the column."""
        d = self.dlg
        self.app.processEvents()
        self.assertFalse(d.panel.isAncestorOf(d.sections["wheels"][0]))


class TestColorSection(K22TestBase):
    """WB · COLOR: Titel, ON bei Start, aufgeklappt."""

    def test_title_wb_color(self):
        d = self.dlg
        header = d.sections["color"][0]
        self.assertEqual(header._title.text(), "WB · COLOR")

    def test_color_on_at_start(self):
        d = self.dlg
        self.assertFalse(d.sections["color"][0].is_on(),
                         "Color starts OFF (2026-10-11), ON once something changes")

    def test_color_expanded_at_start(self):
        d = self.dlg
        self.app.processEvents()
        # Color section should not be collapsed at start
        header = d.sections["color"][0]
        self.assertFalse(header.is_collapsed(), "Color section should be expanded at start")


class TestWheelsNoTriangle(K22TestBase):
    """WHEELS hat kein Einklapp-Dreieck."""

    def test_no_triangle(self):
        d = self.dlg
        header = d.sections["wheels"][0]
        self.assertFalse(header.triangle.isVisible(),
                         "Wheels triangle should be hidden")

    def test_set_collapsed_noop(self):
        d = self.dlg
        self.app.processEvents()
        # Collapsing wheels should have no effect (not in scroll keys)
        header = d.sections["wheels"][0]
        header.set_collapsed(True)
        self.app.processEvents()
        # Wheels content should still be visible
        self.assertTrue(d.wheels["lift"].isVisible(),
                        "Wheels should remain visible after set_collapsed")


class TestScopeBlock(K22TestBase):
    """K25b: Scope-Block (bottom) — Knopfraster + RESET/CANCEL/OK + Histogramm.
    Vectorscope ist jetzt im Wheel-Strip (ein Quadrat von 4)."""

    def test_scope_block_has_buttons_and_histogram(self):
        d = self.dlg
        self.app.processEvents()
        # Scope buttons and histogram are in the scope block (bottom)
        self.assertTrue(d._scope_block.isAncestorOf(d.histogram),
                        "Histogram should be in the scope block")
        for name, btn in d._scope_btns.items():
            self.assertTrue(d._scope_block.isAncestorOf(btn),
                            f"Button {name} should be in the scope block")

    def test_vectorscope_in_strip(self):
        """K25b: Vectorscope is in the wheel strip (one of the 4 squares)."""
        d = self.dlg
        self.app.processEvents()
        self.assertTrue(d._wheel_strip.isAncestorOf(d.vectorscope),
                        "Vectorscope should be in the wheel strip")

    def test_scope_right_edge_flush_with_strip(self):
        """V-Layout: rechte Kante des Scope-Blocks == rechte Kante der Wheel-Spalte."""
        d = self.dlg
        d._set_layout_mode("V")
        self.app.processEvents()
        self.app.processEvents()
        scope_right = d._scope_block.x() + d._scope_block.width()
        strip_right = d._wheel_strip.x() + d._wheel_strip.width()
        self.assertEqual(scope_right, strip_right,
                         f"Scope right {scope_right} != strip right {strip_right}")


class TestSizes1366(K22TestBase):
    """1366×768: Vectorscope ≥ 120, Räder ≥ 70, Spalte scrollt nicht."""

    def test_vectorscope_min_120(self):
        """K29: the vectorscope is as large as a wheel disc (user 2026-10-11), so its floor is
        the wheels' floor (70), no longer the old 120."""
        d = self.dlg
        d.resize(1366, 768)
        self.app.processEvents()
        self.app.processEvents()
        self.assertGreaterEqual(d.vectorscope.height(), 70,
                                f"Vectorscope {d.vectorscope.height()} < 70 at 1366×768")

    def test_wheels_min_70(self):
        d = self.dlg
        d.resize(1366, 768)
        self.app.processEvents()
        self.app.processEvents()
        wheel = d.wheels["lift"]
        self.assertGreaterEqual(wheel.width(), 70,
                                f"Wheel width {wheel.width()} < 70 at 1366×768")

    def test_no_scroll(self):
        d = self.dlg
        d.resize(1366, 768)
        self.app.processEvents()
        self.app.processEvents()
        # The scroll widget's height should fit within the panel
        scroll_h = d._scroll.height()
        panel_h = d.panel.height()
        self.assertLessEqual(scroll_h, panel_h,
                             f"Scroll height {scroll_h} > panel height {panel_h}")


class TestNoScroll1920(K22TestBase):
    """1920×1080: Spalte scrollt nicht, nichts überlappt."""

    def test_no_scroll(self):
        d = self.dlg
        self.app.processEvents()
        scroll_h = d._scroll.height()
        panel_h = d.panel.height()
        self.assertLessEqual(scroll_h, panel_h,
                             f"Scroll height {scroll_h} > panel height {panel_h}")

    def test_vectorscope_size_matches_s(self):
        """K29 (user 2026-10-11): vectorscope circle exactly as large as a wheel disc."""
        d = self.dlg
        self.app.processEvents()
        _c, r = d.wheels["gain"]._geometry()
        self.assertAlmostEqual(d.vectorscope.height(), 2 * r, delta=1,
                               msg=f"Vectorscope {d.vectorscope.height()} vs disc {2 * r:.0f}")


if __name__ == "__main__":
    unittest.main()
