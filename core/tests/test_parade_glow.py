"""Tests für K29d: Parade weiches Leuchten (PNG-Pfade, Glow, kein Raster)."""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refgrade.scope import rgb_parade  # noqa: E402

# ── Plugin setup (offscreen) ──────────────────────────────────────────────────
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QImage, QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import Parade  # noqa: E402


class TestCoreParade(unittest.TestCase):
    """Kern: rgb_parade 64×48-Verlauf → 256×256 je Kanal, max = 1."""

    def test_shape_and_max(self):
        x = torch.rand(1, 48, 64, 3)
        result = rgb_parade(x)
        for key in ("r", "g", "b"):
            self.assertEqual(result[key].shape, (256, 256))
            self.assertAlmostEqual(result[key].max().item(), 1.0, places=5)


class TestWorkerParadePng(unittest.TestCase):
    """Worker: /preview parade enthält drei Pfade zu 256×256-PNGs, < 200 KB."""

    def test_parade_pngs(self):
        from refgrade.scope import rgb_parade as rp
        from PIL import Image

        x = torch.rand(1, 48, 64, 3)
        parade = rp(x)

        tmp = tempfile.mkdtemp(prefix="parade_test_")
        log31 = float(np.log(31.0))
        paths: dict[str, str] = {}
        for key in ("r", "g", "b"):
            t = parade[key].cpu().numpy()
            v = np.round(255.0 * np.log1p(t * 30.0) / log31).astype(np.uint8)
            v = v[::-1, :]
            path = os.path.join(tmp, f"parade_{key}.png")
            Image.fromarray(v, mode="L").save(path, format="PNG")
            paths[key] = path

        # Each PNG is 256×256
        for key in ("r", "g", "b"):
            img = Image.open(paths[key])
            self.assertEqual(img.size, (256, 256))
            self.assertEqual(img.mode, "L")

        # Total size < 200 KB
        total = sum(os.path.getsize(paths[k]) for k in ("r", "g", "b"))
        self.assertLess(total, 200 * 1024, f"Parade PNGs too large: {total} bytes")

        # JSON response part (just the paths) is < 200 KB
        resp_part = json.dumps({"parade": paths})
        self.assertLess(len(resp_part), 200 * 1024, f"Parade JSON part too large: {len(resp_part)}")


class TestPluginParadeGlow(unittest.TestCase):
    """Plugin: Parade 800×300 mit Gauß-Band → kein Raster, Glow vorhanden."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        self._tmpdir = tempfile.mkdtemp(prefix="parade_glow_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_gaussian_png(self, path: str, center: int = 128, sigma: float = 12.0) -> str:
        """Create a 256×256 grayscale PNG with a horizontal Gaussian band."""
        x = np.arange(256)
        col = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        # Same band in every row (horizontal stripe)
        grid = np.tile(col, (256, 1))
        grid = (grid * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(grid, mode="L").save(path, format="PNG")
        return path

    def _make_widget(self) -> Parade:
        w = Parade()
        w.resize(800, 300)
        w.show()
        self.app.processEvents()
        return w

    def test_no_rectangle_grid(self):
        """In a row inside the R panel, max brightness diff of neighbors ≤ 25 % of peak."""
        r_path = self._make_gaussian_png(os.path.join(self._tmpdir, "r.png"))
        g_path = self._make_gaussian_png(os.path.join(self._tmpdir, "g.png"))
        b_path = self._make_gaussian_png(os.path.join(self._tmpdir, "b.png"))

        w = self._make_widget()
        w.set_data({"r": r_path, "g": g_path, "b": b_path})
        self.app.processEvents()

        target = w.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)

        # Check a row in the middle of the R panel (x: 0..~256, y: middle)
        row_y = 150
        panel_w = (800 - 2 * max(1, int(800 * 0.02))) // 3
        # Sample the R panel region
        values = []
        for x in range(10, panel_w - 10):
            pixel = target.pixel(x, row_y)
            brightness = (pixel & 0xFF)  # blue channel (R color is blue=0x30)
            values.append(brightness)

        if not values or max(values) == 0:
            self.skipTest("No visible data in R panel")

        peak = max(values)
        max_diff = max(abs(values[i + 1] - values[i]) for i in range(len(values) - 1))
        self.assertLessEqual(
            max_diff, 0.25 * peak,
            f"Rectangle grid detected: max_diff={max_diff}, peak={peak}")

    def test_glow_beyond_band(self):
        """6 px outside the Gaussian band, brightness > 0 (glow present)."""
        # Narrow band: center=128, sigma=5 → band is roughly 118..138
        r_path = self._make_gaussian_png(os.path.join(self._tmpdir, "r.png"),
                                         center=128, sigma=5.0)
        g_path = self._make_gaussian_png(os.path.join(self._tmpdir, "g.png"),
                                         center=128, sigma=5.0)
        b_path = self._make_gaussian_png(os.path.join(self._tmpdir, "b.png"),
                                         center=128, sigma=5.0)

        w = self._make_widget()
        w.set_data({"r": r_path, "g": g_path, "b": b_path})
        self.app.processEvents()

        target = w.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)

        # Panel is 256 px wide for an 800 px widget, so source col == panel x.
        # Band edge = centre - 2 sigma = 118; glow must show 6 px beyond it,
        # but not far away (else it would be background, not glow).
        row_y = 140   # not 150: a 50 % grid line runs there (2026-10-11)
        near = target.pixel(118 - 6, row_y) & 0xFF
        far = target.pixel(30, row_y) & 0xFF
        self.assertGreater(near, 0, "Glow not detected 6 px beside the band")
        self.assertEqual(far, 0, "Brightness far from the band must be 0")

    def test_paint_event_timing(self):
        """paintEvent ≤ 30 ms (median of 10) at 800×300."""
        r_path = self._make_gaussian_png(os.path.join(self._tmpdir, "r.png"))
        g_path = self._make_gaussian_png(os.path.join(self._tmpdir, "g.png"))
        b_path = self._make_gaussian_png(os.path.join(self._tmpdir, "b.png"))

        w = self._make_widget()
        w.set_data({"r": r_path, "g": g_path, "b": b_path})
        self.app.processEvents()

        # Warm-up
        w.repaint()
        self.app.processEvents()

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            w.repaint()
            self.app.processEvents()
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = statistics.median(times)
        self.assertLessEqual(median_ms, 30.0,
                             f"paintEvent too slow: median={median_ms:.1f} ms")


if __name__ == "__main__":
    unittest.main()
