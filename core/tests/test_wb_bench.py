"""Tests für wb_bench: Schätzer liefern kleinen Winkelfehler auf synthetischen Bildern.

Testbild: 2D-Linear-Gradient (X+Y) in linearem RGB, R=G=B (neutral), max 0.5 linear
(keine Clipping bei 3200 K-Stich). Der sRGB-Roundtrip ist exakt (float32).

Hinweis zu `ours`: Der Lab-basierte Schätzer hat eine inhärente Nichtlinearität von
~5–7° auf synthetischen Bildern, weil a*/b* in Lab nicht linear in der Lichtfarbe sind.
Die anderen Schätzer (grayworld, whitepatch, cheng, greyedge) arbeiten direkt im
linearen RGB und liefern < 0.02° auf diesem Testbild.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from slow import slow
from tools.wb_methods import ESTIMATORS, _d65_illuminant, _blackbody_illuminant, _srgb_to_linear_np
from tools.wb_bench import _linear_to_srgb_np, angular_error_deg

_SIZE = 128


def _neutral_2d_gradient() -> np.ndarray:
    """Synthetisches neutrales Bild: 2D-Gradient (X+Y), R=G=B, linear 0.02..0.5."""
    x = np.linspace(0, 1, _SIZE)
    y = np.linspace(0, 1, _SIZE)
    X, Y = np.meshgrid(x, y)
    pattern = (0.02 + 0.24 * X + 0.24 * Y).astype(np.float32)
    linear = np.stack([pattern] * 3, axis=-1)
    return _linear_to_srgb_np(linear).astype(np.float32)


def _exact_cast(rgb: np.ndarray, ill: np.ndarray, d65: np.ndarray) -> np.ndarray:
    """Stich anwenden ohne Quantisierung: sRGB→linear, ·ill/d65, →sRGB."""
    linear = _srgb_to_linear_np(rgb)
    return _linear_to_srgb_np(linear * (ill / d65)).astype(np.float32)


class TestWBCast(unittest.TestCase):
    """Neutrales 2D-Gradienten-Bild mit 3200 K-Stich."""

    @classmethod
    def setUpClass(cls):
        cls.d65 = _d65_illuminant()
        cls.ill = _blackbody_illuminant(3200)
        cls.img = _neutral_2d_gradient()
        cls.cast = _exact_cast(cls.img, cls.ill, cls.d65)

    def test_all_estimators_finite(self):
        """Kein NaN/Inf, Summe = 1."""
        for name, fn in ESTIMATORS.items():
            est = fn(self.cast[np.newaxis, ...])
            self.assertTrue(np.all(np.isfinite(est)), f"{name}: non-finite {est}")
            self.assertAlmostEqual(float(est.sum()), 1.0, places=4, msg=f"{name}: sum≠1")

    def test_linear_estimators_under_1deg(self):
        """grayworld, whitepatch, cheng, greyedge < 1° (arbeiten im linearen RGB)."""
        linear_names = [n for n in ESTIMATORS if n not in ("ours",)]
        for name in linear_names:
            est = ESTIMATORS[name](self.cast[np.newaxis, ...])
            err = angular_error_deg(est, self.ill)
            self.assertLess(err, 1.0, f"{name}: {err:.3f}° >= 1°")

    def test_ours_under_10deg(self):
        """ours < 10° (Lab-Nichtlinearität, typisch 5–7°)."""
        est = ESTIMATORS["ours"](self.cast[np.newaxis, ...])
        err = angular_error_deg(est, self.ill)
        self.assertLess(err, 10.0, f"ours: {err:.3f}° >= 10°")


class TestWBNoCast(unittest.TestCase):
    """Ohne Stich: ours und grayworld_0 < 2° auf neutralem Testbild."""

    @classmethod
    def setUpClass(cls):
        cls.d65 = _d65_illuminant()
        cls.img = _neutral_2d_gradient()

    def test_ours_no_cast(self):
        est = ESTIMATORS["ours"](self.img[np.newaxis, ...])
        err = angular_error_deg(est, self.d65)
        self.assertLess(err, 2.0, f"ours no-cast: {err:.3f}° >= 2°")

    def test_grayworld0_no_cast(self):
        est = ESTIMATORS["grayworld_0"](self.img[np.newaxis, ...])
        err = angular_error_deg(est, self.d65)
        self.assertLess(err, 2.0, f"grayworld_0 no-cast: {err:.3f}° >= 2°")


@slow
class TestWBQuick(unittest.TestCase):
    """wb_bench.py --quick läuft und schreibt jsonl."""

    def test_quick_runs(self):
        core_dir = Path(__file__).resolve().parents[1]
        reports = core_dir.parent / "reports"
        jsonl = reports / "wb_bench.jsonl"
        if jsonl.exists():
            jsonl.unlink()
        proc = subprocess.run(
            [sys.executable, str(core_dir / "tools" / "wb_bench.py"), "--quick"],
            capture_output=True, text=True, timeout=60,
            cwd=str(core_dir),
        )
        self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr[-500:]}")
        self.assertTrue(jsonl.exists(), "jsonl not written")
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(lines), 0, "jsonl is empty")
        first = json.loads(lines[0])
        self.assertIn("image", first)
        self.assertIn("illuminant", first)


if __name__ == "__main__":
    unittest.main()
