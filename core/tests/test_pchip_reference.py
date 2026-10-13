"""Regression 2026-10-08: the curve editor's hue curves swung far outside their points
(-1.2 … 2.1 for points in 0.2 … 0.75) — wrong tangent sign on falling segments and slopes
not scaled by the interval width, in core AND plugin alike, so core-vs-plugin tests passed.

Reference is scipy's PchipInterpolator; plus the defining property: no overshoot.
"""

import random
import sys
import types
import unittest
from pathlib import Path

import torch
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", pkg)

from refgrade import huecurves  # noqa: E402
from refgrade.curves import pchip_eval  # noqa: E402
from ref_grade import pchip  # noqa: E402


def _random_curves(n_curves: int = 60, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n_curves):
        n = rng.randint(2, 8)
        xs = sorted(rng.sample([i / 40 for i in range(41)], n))
        yield xs, [rng.random() for _ in xs]


class TestPCHIPReference(unittest.TestCase):

    def test_core_and_plugin_match_scipy(self):
        for xs, ys in _random_curves():
            q = [xs[0] + (xs[-1] - xs[0]) * i / 200 for i in range(201)]
            ref = PchipInterpolator(xs, ys)(q)
            core = pchip_eval(torch.tensor(q, dtype=torch.float64), xs, ys).numpy()
            self.assertLess(abs(ref - core).max(), 1e-9)
            plug = max(abs(r - pchip.pchip_eval(xs, ys, v)) for r, v in zip(ref, q))
            self.assertLess(plug, 1e-9)

    def test_no_overshoot_cyclic(self):
        for xs, ys in _random_curves(seed=1):
            q = torch.linspace(0, 1, 401, dtype=torch.float64)
            v = huecurves._cyclic_pchip(q, [[a, b] for a, b in zip(xs, ys)])
            self.assertGreaterEqual(float(v.min()), min(ys) - 1e-9)
            self.assertLessEqual(float(v.max()), max(ys) + 1e-9)
            self.assertAlmostEqual(float(v[0]), float(v[-1]), places=9)    # seamless

    def test_user_case(self):
        xs, ys = [0.0, 0.15, 0.42, 0.7, 1.0], [0.5, 0.75, 0.2, 0.55, 0.5]
        curve = pchip.pchip_cyclic_curve(xs, ys, 401)
        self.assertGreaterEqual(min(curve), 0.2 - 1e-9)
        self.assertLessEqual(max(curve), 0.75 + 1e-9)


if __name__ == "__main__":
    unittest.main()
