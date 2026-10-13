"""Schnelle Plausibilitätstests für den PAIR-Bench (< 5 s gesamt)."""

import os
import sys
import subprocess
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from pair_grades import GRADES, gamma_up
from pair_strategies import strat_splat, strat_hybrid
from refgrade.color import rgb_to_lab


class TestPairBench(unittest.TestCase):

    def test_grades_range_and_determinism(self):
        """Jede Ground-Truth-Funktion: Ausgang in 0..1, deterministisch, nicht die Identität."""
        torch.manual_seed(0)
        x = torch.rand(4, 8, 8, 3)  # [B,H,W,3]

        for name, fn in GRADES.items():
            out1 = fn(x)
            out2 = fn(x)
            self.assertGreaterEqual(out1.min().item(), -0.01, f"{name}: min < 0")
            self.assertLessEqual(out1.max().item(), 1.01, f"{name}: max > 1")
            self.assertTrue(torch.allclose(out1, out2, atol=1e-6), f"{name}: nicht deterministisch")
            diff = (out1 - x).abs().mean().item()
            self.assertGreater(diff, 0.005, f"{name}: zu nah an Identität (diff={diff:.6f})")

    def test_splat_identity(self):
        """splat auf das Identitätspaar (16 Bit): Die Tabelle ist das Gitter (< 1e-3)."""
        N = 33
        grid = torch.stack(
            torch.meshgrid(*[torch.linspace(0, 1, N)] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        grid_q = torch.round(grid * 65535.0) / 65535.0

        lut = strat_splat(grid_q, grid_q, N)
        identity = torch.stack(
            torch.meshgrid(*[torch.linspace(0, 1, N)] * 3, indexing="ij"), -1
        )
        err = (lut - identity).abs().max().item()
        self.assertLess(err, 1e-3, f"splat Identität: max err = {err:.6f} > 1e-3")

    def test_hybrid_gamma(self):
        """hybrid auf gamma_up, 16 Bit, 128² Bild: dE_other < 3.0."""
        torch.manual_seed(42)
        img = torch.rand(128, 128, 3)
        img_q = torch.round(img * 65535.0) / 65535.0

        before = img_q.reshape(-1, 3)
        after = gamma_up(img_q).reshape(-1, 3)

        lut = strat_hybrid(before, after, 33)

        torch.manual_seed(99)
        test_img = torch.rand(64, 64, 3)
        test_target = gamma_up(test_img)

        N = 33
        g = test_img * (N - 1)
        i0 = g.floor().clamp(0, N - 2).long()
        f = g - i0
        out = torch.zeros_like(test_img)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[..., 0] if dx else 1 - f[..., 0]) *
                         (f[..., 1] if dy else 1 - f[..., 1]) *
                         (f[..., 2] if dz else 1 - f[..., 2]))[..., None]
                    out += w * lut[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]

        lab_o = rgb_to_lab(out.unsqueeze(0))
        lab_t = rgb_to_lab(test_target.unsqueeze(0))
        dE = (lab_o - lab_t).norm(dim=-1).flatten().mean().item()
        self.assertLess(dE, 3.0, f"hybrid gamma_up: dE_other = {dE:.3f} >= 3.0")

    def test_quick_run(self):
        """pair_bench.py --quick läuft ohne Fehler durch."""
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools")
        result = subprocess.run(
            [sys.executable, os.path.join(tools_dir, "pair_bench.py"), "--quick"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, f"--quick fehlgeschlagen:\n{result.stderr[-500:]}")


if __name__ == "__main__":
    unittest.main()
