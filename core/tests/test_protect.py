"""Tonal protection reduces the colour match per zone (shadows / midtones / highlights)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.protect import protect
from refgrade.color import rgb_to_lab
from refgrade.zonal import zone_weights


class TestProtect(unittest.TestCase):
    def setUp(self):
        g = torch.linspace(0.02, 0.98, 64)
        self.orig = torch.stack([g, g, g], -1).reshape(1, 8, 8, 3)
        self.matched = (self.orig * torch.tensor([1.2, 1.0, 0.7])).clamp(0, 1)

    def test_all_zero_is_identity(self):
        out = protect(self.orig, self.matched, {})
        self.assertTrue(torch.equal(out, self.matched))

    def test_full_protection_everywhere_returns_original(self):
        v = {"protect_shadows": 1.0, "protect_midtones": 1.0, "protect_highlights": 1.0}
        out = protect(self.orig, self.matched, v)
        self.assertLess((out - self.orig).abs().max().item(), 1e-5)

    def test_midtones_reduce_correction_most_in_the_middle(self):
        out = protect(self.orig, self.matched, {"protect_midtones": 1.0})
        _, _, w_mid = zone_weights(rgb_to_lab(self.orig)[..., 0])
        left = ((out - self.orig).abs().sum(-1) / (self.matched - self.orig).abs().sum(-1).clamp_min(1e-6))
        mid = w_mid > 0.9
        self.assertTrue(mid.any())
        self.assertLess(left[mid].max().item(), 0.1)          # middle: almost no correction left
        self.assertGreater(left[w_mid < 0.1].min().item(), 0.9)  # ends: correction untouched

    def test_half_protection_halves_the_change_in_its_zone(self):
        out = protect(self.orig, self.matched, {"protect_shadows": 0.5})
        w_sh, _, _ = zone_weights(rgb_to_lab(self.orig)[..., 0])
        deep = w_sh > 0.99
        ratio = (out - self.orig)[deep].abs().sum() / (self.matched - self.orig)[deep].abs().sum()
        self.assertAlmostEqual(ratio.item(), 0.5, delta=0.02)


if __name__ == "__main__":
    unittest.main()
