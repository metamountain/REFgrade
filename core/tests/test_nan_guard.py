"""Regression 2026-10-07 (user's Krita test): FIT returned NaN for every slider.

Cause: linear_to_srgb evaluated t ** (1/2.4) at t = 0 in the branch torch.where discards;
its derivative is inf and inf·0 = NaN, so any image with pure black broke the gradient.
"""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade import color
from refgrade.grade import grade


class TestNaNGuard(unittest.TestCase):

    def test_linear_to_srgb_values_unchanged(self):
        t = torch.linspace(0, 1, 10001)
        old = torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t ** (1 / 2.4) - 0.055)
        self.assertEqual(float((color.linear_to_srgb(t) - old).abs().max()), 0.0)

    def test_gradient_finite_with_black_pixels(self):
        x = torch.rand(1, 1, 256, 3, generator=torch.Generator().manual_seed(0))
        x[0, 0, :32] = 0.0                                   # pure black
        names = ("exposure", "contrast", "saturation", "gamma_m", "lift_r", "gain_b")
        params = {n: torch.tensor(v, requires_grad=True) for n, v in
                  zip(names, (0.0, 1.0, 1.0, 1.0, 0.0, 1.0))}
        grade(x, params).clamp(0, 1).mean().backward()
        for n, p in params.items():
            self.assertTrue(math.isfinite(float(p.grad)), f"gradient of {n} is {p.grad}")


if __name__ == "__main__":
    unittest.main()
