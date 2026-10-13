"""Tests für core/refgrade/scope.py: rgb_parade, vectorscope, clipping."""

from __future__ import annotations

import unittest

import torch

from refgrade.scope import clipping, rgb_parade, vectorscope


class TestRgbParade(unittest.TestCase):

    def test_spec_case(self):
        """Bild 1×4×8×3, linke Hälfte R=1.0, rechte Hälfte R=0.0, G=B=0.5.
        cols=2, levels=4 → r[3,0]=1, r[0,1]=1, alle übrigen r-Zellen 0;
        g[2,*]=1 (0.5 fällt in Stufe 2 von 4)."""
        x = torch.zeros(1, 4, 8, 3)
        # Left half (cols 0-3): R = 1.0
        x[0, :, :4, 0] = 1.0
        # Right half (cols 4-7): R = 0.0 (already 0)
        # All: G = B = 0.5
        x[0, :, :, 1] = 0.5
        x[0, :, :, 2] = 0.5

        result = rgb_parade(x, cols=2, levels=4)

        # Check keys
        self.assertEqual(set(result.keys()), {"r", "g", "b"})

        # R: r[3,0] = 1 (left half, value 1.0 → level 3)
        self.assertAlmostEqual(result["r"][3, 0].item(), 1.0)
        # R: r[0,1] = 1 (right half, value 0.0 → level 0)
        self.assertAlmostEqual(result["r"][0, 1].item(), 1.0)
        # All other r cells are 0
        for l in range(4):
            for c in range(2):
                if (l, c) not in ((3, 0), (0, 1)):
                    self.assertAlmostEqual(result["r"][l, c].item(), 0.0)

        # G: g[2,*] = 1 (0.5 * 4 = 2 → level 2)
        for c in range(2):
            self.assertAlmostEqual(result["g"][2, c].item(), 1.0)
        # All other g cells are 0
        for l in range(4):
            if l != 2:
                for c in range(2):
                    self.assertAlmostEqual(result["g"][l, c].item(), 0.0)

        # B: same as G
        for c in range(2):
            self.assertAlmostEqual(result["b"][2, c].item(), 1.0)

    def test_shape(self):
        """Output shape is levels × cols for each channel."""
        x = torch.rand(1, 8, 16, 3)
        result = rgb_parade(x, cols=4, levels=8)
        for key in ("r", "g", "b"):
            self.assertEqual(result[key].shape, (8, 4))

    def test_uniform_image(self):
        """Uniform image: one level per channel is maxed to 1."""
        x = torch.full((1, 4, 4, 3), 0.5)
        result = rgb_parade(x, cols=2, levels=4)
        # 0.5 * 4 = 2 → level 2
        for key in ("r", "g", "b"):
            for c in range(2):
                self.assertAlmostEqual(result[key][2, c].item(), 1.0)
            for l in range(4):
                if l != 2:
                    for c in range(2):
                        self.assertAlmostEqual(result[key][l, c].item(), 0.0)

    def test_normalization(self):
        """Each channel is normalized so its max is 1."""
        x = torch.rand(1, 16, 32, 3)
        result = rgb_parade(x, cols=8, levels=16)
        for key in ("r", "g", "b"):
            self.assertAlmostEqual(result[key].max().item(), 1.0, places=5)

    def test_flat_image_no_nan(self):
        """Flat (single-value) image never produces NaN/Inf."""
        x = torch.zeros(1, 4, 4, 3)
        result = rgb_parade(x, cols=4, levels=4)
        for key in ("r", "g", "b"):
            self.assertTrue(torch.isfinite(result[key]).all().item())


class TestVectorscope(unittest.TestCase):

    def test_shape(self):
        """Output is bins × bins."""
        x = torch.rand(1, 8, 8, 3)
        result = vectorscope(x, bins=16)
        self.assertEqual(len(result), 16)
        self.assertEqual(len(result[0]), 16)

    def test_normalization(self):
        """Max value is 1."""
        x = torch.rand(1, 16, 16, 3)
        result = vectorscope(x, bins=32)
        max_val = max(max(row) for row in result)
        self.assertAlmostEqual(max_val, 1.0, places=5)


class TestClipping(unittest.TestCase):

    def test_spec_case(self):
        """10 % pure white in R, 3 % black in B → r_hi≈0.10, b_lo≈0.03."""
        # 10×10 image = 100 pixels
        x = torch.full((1, 10, 10, 3), 0.5)
        # 10 pixels (10 %) pure white in R
        x[0, :1, :, 0] = 1.0
        # 3 pixels (3 %) pure black in B
        x[0, 0, :3, 2] = 0.0
        result = clipping(x)
        self.assertAlmostEqual(result["r_hi"], 0.10, delta=0.001)
        self.assertAlmostEqual(result["b_lo"], 0.03, delta=0.001)
        # Others should be 0
        self.assertAlmostEqual(result["r_lo"], 0.0, delta=0.001)
        self.assertAlmostEqual(result["g_hi"], 0.0, delta=0.001)
        self.assertAlmostEqual(result["g_lo"], 0.0, delta=0.001)
        self.assertAlmostEqual(result["b_hi"], 0.0, delta=0.001)

    def test_no_clipping(self):
        """Mid-grey image: no clipping."""
        x = torch.full((1, 8, 8, 3), 0.5)
        result = clipping(x)
        for key in ("r_hi", "r_lo", "g_hi", "g_lo", "b_hi", "b_lo"):
            self.assertAlmostEqual(result[key], 0.0, delta=0.001)

    def test_all_keys_present(self):
        """All 6 keys are always present."""
        x = torch.rand(1, 4, 4, 3)
        result = clipping(x)
        self.assertEqual(set(result.keys()), {"r_hi", "r_lo", "g_hi", "g_lo", "b_hi", "b_lo"})


if __name__ == "__main__":
    unittest.main()
