"""Tests für refgrade.color."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.color import srgb_to_linear, linear_to_srgb, rgb_to_lab, lab_to_rgb, luma


class TestColor(unittest.TestCase):
    def test_srgb_linear_roundtrip(self):
        """sRGB → linear → sRGB: max Fehler < 1e-6."""
        t = torch.rand(4, 32, 32, 3, dtype=torch.float64)
        rt = linear_to_srgb(srgb_to_linear(t))
        err = (rt - t).abs().max().item()
        self.assertLess(err, 1e-6, f"sRGB round-trip Fehler: {err}")

    def test_lab_roundtrip(self):
        """RGB → Lab → RGB: max Fehler < 1e-4."""
        t = torch.rand(2, 16, 16, 3, dtype=torch.float32)
        lab = rgb_to_lab(t)
        rt = lab_to_rgb(lab)
        err = (rt - t).abs().max().item()
        self.assertLess(err, 1e-4, f"Lab round-trip Fehler: {err}")

    def test_lab_roundtrip_extremes(self):
        """Lab-Round-Trip an Extremen (schwarz, weiß, gesättigt)."""
        cases = [
            torch.zeros(1, 8, 8, 3),
            torch.ones(1, 8, 8, 3),
            torch.tensor([[[[1.0, 0.0, 0.0]]]] * 8 * 8).reshape(1, 8, 8, 3),
            torch.tensor([[[[0.0, 1.0, 0.0]]]] * 8 * 8).reshape(1, 8, 8, 3),
            torch.tensor([[[[0.0, 0.0, 1.0]]]] * 8 * 8).reshape(1, 8, 8, 3),
        ]
        for i, t in enumerate(cases):
            lab = rgb_to_lab(t)
            rt = lab_to_rgb(lab)
            err = (rt - t).abs().max().item()
            self.assertLess(err, 1e-4, f"Lab round-trip Fall {i}: Fehler {err}")

    def test_luma(self):
        """Rec.709-Luma: 0.2126R + 0.7152G + 0.0722B."""
        t = torch.tensor([[[[0.5, 0.5, 0.5]]]])
        l = luma(t)
        expected = 0.2126 * 0.5 + 0.7152 * 0.5 + 0.0722 * 0.5
        self.assertAlmostEqual(l.item(), expected, places=6)

    def test_luma_shape(self):
        """Luma-Ausgabe hat Shape [B,H,W,1]."""
        t = torch.rand(2, 16, 24, 3)
        l = luma(t)
        self.assertEqual(l.shape, (2, 16, 24, 1))

    def test_srgb_to_linear_known_values(self):
        """Bekannte Werte: 0 → 0, 1 → 1, 0.04045 → ~0.0031308."""
        t = torch.tensor([[[[0.0, 0.04045, 1.0]]]])
        lin = srgb_to_linear(t)
        self.assertAlmostEqual(lin[0, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(lin[0, 0, 0, 1].item(), 0.04045 / 12.92, places=4)
        self.assertAlmostEqual(lin[0, 0, 0, 2].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
