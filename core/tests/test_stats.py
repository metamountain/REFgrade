"""Tests für refgrade.stats und refgrade.masks."""

import sys
import os
import unittest
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.stats import (
    weighted_mean, weighted_std, weighted_covariance,
    weighted_quantile, sample,
)
from refgrade.masks import to_weights


class TestWeightedStats(unittest.TestCase):
    def test_weighted_mean_matches_unmasked(self):
        """Gewichtete Statistik mit Maske == Statistik der nicht maskierten Pixel."""
        torch.manual_seed(42)
        n = 5000
        values = torch.rand(n, 3)
        # Maske: 30 % der Pixel ignorieren
        mask = (torch.rand(n) < 0.3).float()
        weights = 1.0 - mask

        # Gewichtete Statistik
        wm = weighted_mean(values, weights)
        ws = weighted_std(values, weights)

        # Vergleich: nur nicht maskierte Pixel
        selected = values[weights > 0.5]
        um = selected.mean(dim=0)
        us = selected.std(dim=0, unbiased=False)

        err_mean = (wm - um).abs().max().item()
        err_std = (ws - us).abs().max().item()
        self.assertLess(err_mean, 1e-5, f"Mittel-Fehler: {err_mean}")
        self.assertLess(err_std, 1e-5, f"Std-Fehler: {err_std}")

    def test_weighted_covariance_shape_dtype(self):
        """Kovarianz ist [3,3] float64, symmetrisch, positive definit."""
        torch.manual_seed(0)
        n = 1000
        values = torch.rand(n, 3)
        weights = torch.rand(n) + 0.1
        cov = weighted_covariance(values, weights)
        self.assertEqual(cov.shape, (3, 3))
        self.assertEqual(cov.dtype, torch.float64)
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-10))

    def test_weighted_quantile_ordering(self):
        """Quantile sind monoton nicht-fallend."""
        torch.manual_seed(1)
        n = 2000
        values = torch.rand(n)
        weights = torch.ones(n)
        q = torch.linspace(0.01, 0.99, 50)
        result = weighted_quantile(values, weights, q)
        diffs = result[1:] - result[:-1]
        self.assertTrue((diffs >= -1e-10).all(), "Quantile nicht monoton")

    def test_sample_determinism(self):
        """Sample mit gleichem Seed → bitgleiches Ergebnis."""
        torch.manual_seed(7)
        img = torch.rand(2, 32, 32, 3)
        w = torch.rand(2, 32, 32)
        p1, w1 = sample(img, w, n=5000, seed=0)
        p2, w2 = sample(img, w, n=5000, seed=0)
        self.assertTrue(torch.equal(p1, p2))
        self.assertTrue(torch.equal(w1, w2))

    def test_sample_different_seeds(self):
        """Verschiedene Seeds → unterschiedliche Samples."""
        img = torch.rand(2, 64, 64, 3)  # 8192 Pixel > n
        p1, _ = sample(img, None, n=5000, seed=0)
        p2, _ = sample(img, None, n=5000, seed=1)
        self.assertFalse(torch.equal(p1, p2))

    def test_sample_no_subsample(self):
        """Wenn Pixel < n: alle zurück, keine Subsampling."""
        img = torch.rand(1, 4, 4, 3)  # 16 Pixel
        p, w = sample(img, None, n=1_000_000)
        self.assertEqual(p.shape[0], 16)
        self.assertTrue(torch.all(w == 1.0))


class TestMasks(unittest.TestCase):
    def test_to_weights_basic(self):
        """Maske 0 → Gewicht 1, Maske 1 → Gewicht 0."""
        mask = torch.zeros(1, 16, 16)
        like = torch.rand(1, 16, 16, 3)
        w = to_weights(mask, like)
        self.assertTrue(torch.all(w == 1.0))

        mask = torch.ones(1, 16, 16)
        # 100% ignoriert → verworfen, Gewicht 1
        w = to_weights(mask, like)
        self.assertTrue(torch.all(w == 1.0))

    def test_to_weights_partial(self):
        """Teilmaske: Gewicht = 1 - mask."""
        mask = torch.full((1, 16, 16), 0.5)
        like = torch.rand(1, 16, 16, 3)
        w = to_weights(mask, like)
        self.assertTrue(torch.allclose(w, torch.full_like(w, 0.5)))

    def test_to_weights_99_percent_discarded(self):
        """Maske ≥ 99 % → verworfen + Warnung."""
        mask = torch.full((1, 32, 32), 0.995)  # 99.5 % ignoriert
        like = torch.rand(1, 32, 32, 3)

        with self.assertLogs("RefGrade", level="WARNING") as cm:
            w = to_weights(mask, like)
        self.assertTrue(torch.all(w == 1.0))
        self.assertTrue(any("verworfen" in msg for msg in cm.output))

    def test_to_weights_resize(self):
        """Maske anderer Größe wird bilinear skaliert."""
        mask = torch.zeros(1, 8, 8)
        like = torch.rand(1, 32, 32, 3)
        w = to_weights(mask, like)
        self.assertEqual(w.shape, (1, 32, 32))
        self.assertTrue(torch.all(w == 1.0))

    def test_to_weights_batch_expand(self):
        """Maske [1,H,W] wird auf Batch B expandiert."""
        mask = torch.zeros(1, 16, 16)
        like = torch.rand(4, 16, 16, 3)
        w = to_weights(mask, like)
        self.assertEqual(w.shape, (4, 16, 16))

    def test_to_weights_2d_mask(self):
        """2D-Maske [H,W] wird als [1,H,W] behandelt."""
        mask = torch.zeros(16, 16)
        like = torch.rand(2, 16, 16, 3)
        w = to_weights(mask, like)
        self.assertEqual(w.shape, (2, 16, 16))


if __name__ == "__main__":
    unittest.main()
