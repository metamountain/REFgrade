"""Tests für refgrade.metrics."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from skimage import data as skdata

from refgrade.metrics import coverage, swd, delta_e00, structure, banding, lut_fidelity
from refgrade.transfer import reinhard
from refgrade.color import rgb_to_lab


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _make_image(seed=0, size=(1, 64, 64, 3), device=None):
    if device is None:
        device = _get_device()
    torch.manual_seed(seed)
    return torch.rand(*size, device=device)


class TestCoverage(unittest.TestCase):
    """Coverage: Histogramm-Intersection."""

    def test_flat_100(self):
        """Flaches Bild (Delta-Verteilung) → 100 %."""
        dev = _get_device()
        img = torch.full((1, 16, 16, 3), 0.5, device=dev)
        c = coverage(img, img)
        self.assertAlmostEqual(c, 100.0, delta=1.0, msg=f"coverage flach: {c}")

    def test_identical_geq_different(self):
        """Identisch ≥ unterschiedlich."""
        dev = _get_device()
        img1 = _make_image(1, (1, 32, 32, 3), dev)
        img2 = _make_image(2, (1, 32, 32, 3), dev)
        c_same = coverage(img1, img1)
        c_diff = coverage(img1, img2)
        self.assertGreaterEqual(c_same, c_diff - 0.01)
        self.assertGreater(c_same, 0.0)

    def test_with_weights(self):
        dev = _get_device()
        img = torch.full((1, 16, 16, 3), 0.5, device=dev)
        w = torch.ones(1, 16, 16, device=dev)
        c = coverage(img, img, image_weights=w, reference_weights=w)
        self.assertAlmostEqual(c, 100.0, delta=1.0)


class TestSWD(unittest.TestCase):
    """Sliced Wasserstein: identische Bilder → 0."""

    def test_identical_zero(self):
        dev = _get_device()
        img = _make_image(1, (1, 32, 32, 3), dev)
        d = swd(img, img)
        self.assertAlmostEqual(d, 0.0, delta=1e-5, msg=f"swd identisch: {d}")

    def test_different_positive(self):
        dev = _get_device()
        img1 = _make_image(1, (1, 32, 32, 3), dev)
        img2 = _make_image(2, (1, 32, 32, 3), dev)
        d = swd(img1, img2)
        self.assertGreater(d, 0.0)

    def test_deterministic(self):
        dev = _get_device()
        img1 = _make_image(1, (1, 32, 32, 3), dev)
        img2 = _make_image(2, (1, 32, 32, 3), dev)
        d1 = swd(img1, img2)
        d2 = swd(img1, img2)
        self.assertEqual(d1, d2, "swd nicht deterministisch")


class TestDeltaE00(unittest.TestCase):
    """ΔE00: identische Bilder → 0."""

    def test_identical_zero(self):
        dev = _get_device()
        img = _make_image(1, (1, 32, 32, 3), dev)
        mean, p95 = delta_e00(img, img)
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(p95, 0.0, places=5)

    def test_different_positive(self):
        dev = _get_device()
        img1 = _make_image(1, (1, 32, 32, 3), dev)
        img2 = _make_image(2, (1, 32, 32, 3), dev)
        mean, p95 = delta_e00(img1, img2)
        self.assertGreater(mean, 0.0)
        self.assertGreaterEqual(p95, mean)

    def test_with_mask(self):
        dev = _get_device()
        img1 = _make_image(1, (1, 32, 32, 3), dev)
        img2 = _make_image(2, (1, 32, 32, 3), dev)
        mask = torch.ones(1, 32, 32, device=dev)
        mean_full, _ = delta_e00(img1, img2)
        mean_masked, _ = delta_e00(img1, img2, mask=mask)
        self.assertAlmostEqual(mean_full, mean_masked, places=5)


class TestStructure(unittest.TestCase):
    """Structure: identische Bilder → 1."""

    def test_identical_one(self):
        dev = _get_device()
        img = _make_image(1, (1, 64, 64, 3), dev)
        s = structure(img, img)
        self.assertAlmostEqual(s, 1.0, places=5, msg=f"structure identisch: {s}")

    def test_different_less(self):
        dev = _get_device()
        img1 = _make_image(1, (1, 64, 64, 3), dev)
        img2 = _make_image(2, (1, 64, 64, 3), dev)
        s = structure(img1, img2)
        self.assertLess(s, 1.0)
        self.assertGreater(s, -1.0)

    def test_smooth_gradient(self):
        """Glatte Gradienten: hohe Korrelation."""
        dev = _get_device()
        # Linienbild
        img = torch.zeros(1, 64, 64, 3, device=dev)
        img[:, :, :32, :] = 0.3
        img[:, :, 32:, :] = 0.7
        # Leicht verrückte Version
        shifted = torch.zeros(1, 64, 64, 3, device=dev)
        shifted[:, :, :30, :] = 0.3
        shifted[:, :, 30:, :] = 0.7
        s = structure(img, shifted)
        self.assertGreater(s, 0.5)


class TestBanding(unittest.TestCase):
    """Banding: identische Bilder → 0."""

    def test_identical_zero(self):
        dev = _get_device()
        img = _make_image(1, (1, 64, 64, 3), dev)
        b = banding(img, img)
        self.assertAlmostEqual(b, 0.0, delta=0.01, msg=f"banding identisch: {b}")

    def test_quantized_higher(self):
        """Quantisiertes Bild → mehr Banding."""
        dev = _get_device()
        img = _make_image(1, (1, 64, 64, 3), dev)
        # Quantisieren auf 8 Stufen
        quantized = torch.round(img * 7.0) / 7.0
        b = banding(img, quantized)
        self.assertGreaterEqual(b, 0.0)

    def test_flat_zero(self):
        """Flaches Bild → 0 (keine belegten Bins)."""
        dev = _get_device()
        img = torch.full((1, 16, 16, 3), 0.5, device=dev)
        b = banding(img, img)
        self.assertAlmostEqual(b, 0.0, delta=0.01)


class TestLutFidelity(unittest.TestCase):
    """LUT-Fidelity: kleine Differenz."""

    def test_reinhard_lut(self):
        dev = _get_device()
        img = _make_image(1, (1, 32, 32, 3), dev)
        ref = _make_image(2, (1, 32, 32, 3), dev)
        fn = reinhard(img, ref)
        mean_diff, max_diff = lut_fidelity(img, fn, lut_size=33)
        self.assertLess(mean_diff, 0.02, f"LUT-Fidelity mean: {mean_diff}")
        self.assertLess(max_diff, 0.05, f"LUT-Fidelity max: {max_diff}")


class TestMetricsFlatImages(unittest.TestCase):
    """Flache Bilder → keine NaN/Inf."""

    def test_flat_all_metrics(self):
        dev = _get_device()
        img = torch.zeros(1, 16, 16, 3, device=dev)
        ref = torch.ones(1, 16, 16, 3, device=dev)

        c = coverage(img, ref)
        self.assertTrue(np.isfinite(c))

        d = swd(img, ref)
        self.assertTrue(np.isfinite(d))

        mean, p95 = delta_e00(img, ref)
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isfinite(p95))

        s = structure(img, ref)
        self.assertTrue(np.isfinite(s))

        b = banding(img, ref)
        self.assertTrue(np.isfinite(b))


if __name__ == "__main__":
    unittest.main()
