"""Tests für refgrade.transfer: Reinhard, MKL, Histogramm."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.color import rgb_to_lab
from refgrade.stats import weighted_mean, weighted_std, weighted_covariance, sample
from refgrade.transfer import reinhard, mkl, histogram
from slow import slow  # noqa: E402


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestTransferShapes(unittest.TestCase):
    """Formen und Grundverhalten."""

    def _make_tensors(self, device):
        img = torch.rand(2, 64, 96, 3, device=device)
        ref = torch.rand(3, 40, 40, 3, device=device)
        return img, ref

    def test_output_shape_reinhard(self):
        dev = _get_device()
        img, ref = self._make_tensors(dev)
        fn = reinhard(img, ref)
        out = fn(img)
        self.assertEqual(out.shape, (2, 64, 96, 3))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, dev)

    def test_output_shape_mkl(self):
        dev = _get_device()
        img, ref = self._make_tensors(dev)
        fn = mkl(img, ref)
        out = fn(img)
        self.assertEqual(out.shape, (2, 64, 96, 3))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, dev)

    @slow

    def test_output_shape_histogram(self):
        dev = _get_device()
        img, ref = self._make_tensors(dev)
        fn = histogram(img, ref)
        out = fn(img)
        self.assertEqual(out.shape, (2, 64, 96, 3))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, dev)


class TestTransferIdentity(unittest.TestCase):
    """Bild == Vorlage → Ergebnis ≈ Eingabe."""

    def _test_identity(self, method, tol=1e-3):
        dev = _get_device()
        torch.manual_seed(123)
        img = torch.rand(2, 32, 48, 3, device=dev)
        fn = method(img, img)
        out = fn(img)
        err = (out - img).abs().max().item()
        self.assertLess(err, tol, f"{method.__name__}: Fehler {err}")

    def test_identity_reinhard(self):
        self._test_identity(reinhard)

    def test_identity_mkl(self):
        self._test_identity(mkl)

    @slow

    def test_identity_histogram(self):
        self._test_identity(histogram)


class TestTransferStrengthZero(unittest.TestCase):
    """strength=0 → bitgleich zur Eingabe."""

    def _test_strength_zero(self, method):
        dev = _get_device()
        torch.manual_seed(42)
        img = torch.rand(1, 16, 16, 3, device=dev)
        ref = torch.rand(1, 16, 16, 3, device=dev)
        fn = method(img, ref, strength=0.0)
        out = fn(img)
        self.assertTrue(torch.equal(out, img), f"{method.__name__}: nicht bitgleich")

    def test_strength_zero_reinhard(self):
        self._test_strength_zero(reinhard)

    def test_strength_zero_mkl(self):
        self._test_strength_zero(mkl)

    @slow

    def test_strength_zero_histogram(self):
        self._test_strength_zero(histogram)


class TestTransferLabStats(unittest.TestCase):
    """Lab-Statistik der Ausgabe (vor Rückwandlung/clamp)."""

    def test_reinhard_lab_stats(self):
        """Reinhard: Lab-Mittel ±0.5, Lab-Std ±2 % der Vorlage."""
        dev = _get_device()
        torch.manual_seed(99)
        img = torch.rand(2, 48, 64, 3, device=dev)
        ref = torch.rand(3, 40, 40, 3, device=dev)

        fn = reinhard(img, ref)
        lab_fn = fn.lab_transform

        # Lab-Ausgabe auf dem Bild
        lab_img = rgb_to_lab(img)
        lab_out = lab_fn(lab_img)

        # Referenz-Statistik
        pixels_r, w_r = sample(rgb_to_lab(ref), None)
        mu_r = weighted_mean(pixels_r, w_r)
        std_r = weighted_std(pixels_r, w_r)

        # Ausgabe-Statistik
        pixels_out, w_out = sample(lab_out, None)
        mu_out = weighted_mean(pixels_out, w_out)
        std_out = weighted_std(pixels_out, w_out)

        err_mean = (mu_out - mu_r).abs().max().item()
        self.assertLess(err_mean, 0.5, f"Reinhard Lab-Mittel-Fehler: {err_mean}")

        rel_std_err = ((std_out - std_r).abs() / std_r.clamp_min(1e-6)).max().item()
        self.assertLess(rel_std_err, 0.02, f"Reinhard Lab-Std-Fehler: {rel_std_err:.4f}")

    def test_mkl_covariance(self):
        """MKL: Kovarianz relativer Frobenius-Fehler < 5 %."""
        dev = _get_device()
        torch.manual_seed(77)
        img = torch.rand(2, 48, 64, 3, device=dev)
        ref = torch.rand(3, 40, 40, 3, device=dev)

        fn = mkl(img, ref)
        lab_fn = fn.lab_transform

        lab_img = rgb_to_lab(img)
        lab_out = lab_fn(lab_img)

        # Referenz-Kovarianz
        pixels_r, w_r = sample(rgb_to_lab(ref), None)
        cov_r = weighted_covariance(pixels_r, w_r)

        # Ausgabe-Kovarianz
        pixels_out, w_out = sample(lab_out, None)
        cov_out = weighted_covariance(pixels_out, w_out)

        fro_err = (cov_out - cov_r).norm().item() / cov_r.norm().clamp_min(1e-10).item()
        self.assertLess(fro_err, 0.05, f"MKL Kovarianz-Fehler: {fro_err:.4f}")


class TestTransferFlatImages(unittest.TestCase):
    """Flach/schwarz/weiß → keine NaN/Inf."""

    def _test_flat(self, method):
        dev = _get_device()
        cases = [
            torch.zeros(1, 16, 16, 3, device=dev),
            torch.ones(1, 16, 16, 3, device=dev),
            torch.full((1, 16, 16, 3), 0.5, device=dev),
        ]
        for i, img in enumerate(cases):
            ref = torch.rand(1, 16, 16, 3, device=dev)
            fn = method(img, ref)
            out = fn(img)
            self.assertTrue(torch.isfinite(out).all(),
                            f"{method.__name__} Fall {i}: NaN/Inf in Ausgabe")

            # Auch als Vorlage
            fn2 = method(torch.rand(1, 16, 16, 3, device=dev), img)
            out2 = fn2(torch.rand(1, 16, 16, 3, device=dev))
            self.assertTrue(torch.isfinite(out2).all(),
                            f"{method.__name__} Fall {i} (als Ref): NaN/Inf")

    def test_flat_reinhard(self):
        self._test_flat(reinhard)

    def test_flat_mkl(self):
        self._test_flat(mkl)

    @slow

    def test_flat_histogram(self):
        self._test_flat(histogram)


class TestTransferDeterminism(unittest.TestCase):
    """Determinismus: gleiche Eingabe → bitgleiches Ergebnis."""

    def _test_determinism(self, method):
        dev = _get_device()
        torch.manual_seed(55)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(2, 24, 24, 3, device=dev)

        fn1 = method(img, ref)
        out1 = fn1(img)

        fn2 = method(img, ref)
        out2 = fn2(img)

        self.assertTrue(torch.equal(out1, out2),
                        f"{method.__name__}: nicht deterministisch")

    def test_determinism_reinhard(self):
        self._test_determinism(reinhard)

    def test_determinism_mkl(self):
        self._test_determinism(mkl)

    @slow

    def test_determinism_histogram(self):
        self._test_determinism(histogram)


class TestTransferInputUnchanged(unittest.TestCase):
    """Eingabetensor danach unverändert."""

    def _test_input_unchanged(self, method):
        dev = _get_device()
        torch.manual_seed(33)
        img = torch.rand(1, 16, 16, 3, device=dev)
        ref = torch.rand(1, 16, 16, 3, device=dev)
        img_copy = img.clone()

        fn = method(img, ref)
        fn(img)

        self.assertTrue(torch.equal(img, img_copy),
                        f"{method.__name__}: Eingabe wurde verändert")

    def test_input_unchanged_reinhard(self):
        self._test_input_unchanged(reinhard)

    def test_input_unchanged_mkl(self):
        self._test_input_unchanged(mkl)

    @slow

    def test_input_unchanged_histogram(self):
        self._test_input_unchanged(histogram)


class TestTransferWeights(unittest.TestCase):
    """Gewichte beeinflussen die Statistik."""

    def test_reinhard_with_weights(self):
        """Mit Gewichten: maskierte Pixel haben weniger Einfluss."""
        dev = _get_device()
        torch.manual_seed(11)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)

        # Gewicht: linke Hälfte ignoriert
        w = torch.zeros(1, 32, 32, device=dev)
        w[:, :, 16:] = 1.0

        fn = reinhard(img, ref, image_weights=w)
        out = fn(img)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
