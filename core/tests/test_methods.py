"""Tests für refgrade.methods: Registry und klassische Methoden."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.methods import get, list_all, list_stable, is_stable, register
from refgrade.methods import classic  # noqa: F401 — Triggert Registrierung
from refgrade.color import rgb_to_lab
from slow import slow  # noqa: E402


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestRegistry(unittest.TestCase):
    """Registry: Registrierung und Abruf."""

    def test_classic_registered(self):
        names = list_all()
        for expected in ("reinhard", "mkl", "histogram", "zonal"):
            self.assertIn(expected, names, f"{expected} nicht registriert")

    def test_stable(self):
        stable = list_stable()
        for expected in ("reinhard", "mkl", "histogram", "zonal"):
            self.assertIn(expected, stable)
            self.assertTrue(is_stable(expected))

    def test_get_returns_callable(self):
        fn = get("reinhard")
        self.assertTrue(callable(fn))

    def test_get_unknown_raises(self):
        with self.assertRaises(KeyError):
            get("nonexistent_method")

    def test_register_new(self):
        @register("test_method", stable=False)
        def fit_test(image, reference, image_weights=None, reference_weights=None):
            return lambda x: x

        self.assertIn("test_method", list_all())
        self.assertFalse(is_stable("test_method"))
        fn = get("test_method")
        self.assertTrue(callable(fn))


class TestClassicMethods(unittest.TestCase):
    """Klassische Methoden: Form, Identität, Determinismus."""

    def _make_tensors(self, device):
        torch.manual_seed(42)
        img = torch.rand(1, 32, 32, 3, device=device)
        ref = torch.rand(1, 32, 32, 3, device=device)
        return img, ref

    @slow

    def test_all_output_shape(self):
        dev = _get_device()
        img, ref = self._make_tensors(dev)
        for name in ("reinhard", "mkl", "histogram", "zonal"):
            fn = get(name)
            out = fn(img, ref)(img)
            self.assertEqual(out.shape, img.shape, f"{name}: Form {out.shape}")
            self.assertEqual(out.dtype, torch.float32, f"{name}: dtype")

    @slow

    def test_all_identity(self):
        """Bild == Vorlage → Ergebnis ≈ Eingabe."""
        dev = _get_device()
        torch.manual_seed(77)
        img = torch.rand(1, 32, 32, 3, device=dev)
        for name in ("reinhard", "mkl", "histogram", "zonal"):
            fn = get(name)
            out = fn(img, img)(img)
            err = (out - img).abs().max().item()
            self.assertLess(err, 0.01, f"{name}: Identitätsfehler {err}")

    @slow

    def test_all_determinism(self):
        """Gleiche Eingabe → bitgleiches Ergebnis."""
        dev = _get_device()
        torch.manual_seed(55)
        img = torch.rand(1, 24, 24, 3, device=dev)
        ref = torch.rand(1, 24, 24, 3, device=dev)
        for name in ("reinhard", "mkl", "histogram", "zonal"):
            fn = get(name)
            out1 = fn(img, ref)(img)
            out2 = fn(img, ref)(img)
            self.assertTrue(torch.equal(out1, out2), f"{name}: nicht deterministisch")

    @slow

    def test_all_input_unchanged(self):
        """Eingabe wird nicht verändert."""
        dev = _get_device()
        torch.manual_seed(33)
        img = torch.rand(1, 16, 16, 3, device=dev)
        ref = torch.rand(1, 16, 16, 3, device=dev)
        img_copy = img.clone()
        for name in ("reinhard", "mkl", "histogram", "zonal"):
            fn = get(name)
            fn(img, ref)(img)
        self.assertTrue(torch.equal(img, img_copy), "Eingabe wurde verändert")

    @slow

    def test_all_flat_no_nan(self):
        """Flache Bilder → keine NaN/Inf."""
        dev = _get_device()
        for val in (0.0, 1.0, 0.5):
            img = torch.full((1, 16, 16, 3), val, device=dev)
            ref = torch.rand(1, 16, 16, 3, device=dev)
            for name in ("reinhard", "mkl", "histogram", "zonal"):
                fn = get(name)
                out = fn(img, ref)(img)
                self.assertTrue(torch.isfinite(out).all(),
                                f"{name} (val={val}): NaN/Inf")


class TestZonalPipeline(unittest.TestCase):
    """Zonal: hat lab_transform-Attribut."""

    def test_has_lab_transform(self):
        dev = _get_device()
        torch.manual_seed(11)
        img = torch.rand(1, 16, 16, 3, device=dev)
        ref = torch.rand(1, 16, 16, 3, device=dev)
        fn = get("zonal")
        pipeline = fn(img, ref)
        self.assertTrue(hasattr(pipeline, "lab_transform"))


if __name__ == "__main__":
    unittest.main()
