"""Tests für Paket K39a: Highlight-Rolloff als echte tanh-Schulter (spec/101)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.color import luma
from refgrade.grade import grade, _rolloff


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _grey(Y: float, dev) -> torch.Tensor:
    """Graues Pixel [1,1,1,3] mit Luma Y (alle Kanäle = Y)."""
    return torch.full((1, 1, 1, 3), Y, device=dev, dtype=torch.float32)


class TestRolloffIdentity(unittest.TestCase):
    """1. r=0 → bitgleich, auch für Werte > 1."""

    def test_r0_random_bitidentical(self):
        dev = _get_device()
        torch.manual_seed(0)
        img = torch.rand(2, 16, 24, 3, device=dev) * 2.0  # Werte 0..2
        out = _rolloff(img, 0.0)
        self.assertTrue(torch.equal(out, img), "r=0: Ausgabe nicht bitgleich")


class TestRolloffFixedValues(unittest.TestCase):
    """2. Graue Pixel Y = 0.3 / 1.0 / 2.0, r=1 → Sollwerte (Luma des Ergebnisses)."""

    def test_fixed_values(self):
        dev = _get_device()
        cases = [(0.3, 0.3, 1e-4), (1.0, 0.856956, 1e-4), (2.0, 0.994, 0.002)]
        for Y, expected, tol in cases:
            with self.subTest(Y=Y):
                out = _rolloff(_grey(Y, dev), 1.0)
                got = luma(out).item()
                self.assertAlmostEqual(got, expected, delta=tol,
                                       msg=f"Y={Y}: Luma {got} != {expected} (±{tol})")


class TestRolloffMonotonicity(unittest.TestCase):
    """3. Y-Rampe 0..3 (1000 Schritte), r=0.7 → Ausgabe-Luma streng steigend, max < 1.0."""

    def test_monotone_and_bounded(self):
        dev = _get_device()
        ys = torch.linspace(0.0, 3.0, 1000, device=dev, dtype=torch.float32)
        img = ys.view(1, 1, 1000, 1).expand(1, 1, 1000, 3).contiguous()  # Graupixel je Y
        out = _rolloff(img, 0.7)
        lum = luma(out).squeeze().cpu()
        diffs = lum[1:] - lum[:-1]
        # monoton steigend; im tanh-Sättigungsbereich (Y→3) liegt der Δ unter der
        # float32-Präzision, daher Toleranz statt bit-genau > 0
        self.assertGreater(diffs.min().item(), -1e-6,
                           f"Abfall (nicht monoton): min Δ={diffs.min().item():.3e}")
        self.assertGreater(lum[-1].item(), lum[0].item() + 0.5,
                           f"insgesamt nicht steigend: {lum[0].item():.3f} → {lum[-1].item():.3f}")
        self.assertLess(lum.max().item(), 1.0,
                        f"max Luma {lum.max().item()} >= 1.0")


class TestRolloffHue(unittest.TestCase):
    """4. Pixel (1.6, 0.8, 0.4), r=1 → Verhältnis R:G:B bleibt 2:1:0.5 (±1e-5)."""

    def test_hue_preserved(self):
        dev = _get_device()
        img = torch.tensor([[[[1.6, 0.8, 0.4]]]], device=dev, dtype=torch.float32)
        out = _rolloff(img, 1.0).squeeze()
        r, g, b = out[0].item(), out[1].item(), out[2].item()
        self.assertAlmostEqual(r / g, 2.0, delta=1e-5, msg=f"R/G={r / g}")
        self.assertAlmostEqual(r / b, 4.0, delta=1e-5, msg=f"R/B={r / b}")
        self.assertAlmostEqual(g / b, 2.0, delta=1e-5, msg=f"G/B={g / b}")


class TestRolloffInGrade(unittest.TestCase):
    """5. grade(exposure=2.0, highlight_rolloff=1.0) auf Grau 0.6 → Luma < 0.999;
    mit highlight_rolloff=0 → ≥ 0.999 (würde sonst hart clippen)."""

    def test_recover_vs_clip(self):
        dev = _get_device()
        img = _grey(0.6, dev)
        out_recover = grade(img, {"exposure": 2.0, "highlight_rolloff": 1.0})
        luma_recover = luma(out_recover).item()
        self.assertLess(luma_recover, 0.999,
                        f"mit Rolloff: Luma {luma_recover} >= 0.999 (nicht gerollt)")

        out_clip = grade(img, {"exposure": 2.0, "highlight_rolloff": 0.0})
        luma_clip = luma(out_clip).item()
        self.assertGreaterEqual(luma_clip, 0.999,
                                f"ohne Rolloff: Luma {luma_clip} < 0.999 (erwartet geclippt ≈1.0)")


if __name__ == "__main__":
    unittest.main()
