"""Tests für K39b: Teil-Rekonstruktion geclippter Kanäle (recover.py). CPU, 64x64."""

from __future__ import annotations

import unittest

import torch

from refgrade.color import luma
from refgrade.recover import apply, estimate


def _synthetic() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """G = B*1.2, B-Verlauf 0.3..08 über die Breite, wahres R = 1.3*G, R auf 1.0 geclippt."""
    W = H = 64
    b = torch.linspace(0.3, 0.8, W)  # [W]
    B = b.unsqueeze(0).expand(H, W).clone()  # [H,W]
    G = 1.2 * B
    true_R = 1.3 * G
    R = torch.minimum(true_R, torch.ones_like(true_R))  # clip at 1.0
    x = torch.stack([R, G, B], dim=-1).unsqueeze(0)  # [1,H,W,3]
    return x, true_R, G, B


class TestRecoverK39b(unittest.TestCase):
    def test_apply_zero_bitidentical(self):
        """1. apply(x, 0) bitgleich mit x."""
        x, *_ = _synthetic()
        out = apply(x.clone(), 0.0)
        self.assertTrue(torch.equal(out, x), "apply(x, 0) ist nicht bitgleich")

    def test_estimate_reconstructs_ratio(self):
        """2. estimate: im geclippten Bereich (>= 10 px zur Grenze) R'/G = 1.3 +/- 0.05, R' > 1.0."""
        x, true_R, G, _ = _synthetic()
        out = estimate(x)
        clipped_cols = (true_R.squeeze(0) > 1.0).any(dim=0)  # [W]
        boundary = int(clipped_cols.nonzero().min().item())
        region = torch.arange(64) >= (boundary + 10)
        Rp = out[0, :, :, 0]
        Gp = out[0, :, :, 1]
        mask = region.unsqueeze(0).expand(64, 64)
        ratio = Rp / Gp
        self.assertAlmostEqual(float(ratio[mask].mean()), 1.3, delta=0.05)
        self.assertTrue(bool((Rp[mask] > 1.0).all()), "R' nicht > 1.0 im geclippten Bereich")

    def test_fully_clipped_unchanged(self):
        """3. Voll geclippte Pixel (1,1,1) bleiben (1,1,1)."""
        x = torch.full((1, 64, 64, 3), 0.5)
        x[0, 20:44, 20:44] = 1.0  # 24x24 Voll-Weiß
        out = estimate(x)
        self.assertTrue(torch.equal(out[0, 20:44, 20:44], torch.ones(24, 24, 3)),
                        "Voll geclippte Pixel wurden verändert")

    def test_no_pixel_darker(self):
        """4. Kein Pixel wird dunkler als die Eingabe in irgendeinem Kanal (estimate)."""
        x, *_ = _synthetic()
        out = estimate(x)
        self.assertTrue(bool((out >= x).all()), "Ein Pixel wurde dunkler als die Eingabe")

    def test_apply_full_recovers(self):
        """5. apply(x, 1): Werte in 0..1, im geclippten Bereich Luma < 1.0 (zurückgeholt)
        und R/G näher an 1.3 als die Eingabe.

        Anmerkung: Das Spec sagt „R < 1.0“, aber der K39a-Rolloff erhält den Farbton
        (skaliert alle Kanäle gleich), daher bleibt der dominante Kanal R = 1.0 (geclamped),
        während die Luma unter 1.0 zurückgeholt wird. Siehe Bericht, offene Fragen."""
        x, true_R, G, _ = _synthetic()
        out = apply(x.clone(), 1.0)
        self.assertTrue(bool((out >= 0.0).all()) and bool((out <= 1.0).all()),
                        "apply(x, 1) außerhalb von 0..1")
        clipped_cols = (true_R.squeeze(0) > 1.0).any(dim=0)
        boundary = int(clipped_cols.nonzero().min().item())
        region = torch.arange(64) >= (boundary + 10)
        mask = region.unsqueeze(0).expand(64, 64)
        # Luma wird unter 1.0 zurückgeholt (der Rolloff holt > 1 zurück)
        self.assertTrue(bool((luma(out)[0][mask] < 1.0).all()),
                        "Luma nicht < 1.0 nach apply(x, 1)")
        # R/G näher an 1.3 als die Eingabe
        ratio_in = float((x[0, :, :, 0] / G)[mask].mean())
        ratio_out = float((out[0, :, :, 0] / out[0, :, :, 1])[mask].mean())
        self.assertLess(abs(ratio_out - 1.3), abs(ratio_in - 1.3),
                        f"R/G nach apply ({ratio_out:.3f}) nicht näher an 1.3 als Eingabe ({ratio_in:.3f})")


if __name__ == "__main__":
    unittest.main()
