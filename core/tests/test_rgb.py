"""Tests für refgrade.rgb: Identität, Einzelregler, Zonen, Kurven, Histogramm."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.rgb import rgb_grade, rgb_channel_curves, rgb_histogram


def _dev() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestRgbIdentity(unittest.TestCase):
    """Alle Regler auf Default → bitgleich."""

    def test_all_defaults_identity(self):
        d = _dev()
        torch.manual_seed(42)
        img = torch.rand(1, 32, 48, 3, device=d)
        out = rgb_grade(img)
        self.assertTrue(torch.equal(out, img), "Defaults: nicht bitgleich")

    def test_flat_black(self):
        d = _dev()
        img = torch.zeros(1, 16, 16, 3, device=d)
        out = rgb_grade(img, master_gain=2.0, r_gamma=0.5)
        self.assertFalse(torch.isnan(out).any(), "Schwarz: NaN")
        self.assertFalse(torch.isinf(out).any(), "Schwarz: Inf")

    def test_flat_white(self):
        d = _dev()
        img = torch.ones(1, 16, 16, 3, device=d)
        out = rgb_grade(img, master_pedestal=-0.5, b_gain=0.0)
        self.assertFalse(torch.isnan(out).any(), "Weiß: NaN")
        self.assertFalse(torch.isinf(out).any(), "Weiß: Inf")


class TestRgbMaster(unittest.TestCase):
    """Master-Regler wirken global auf alle Kanäle."""

    def test_master_gain_brightens(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, master_gain=1.5)
        self.assertGreater(out.mean().item(), img.mean().item())

    def test_master_pedestal_shifts(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, master_pedestal=0.2)
        self.assertGreater(out.mean().item(), img.mean().item())

    def test_master_gamma_contrast(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, master_gamma=2.0)
        # Gamma > 1 → x^(1/gamma) = Wurzel → hellere mittlere Töne
        self.assertGreater(out.mean().item(), img.mean().item())


class TestRgbChannels(unittest.TestCase):
    """Kanal-Regler wirken nur auf den jeweiligen Kanal."""

    def test_r_gain_only_red(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, r_gain=2.0)
        # R-Kanal geändert, G/B unverändert
        self.assertGreater((out[..., 0] - img[..., 0]).abs().mean().item(), 0.01)
        self.assertTrue(torch.equal(out[..., 1], img[..., 1]))
        self.assertTrue(torch.equal(out[..., 2], img[..., 2]))

    def test_g_pedestal_only_green(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, g_pedestal=0.3)
        self.assertGreater((out[..., 1] - img[..., 1]).abs().mean().item(), 0.01)
        self.assertTrue(torch.equal(out[..., 0], img[..., 0]))
        self.assertTrue(torch.equal(out[..., 2], img[..., 2]))


class TestRgbZones(unittest.TestCase):
    """Zonen-Blending: Gewicht 0..1 skaliert die weiche Zonenmaske."""

    def test_shadows_only(self):
        d = _dev()
        # Bild mit klaren Schatten (dunkel) und Lichtern (hell)
        img = torch.zeros(1, 1, 4, 3, device=d)
        img[0, 0, 0] = torch.tensor([0.1, 0.1, 0.1], device=d)  # Schatten
        img[0, 0, 1] = torch.tensor([0.5, 0.5, 0.5], device=d)  # Mitte
        img[0, 0, 2] = torch.tensor([0.9, 0.9, 0.9], device=d)  # Lichter
        img[0, 0, 3] = torch.tensor([0.5, 0.5, 0.5], device=d)

        out = rgb_grade(img, r_gain=2.0, use_shadows=1.0, use_midtones=0.0, use_highlights=0.0)
        # Schatten (Pixel 0) sollte sich ändern, Lichter (Pixel 2) nicht
        shadow_diff = (out[0, 0, 0, 0] - img[0, 0, 0, 0]).abs().item()
        highlight_diff = (out[0, 0, 2, 0] - img[0, 0, 2, 0]).abs().item()
        self.assertGreater(shadow_diff, 0.01, "Schatten nicht beeinflusst")
        self.assertLess(highlight_diff, 0.01, "Lichter sollten unverändert bleiben")

    def test_no_zones_no_effect(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 16, 16, 3, device=d)
        out = rgb_grade(img, r_gain=2.0, use_shadows=0.0, use_midtones=0.0, use_highlights=0.0)
        # Alle Gewichte 0 → keine Wirkung, Output = Input
        self.assertTrue(torch.equal(out, img), "Alle Zonen 0: sollte bitgleich sein")

    def test_partial_weight_blends(self):
        d = _dev()
        # Graustufen-Verlauf: jede Luminanz-Zone klar besetzt
        img = torch.zeros(1, 1, 8, 3, device=d)
        for i in range(8):
            v = i / 7.0
            img[0, 0, i] = torch.tensor([v, v, v], device=d)
        # Volle Zonen: starke Wirkung auf den R-Kanal
        full = rgb_grade(img, r_gain=2.0, use_shadows=1.0, use_midtones=1.0, use_highlights=1.0)
        full_diff = (full[..., 0] - img[..., 0]).abs().mean().item()
        # Halbe Zonen: Wirkung liegt zwischen 0 und voll
        half = rgb_grade(img, r_gain=2.0, use_shadows=0.5, use_midtones=0.5, use_highlights=0.5)
        half_diff = (half[..., 0] - img[..., 0]).abs().mean().item()
        self.assertGreater(full_diff, 0.01, "Volle Zonen: keine Wirkung")
        self.assertGreater(half_diff, 0.0, "Halbe Zonen: gar keine Wirkung")
        self.assertLess(half_diff, full_diff, "Halbe Zonen: sollte schwächer sein als volle")


class TestRgbMasterThenChannel(unittest.TestCase):
    """K25e: Kanalschritt arbeitet auf dem Master-Ergebnis, nicht auf dem Original."""

    def test_master_gain_plus_r_gain(self):
        """x=0.5, master_gain=1.4, r_gain=1.01 → R=0.707, G=B=0.700."""
        d = _dev()
        img = torch.full((1, 4, 4, 3), 0.5, device=d)
        out = rgb_grade(img, master_gain=1.4, r_gain=1.01)
        r_val = out[0, 0, 0, 0].item()
        g_val = out[0, 0, 0, 1].item()
        b_val = out[0, 0, 0, 2].item()
        self.assertAlmostEqual(r_val, 0.707, delta=0.002,
                               msg=f"R={r_val}, expected 0.707±0.002")
        self.assertAlmostEqual(g_val, 0.700, delta=0.002,
                               msg=f"G={g_val}, expected 0.700±0.002")
        self.assertAlmostEqual(b_val, 0.700, delta=0.002,
                               msg=f"B={b_val}, expected 0.700±0.002")

    def test_master_gamma_plus_g_pedestal(self):
        """x=0.5, master_gamma=2.0, g_pedestal=0.05 → G=0.5^(1/2)+0.05=0.7571."""
        d = _dev()
        img = torch.full((1, 4, 4, 3), 0.5, device=d)
        out = rgb_grade(img, master_gamma=2.0, g_pedestal=0.05)
        g_val = out[0, 0, 0, 1].item()
        expected = 0.5 ** 0.5 + 0.05
        self.assertAlmostEqual(g_val, expected, delta=0.002,
                               msg=f"G={g_val}, expected {expected:.4f}±0.002")

    def test_zones_zero_only_master(self):
        """Zonen alle 0, Kanalregler gesetzt → nur Master wirkt."""
        d = _dev()
        img = torch.full((1, 4, 4, 3), 0.5, device=d)
        out = rgb_grade(img, master_gain=1.2, r_gain=2.0,
                        use_shadows=0.0, use_midtones=0.0, use_highlights=0.0)
        r_val = out[0, 0, 0, 0].item()
        self.assertAlmostEqual(r_val, 0.600, delta=0.002,
                               msg=f"R={r_val}, expected 0.600±0.002 (nur Master)")


class TestRgbChannelCurves(unittest.TestCase):
    """Transferkurven: Form und Werte."""

    def test_identity_curves(self):
        c = rgb_channel_curves()
        self.assertEqual(len(c["x"]), 256)
        self.assertEqual(len(c["r"]), 256)
        # Default: Kurve = Identität
        for i in range(0, 256, 32):
            self.assertAlmostEqual(c["r"][i], c["x"][i], places=4)

    def test_gain_curve(self):
        c = rgb_channel_curves(r_gain=2.0)
        # Gain 2.0: bei Input ≈0.5 → Output ≈1.0 (geclampt)
        self.assertGreater(c["r"][128], 0.9)
        # G/B bleiben Identität: Kurve folgt x-Achse
        self.assertAlmostEqual(c["g"][128], c["x"][128], places=4)


class TestRgbHistogram(unittest.TestCase):
    """Histogramm: Form und Normalisierung."""

    def test_shape(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 32, 32, 3, device=d)
        h = rgb_histogram(img, bins=32)
        self.assertEqual(len(h["r"]), 32)
        self.assertEqual(len(h["g"]), 32)
        self.assertEqual(len(h["b"]), 32)

    def test_normalized(self):
        d = _dev()
        torch.manual_seed(1)
        img = torch.rand(1, 32, 32, 3, device=d)
        h = rgb_histogram(img, bins=32)
        for ch in ("r", "g", "b"):
            self.assertLessEqual(max(h[ch]), 1.0 + 1e-6)
            self.assertGreater(max(h[ch]), 0.0)

    def test_flat_image(self):
        d = _dev()
        img = torch.full((1, 8, 8, 3), 0.5, device=d)
        h = rgb_histogram(img, bins=16)
        # Alle Pixel in einem Bin
        self.assertEqual(max(h["r"]), 1.0)
        self.assertEqual(sum(h["r"]), 1.0)


if __name__ == "__main__":
    unittest.main()
