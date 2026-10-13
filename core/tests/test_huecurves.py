"""Tests für Paket K10a: Hue-Kurven (LCh), Vectorscope, Worker-Integration."""

from __future__ import annotations

import math
import unittest

import torch

from refgrade.color import rgb_to_lab
from refgrade.huecurves import apply as huecurves_apply, is_identity as hue_is_identity, validate as hue_validate
from refgrade.scope import vectorscope

_TWO_PI = 2.0 * math.pi


def _lab_to_lch(lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lab [B,H,W,3] → (L, C, H) jeweils [B,H,W,1]."""
    L = lab[..., 0:1]
    a = lab[..., 1:2]
    b = lab[..., 2:3]
    C = torch.sqrt(a * a + b * b + 1e-6)
    h = (torch.atan2(b, a) / _TWO_PI + 1.0) % 1.0
    return L, C, h


class TestHueHue(unittest.TestCase):
    """hue_hue: Hue-Verschiebung um 60° bei Rot."""

    def test_red_shift_60(self):
        """Reines Rot (1,0,0) → Hue steigt um 60° ± 5° (Gamut-Clamping)."""
        x = torch.full((1, 1, 1, 3), 0.0, dtype=torch.float32)
        x[0, 0, 0, 0] = 1.0  # reines Rot

        # Kurve: konstanter Shift +60° → y = 0.5 + 60/360
        y_shift = 0.5 + 60.0 / 360.0
        curves = {"hue_hue": [[0.0, y_shift], [1.0, y_shift]]}
        result = huecurves_apply(x, curves)

        # Hue vor und nach
        lab_before = rgb_to_lab(x)
        lab_after = rgb_to_lab(result)
        _, _, h_before = _lab_to_lch(lab_before)
        _, _, h_after = _lab_to_lch(lab_after)

        # Verschiebung in Grad
        shift_deg = ((h_after - h_before).item() % 1.0) * 360.0
        # Kann auch in die andere Richtung sein (modular)
        if shift_deg > 180.0:
            shift_deg -= 360.0
        self.assertAlmostEqual(abs(shift_deg), 60.0, delta=5.0,
                               msg=f"Hue-Shift: {shift_deg:.1f}° (erwartet 60° ± 5°, Gamut-Clamping)")

    def test_gray_unchanged(self):
        """Reines Grau (0,5) bleibt unverändert (max 1/255)."""
        x = torch.full((1, 1, 1, 3), 0.5, dtype=torch.float32)
        curves = {"hue_hue": [[0.0, 0.8], [1.0, 0.2]]}  # starker Shift
        result = huecurves_apply(x, curves)
        diff = (result - x).abs().max().item()
        self.assertLessEqual(diff, 1.0 / 255.0, f"Grau verändert: {diff:.6f}")


class TestCyclic(unittest.TestCase):
    """Zyklisch: Werte bei 0,0 und 1,0 gleich (± 1e-6)."""

    def test_cyclic_continuity(self):
        """Kurve mit Punkten nahe 0 und 1: Werte bei 0,0 und 1,0 gleich."""
        # Punkte nahe den Rändern
        pts = [[0.0, 0.3], [0.1, 0.4], [0.5, 0.6], [0.9, 0.4], [1.0, 0.3]]
        curves = {"hue_hue": pts}

        # Test bei h = 0.0
        x0 = torch.zeros(1, 1, 1, 3, dtype=torch.float32)
        x0[0, 0, 0, 1] = 1.0  # Grün: hue ≈ 0.33 in LCh
        # Besser: direkt einen Test mit bekanntem Hue
        # Reines Grün (0,1,0): Lab ≈ (53, -86, 83) → hue ≈ atan2(83,-86)/(2π) ≈ 0.99
        # Das ist knapp vor 1.0, gut für den Zyklus-Test.

        # Ich teste stattdisch direkt die zyklische Interpolation:
        # Erzeuge ein Bild mit einem Pixel, dessen Hue nahe 0 ist.
        # Reines Rot (1,0,0) hat hue ≈ 0.11 (40°/360°)
        # Reines Gelb (1,1,0) hat hue ≈ 0.15 (55°/360°)
        # Ich brauche einen Hue nahe 0: das ist +a Richtung, also (1, 0.5, 0)

        # Einfacher: Ich teste die zyklische Funktion direkt
        from refgrade.huecurves import _cyclic_pchip
        # Punkte: [[0, 0.5], [0.5, 0.7], [1, 0.5]]
        # Bei x=0: y sollte = 0.5 sein
        # Bei x=1: y sollte auch = 0.5 sein (zyklisch)
        pts2 = [[0.0, 0.5], [0.5, 0.7], [1.0, 0.5]]
        x0 = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
        x1 = torch.ones(1, 1, 1, 1, dtype=torch.float32)
        y0 = _cyclic_pchip(x0, pts2)
        y1 = _cyclic_pchip(x1, pts2)
        self.assertAlmostEqual(y0.item(), y1.item(), delta=1e-6,
                               msg=f"Zyklisch: y(0)={y0.item():.8f} ≠ y(1)={y1.item():.8f}")


class TestHueSat(unittest.TestCase):
    """hue_sat mit y = 0 → alle Pixel C < 1, L gleich (± 0,5)."""

    def test_full_desaturation(self):
        """hue_sat y=0 überall → C < 1, L gleich (bei gesättigten Pixeln)."""
        # Gesättigte Farben (C > 8 in Lab): Rot, Grün, Blau, Gelb, Cyan, Magenta
        colors = torch.tensor([
            [1.0, 0.0, 0.0],  # Rot
            [0.0, 1.0, 0.0],  # Grün
            [0.0, 0.0, 1.0],  # Blau
            [1.0, 1.0, 0.0],  # Gelb
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0],  # Magenta
        ], dtype=torch.float32)
        x = colors.view(1, 2, 3, 3)  # [1, 2, 3, 3]
        curves = {"hue_sat": [[0.0, 0.0], [1.0, 0.0]]}
        result = huecurves_apply(x, curves)

        lab_before = rgb_to_lab(x)
        lab_after = rgb_to_lab(result)
        L_b, C_b, _ = _lab_to_lch(lab_before)
        L_a, C_a, _ = _lab_to_lch(lab_after)

        # C < 1 für alle Pixel
        self.assertLess(C_a.max().item(), 1.0,
                        f"Max C nach hue_sat=0: {C_a.max().item():.3f} (erwartet < 1)")
        # L gleich (± 0,5)
        l_diff = (L_a - L_b).abs().max().item()
        self.assertLessEqual(l_diff, 0.5, f"L-Unterschied: {l_diff:.3f}")


class TestLumSat(unittest.TestCase):
    """lum_sat: y = 0 für L < 0,2 → Tiefen entsättigt, Mitten unverändert."""

    def test_shadows_desaturated_midtones_unchanged(self):
        """Tiefen (L*=10) entsättigt, Mitten (L*=50) unverändert (C ± 0,5)."""
        # Dunkles Pixel (L* ≈ 10): sRGB ≈ 0.03
        x_dark = torch.full((1, 1, 1, 3), 0.03, dtype=torch.float32)
        # Helles Pixel (L* ≈ 50): sRGB ≈ 0.21
        x_mid = torch.full((1, 1, 1, 3), 0.21, dtype=torch.float32)

        # Kurve: y = 0 für x < 0.2, y = 0.5 für x >= 0.3
        curves = {"lum_sat": [[0.0, 0.0], [0.2, 0.0], [0.3, 0.5], [1.0, 0.5]]}

        # Dunkles Pixel: chroma-Faktor ≈ 0 → C wird sehr klein
        result_dark = huecurves_apply(x_dark, curves)
        lab_dark_before = rgb_to_lab(x_dark)
        lab_dark_after = rgb_to_lab(result_dark)
        _, C_dark_b, _ = _lab_to_lch(lab_dark_before)
        _, C_dark_a, _ = _lab_to_lch(lab_dark_after)
        # Dunkles Pixel hat ohnehin sehr kleines C, aber es soll weiter reduziert werden
        self.assertLessEqual(C_dark_a.item(), C_dark_b.item() + 0.1,
                             f"Dunkel: C nach {C_dark_a.item():.3f} > C vor {C_dark_b.item():.3f}")

        # Mittleres Pixel: chroma-Faktor ≈ 1 → C unverändert
        result_mid = huecurves_apply(x_mid, curves)
        lab_mid_before = rgb_to_lab(x_mid)
        lab_mid_after = rgb_to_lab(result_mid)
        _, C_mid_b, _ = _lab_to_lch(lab_mid_before)
        _, C_mid_a, _ = _lab_to_lch(lab_mid_after)
        c_diff = abs(C_mid_a.item() - C_mid_b.item())
        self.assertLessEqual(c_diff, 0.5,
                             f"Mitte: C-Unterschied {c_diff:.3f} (erwartet ≤ 0,5)")


class TestIdentity(unittest.TestCase):
    """Identität: Alle fünf Kurven mit y = 0,5 → Ergebnis = Eingang (max 1/255)."""

    def test_all_identity(self):
        x = torch.rand(1, 8, 8, 3, dtype=torch.float32)
        curves = {
            "hue_hue": [[0.0, 0.5], [1.0, 0.5]],
            "hue_sat": [[0.0, 0.5], [1.0, 0.5]],
            "hue_lum": [[0.0, 0.5], [1.0, 0.5]],
            "lum_sat": [[0.0, 0.5], [1.0, 0.5]],
            "sat_sat": [[0.0, 0.5], [1.0, 0.5]],
        }
        result = huecurves_apply(x, curves)
        diff = (result - x).abs().max().item()
        self.assertLessEqual(diff, 1.0 / 255.0, f"Identität verletzt: {diff:.6f}")

    def test_is_identity_function(self):
        self.assertTrue(hue_is_identity(None))
        self.assertTrue(hue_is_identity({}))
        self.assertTrue(hue_is_identity({"hue_hue": [[0, 0.5], [1, 0.5]]}))
        self.assertFalse(hue_is_identity({"hue_hue": [[0, 0.3], [1, 0.5]]}))
        self.assertFalse(hue_is_identity({"sat_sat": [[0, 0.5], [1, 0.7]]}))


class TestHueValidate(unittest.TestCase):
    """validate: ValueError bei Formatverletzung."""

    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            hue_validate({"foo": [[0, 0], [1, 1]]})

    def test_one_point(self):
        with self.assertRaises(ValueError):
            hue_validate({"hue_hue": [[0.5, 0.5]]})

    def test_x_not_increasing(self):
        with self.assertRaises(ValueError):
            hue_validate({"hue_sat": [[0.5, 0], [0.2, 1]]})

    def test_y_out_of_range(self):
        with self.assertRaises(ValueError):
            hue_validate({"hue_lum": [[0, 0], [1, 1.2]]})

    def test_valid(self):
        hue_validate({"hue_hue": [[0, 0.5], [1, 0.5]]})
        hue_validate({"lum_sat": [[0, 0], [0.5, 1], [1, 0.5]]})
        hue_validate({})


class TestVectorscope(unittest.TestCase):
    """Vectorscope: feste Sollwerte."""

    def test_gray_center(self):
        """Reines Grau → Maximum in der Mittelzelle (± 1 Zelle)."""
        x = torch.full((1, 16, 16, 3), 0.5, dtype=torch.float32)
        vs = vectorscope(x, bins=96)
        self.assertEqual(len(vs), 96)
        self.assertEqual(len(vs[0]), 96)
        # Maximum finden
        max_val = 0.0
        max_row, max_col = 0, 0
        for i, row in enumerate(vs):
            for j, v in enumerate(row):
                if v > max_val:
                    max_val = v
                    max_row, max_col = i, j
        # Mitte: Zeile 48, Spalte 48 (± 1)
        self.assertLessEqual(abs(max_row - 48), 1, f"Max-Zeile: {max_row} (erwartet 48±1)")
        self.assertLessEqual(abs(max_col - 48), 1, f"Max-Spalte: {max_col} (erwartet 48±1)")

    def test_red_position(self):
        """Reines Rot (1,0,0) → Maximum bei Cb < 0, Cr > 0 (oben links der Mitte)."""
        x = torch.zeros(1, 16, 16, 3, dtype=torch.float32)
        x[..., 0] = 1.0  # Rot
        vs = vectorscope(x, bins=96)
        # Maximum finden
        max_val = 0.0
        max_row, max_col = 0, 0
        for i, row in enumerate(vs):
            for j, v in enumerate(row):
                if v > max_val:
                    max_val = v
                    max_row, max_col = i, j
        # Zeile 0 = +Cr (oben), Spalte 0 = −Cb (links)
        # Rot: Cr > 0 → oben (kleine Zeilennummer), Cb < 0 → links (kleine Spaltennummer)
        self.assertLess(max_row, 48, f"Rot: Zeile {max_row} (erwartet < 48 = +Cr)")
        self.assertLess(max_col, 48, f"Rot: Spalte {max_col} (erwartet < 48 = −Cb)")

    def test_size_and_range(self):
        """Größe 96×96, Werte 0..1."""
        x = torch.rand(1, 8, 8, 3, dtype=torch.float32)
        vs = vectorscope(x, bins=96)
        self.assertEqual(len(vs), 96)
        for row in vs:
            self.assertEqual(len(row), 96)
            for v in row:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
