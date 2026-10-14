"""Tests für K11: Blending und Balance der Farbräder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refgrade.grade import grade
from refgrade.sections import FITTABLE, VISIBLE
from refgrade.sliders import SLIDERS

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _make_pixel(val: float) -> torch.Tensor:
    """Ein einzelner grauer Pixel als [1,1,1,3] Tensor."""
    return torch.full((1, 1, 1, 3), val, device=_DEVICE, dtype=torch.float32)


class TestBlendOneMatchesOld(unittest.TestCase):
    """blend=1: bytegleich zum alten LGG-Verhalten (100 zufällige Farben, max 1e-6)."""

    def test_blend_one_byte_equal(self):
        torch.manual_seed(42)
        img = torch.rand(1, 10, 10, 3, device=_DEVICE, dtype=torch.float32)

        # Alte Formel: gain_c * (ch + lift_c*(1-ch)) dann ^(1/gamma_c)
        vals = _defaults()
        vals["lift_r"] = 0.1
        vals["lift_g"] = -0.05
        vals["lift_b"] = 0.08
        vals["gamma_r"] = 1.2
        vals["gamma_g"] = 0.9
        vals["gamma_b"] = 1.1
        vals["gain_r"] = 1.15
        vals["gain_g"] = 0.95
        vals["gain_b"] = 1.1
        vals["lift_m"] = 0.02
        vals["gamma_m"] = 1.05
        vals["gain_m"] = 1.03
        vals["wheel_blending"] = 1.0
        vals["wheel_balance"] = 0.0

        out_new = grade(img, vals)

        # Alte Formel manuell
        d = _defaults()
        out_old = img.clone()
        # klassisches LGG, aber Master zuerst (2026-10-12: Helligkeit ist Vorverarbeitung,
        # die Farbe folgt auf dem verschobenen Bild)
        out_old = vals["gain_m"] * (out_old + vals["lift_m"] * (1.0 - out_old))
        out_old = out_old.clamp_min(1e-6) ** (1.0 / vals["gamma_m"])
        for i, c in enumerate(("r", "g", "b")):
            ch = out_old[..., i]
            ch = vals[f"gain_{c}"] * (ch + vals[f"lift_{c}"] * (1.0 - ch))
            ch = ch.clamp_min(1e-6) ** (1.0 / vals[f"gamma_{c}"])
            out_old[..., i] = ch

        diff = (out_new - out_old).abs().max().item()
        self.assertLess(diff, 1e-6, f"blend=1 weicht von alter Formel ab: max Δ={diff}")


class TestBlendZeroZones(unittest.TestCase):
    """blend=0: strenge Zonen."""

    def test_gain_r_highlights_only(self):
        """blend=0, balance=0, gain_r=1.5: dunkles Grau R <0.01, helles Grau R >0.1."""
        vals = _defaults()
        vals["wheel_blending"] = 0.0
        vals["wheel_balance"] = 0.0
        vals["gain_r"] = 1.5

        dark = _make_pixel(0.1)
        light = _make_pixel(0.9)

        out_dark = grade(dark, vals)
        out_light = grade(light, vals)

        dark_r_change = (out_dark[0, 0, 0, 0] - 0.1).abs().item()
        light_r_change = (out_light[0, 0, 0, 0] - 0.9).abs().item()

        self.assertLess(dark_r_change, 0.01,
                        f"dunkles Grau: R-Änderung {dark_r_change} >= 0.01")
        self.assertGreater(light_r_change, 0.1,
                           f"helles Grau: R-Änderung {light_r_change} <= 0.1")

    def test_lift_b_shadows_only(self):
        """blend=0, lift_b=0.2: dunkles Grau B >0.1, helles Grau B <0.02."""
        vals = _defaults()
        vals["wheel_blending"] = 0.0
        vals["wheel_balance"] = 0.0
        vals["lift_b"] = 0.2

        dark = _make_pixel(0.1)
        light = _make_pixel(0.9)

        out_dark = grade(dark, vals)
        out_light = grade(light, vals)

        dark_b_change = (out_dark[0, 0, 0, 2] - 0.1).abs().item()
        light_b_change = (out_light[0, 0, 0, 2] - 0.9).abs().item()

        self.assertGreater(dark_b_change, 0.1,
                           f"dunkles Grau: B-Änderung {dark_b_change} <= 0.1")
        self.assertLess(light_b_change, 0.02,
                        f"helles Grau: B-Änderung {light_b_change} >= 0.02")


class TestBalance(unittest.TestCase):
    """Balance verschiebt den Split zwischen Schatten und Highlights."""

    def test_balance_shifts_split(self):
        """blend=0, gain_r=1.5, Y=0.4: balance=+1 R-Änderung > 2× balance=-1."""
        vals_pos = _defaults()
        vals_pos["wheel_blending"] = 0.0
        vals_pos["wheel_balance"] = 1.0
        vals_pos["gain_r"] = 1.5

        vals_neg = _defaults()
        vals_neg["wheel_blending"] = 0.0
        vals_neg["wheel_balance"] = -1.0
        vals_neg["gain_r"] = 1.5

        # Y=0.4 → graues Pixel
        img = _make_pixel(0.4)

        out_pos = grade(img, vals_pos)
        out_neg = grade(img, vals_neg)

        r_change_pos = (out_pos[0, 0, 0, 0] - 0.4).abs().item()
        r_change_neg = (out_neg[0, 0, 0, 0] - 0.4).abs().item()

        self.assertGreater(r_change_pos, 2.0 * r_change_neg,
                           f"balance=+1 ({r_change_pos}) nicht > 2× balance=-1 ({r_change_neg})")


class TestMidtonesUntouched(unittest.TestCase):
    """Nutzer 2026-10-12: Lichter gelb, Schatten blau — die Mitten müssen unberührt bleiben."""

    def _yellow_hi_blue_lo(self, blend: float) -> dict:
        vals = _defaults()
        vals["wheel_blending"] = blend
        vals["wheel_balance"] = 0.0                # Zonengeometrie ohne Verschiebung prüfen
        vals["lift_b"] = 0.2                       # Schatten → blau
        vals["gain_r"], vals["gain_g"], vals["gain_b"] = 1.2, 1.2, 0.8   # Lichter → gelb
        return vals

    def test_midtones_unchanged_at_blend_zero(self):
        vals = self._yellow_hi_blue_lo(0.0)
        for y in (0.42, 0.5, 0.58):
            out = grade(_make_pixel(y), vals)
            d = (out - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Mitte Y={y} verändert um {d}")

    def test_midtone_nearly_unchanged_at_default_blend(self):
        vals = self._yellow_hi_blue_lo(0.5)
        out = grade(_make_pixel(0.5), vals)
        d = (out - 0.5).abs().max().item()
        self.assertLess(d, 0.01, f"Mitte bei Blending 0,5 verändert um {d}")

    def test_shadows_blue_highlights_yellow(self):
        vals = self._yellow_hi_blue_lo(0.0)
        dark = grade(_make_pixel(0.1), vals)[0, 0, 0]
        light = grade(_make_pixel(0.85), vals)[0, 0, 0]
        self.assertGreater(dark[2].item() - dark[0].item(), 0.1, "Schatten blau")
        self.assertGreater(light[0].item() - light[2].item(), 0.2, "Lichter gelb")

    def test_default_balance_keeps_highlights_out_of_the_mids(self):
        """Nutzer 2026-10-12: Standard-Balance −0,3, Lichter sonst zu dominant — ein Pixel bei
        Y = 0,55 bekommt mit Standard-Balance weniger Lichter-Gelb als mit Balance 0."""
        v0 = self._yellow_hi_blue_lo(0.5)
        vd = dict(v0)
        vd["wheel_balance"] = _defaults()["wheel_balance"]
        self.assertAlmostEqual(vd["wheel_balance"], -0.3)
        d0 = (grade(_make_pixel(0.55), v0) - 0.55).abs().max().item()
        dd = (grade(_make_pixel(0.55), vd) - 0.55).abs().max().item()
        self.assertLess(dd, d0)

    def test_master_slider_stays_in_its_zone(self):
        """Nutzer 2026-10-12: Highlights-Regler unter dem Rad runter → nur die Lichter dunkler,
        Mitten und Schatten unverändert (vorher global über das ganze Bild)."""
        vals = _defaults()
        vals["wheel_blending"], vals["wheel_balance"] = 0.0, 0.0
        vals["gain_m"] = 0.7
        for y in (0.1, 0.5):
            d = (grade(_make_pixel(y), vals) - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Y={y} verändert um {d}")
        self.assertLess(grade(_make_pixel(0.9), vals)[0, 0, 0, 0].item(), 0.8)
        vals = _defaults()
        vals["wheel_blending"], vals["wheel_balance"] = 0.0, 0.0
        vals["gamma_m"] = 0.7
        for y in (0.1, 0.9):
            d = (grade(_make_pixel(y), vals) - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Midtones-Regler: Y={y} verändert um {d}")

    def test_darkened_highlight_takes_the_midtone_colour(self):
        """Nutzer 2026-10-12: Helligkeit zuerst, dann Farbe — ein Licht (Y = 0,85), das der
        Highlights-Regler in die Mitten drückt, bekommt die Mittenfarbe, nicht die Lichterfarbe."""
        vals = _defaults()
        vals["wheel_blending"], vals["wheel_balance"] = 0.0, 0.0
        vals["gain_m"] = 0.6                       # 0,85 → ~0,51 (Mitte)
        vals["gain_b"] = 1.4                       # Lichter blau
        vals["gamma_r"] = 1.4                      # Mitten rot
        out = grade(_make_pixel(0.85), vals)[0, 0, 0]
        self.assertGreater(out[0].item(), out[2].item() + 0.02, f"Mittenfarbe erwartet: {out.tolist()}")

    def test_wider_blend_reaches_further(self):
        a = grade(_make_pixel(0.45), self._yellow_hi_blue_lo(0.0))
        b = grade(_make_pixel(0.45), self._yellow_hi_blue_lo(0.5))
        self.assertGreater((b - 0.45).abs().max().item(), (a - 0.45).abs().max().item())


class TestIdentity(unittest.TestCase):
    """Alle Räder neutral + blend=0, balance=0.7 → Identität."""

    def test_identity_with_extreme_blend_balance(self):
        torch.manual_seed(123)
        img = torch.rand(1, 8, 8, 3, device=_DEVICE, dtype=torch.float32)

        vals = _defaults()
        vals["wheel_blending"] = 0.0
        vals["wheel_balance"] = 0.7

        out = grade(img, vals)
        diff = (out - img).abs().max().item()
        self.assertLess(diff, 1e-6, f"Identität verletzt: max Δ={diff}")


class TestSections(unittest.TestCase):
    """FITTABLE schließt beide aus; VISIBLE +2."""

    def test_fittable_excludes(self):
        self.assertNotIn("wheel_blending", FITTABLE)
        self.assertNotIn("wheel_balance", FITTABLE)

    def test_visible_count(self):
        # 62 (vor K28d) + 2 (texture, dehaze) = 64
        self.assertEqual(len(VISIBLE), 95)
        self.assertIn("wheel_blending", VISIBLE)
        self.assertIn("wheel_balance", VISIBLE)


class TestWindow(unittest.TestCase):
    """Blending/Balance im Preview-Pipeline: kein NaN, Regler wirkt sichtbar."""

    def test_blending_in_preview(self):
        import numpy as np
        from refgrade.session import Session, preview
        _POOL = Path(__file__).resolve().parent.parent.parent / "bench" / "pool"
        _REF = str(_POOL / "01.jpg")

        torch.manual_seed(99)
        img = torch.rand(1, 16, 16, 3, device=_DEVICE, dtype=torch.float32)

        alpha = np.ones((16, 16), dtype=np.uint8) * 255
        sess = Session(
            source_full=img,
            alpha=alpha,
            depth="U8",
            source=img,
            ignore=None,
        )
        vals = _defaults()
        vals["wheel_blending"] = 0.3
        vals["wheel_balance"] = 0.1
        vals["gain_r"] = 1.3  # damit der Regler sichtbar wirkt

        result, _ = preview(sess, [_REF], "reinhard", vals, 256)
        self.assertTrue(torch.isfinite(result).all(), "NaN/Inf in Preview")
        diff = (result - img).abs().max().item()
        self.assertGreater(diff, 1e-6, "Blending/Balance haben keine Auswirkung")


if __name__ == "__main__":
    unittest.main()
