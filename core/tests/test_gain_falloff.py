"""Tests für die Wheel-Zonen: Gain (Kerngröße) und Falloff (Blur-Breite), 2026-10-18.

Modell: jede Zone ist ein kleiner weißer Bereich (Kern) an ihrem Zentrum — Schatten bei
Y=0, Mitten bei Y=0.5, Lichter bei Y=1 — mit weichem Falloff (Glow) darum. Gain macht
den Kern größer, Falloff den Rand weicher (1.0 = volle Reichweite 0→1). Im Standard
(0/0.5) überlappen die Zonen weich über die ganze Skala; niedrige Gain/Falloff-Werte
öffnen Lücken, in denen kein Rad wirkt.
"""

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


class TestDefaults(unittest.TestCase):
    def test_defaults(self):
        d = _defaults()
        self.assertAlmostEqual(d["wheel_gain"], 0.0)
        self.assertAlmostEqual(d["wheel_falloff"], 0.5)

    def test_zones_overlap_at_default(self):
        """Default (0/0.5): Zonen überlappen — in Y=0.25 und Y=0.75 wirken die Räder."""
        vals = _defaults()
        vals["lift_b"] = 0.2
        vals["gain_r"] = 1.5
        for y in (0.25, 0.75):
            out = grade(_make_pixel(y), vals)
            d = (out - y).abs().max().item()
            self.assertGreater(d, 1e-4, f"Y={y} im Default nicht verändert (Δ={d})")

    def test_narrow_zones_have_gaps(self):
        """Enge Zonen (0.1/0.1): grosse Lücken — in Y=0.25 und Y=0.75 wirkt kein Rad."""
        vals = _defaults()
        vals["wheel_gain"], vals["wheel_falloff"] = 0.1, 0.1
        vals["lift_b"] = 0.2
        vals["gain_r"] = 1.5
        for y in (0.25, 0.75):
            out = grade(_make_pixel(y), vals)
            d = (out - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Y={y} in der Lücke verändert um {d}")


class TestGainFalloff(unittest.TestCase):
    def test_gain_grows_zones(self):
        """gain=0.5, falloff=0: Lichterkern [0.5,1] → Y=0.6 wird getroffen; gain=0.1 nicht."""
        big = _defaults(); big["wheel_gain"] = 0.5; big["wheel_falloff"] = 0.0; big["gain_r"] = 1.5
        small = _defaults(); small["wheel_gain"] = 0.1; small["wheel_falloff"] = 0.0; small["gain_r"] = 1.5
        db = (grade(_make_pixel(0.6), big) - 0.6).abs().max().item()
        ds = (grade(_make_pixel(0.6), small) - 0.6).abs().max().item()
        self.assertGreater(db, 0.05, f"gain=0.5: Y=0.6 sollte in der Lichterzone sein (Δ={db})")
        self.assertLess(ds, 1e-4, f"gain=0.1: Y=0.6 sollte in der Lücke sein (Δ={ds})")

    def test_falloff_widens_zones(self):
        """falloff=0.3: Lichter-Falloff reicht bis Y=0.6; falloff=0.1 nur bis 0.8."""
        wide = _defaults(); wide["wheel_gain"] = 0.1; wide["wheel_falloff"] = 0.3; wide["gain_r"] = 1.5
        narrow = _defaults(); narrow["wheel_gain"] = 0.1; narrow["wheel_falloff"] = 0.1; narrow["gain_r"] = 1.5
        dw = (grade(_make_pixel(0.65), wide) - 0.65).abs().max().item()
        dn = (grade(_make_pixel(0.65), narrow) - 0.65).abs().max().item()
        self.assertGreater(dw, 0.01, f"falloff=0.3: Y=0.65 sollte erreicht werden (Δ={dw})")
        self.assertLess(dn, 1e-4, f"falloff=0.1: Y=0.65 sollte in der Lücke sein (Δ={dn})")

    def test_gain_and_falloff_close_gaps(self):
        """gain=0.3, falloff=0.3: keine Lücke mehr — Y=0.25 (Schatten) und Y=0.75 (Lichter)
        werden getroffen."""
        vals = _defaults()
        vals["wheel_gain"] = 0.3
        vals["wheel_falloff"] = 0.3
        vals["lift_b"] = 0.2
        vals["gain_r"] = 1.5
        d_lo = (grade(_make_pixel(0.25), vals) - 0.25).abs().max().item()
        d_hi = (grade(_make_pixel(0.75), vals) - 0.75).abs().max().item()
        self.assertGreater(d_lo, 0.01, f"Lücke Y=0.25 nicht geschlossen (Δ={d_lo})")
        self.assertGreater(d_hi, 0.01, f"Lücke Y=0.75 nicht geschlossen (Δ={d_hi})")

    def test_hard_edges_at_zero_falloff(self):
        """falloff=0: harte Kanten — Y=0.89 (unter Kern 0.9) unberührt, Y=0.91 getroffen."""
        vals = _defaults()
        vals["wheel_gain"] = 0.1
        vals["wheel_falloff"] = 0.0
        vals["gain_r"] = 1.5
        d_lo = (grade(_make_pixel(0.89), vals) - 0.89).abs().max().item()
        d_hi = (grade(_make_pixel(0.91), vals) - 0.91).abs().max().item()
        self.assertLess(d_lo, 1e-4, f"harte Kante: Y=0.89 verändert um {d_lo}")
        self.assertGreater(d_hi, 0.05, f"harte Kante: Y=0.91 nicht getroffen (Δ={d_hi})")

    def test_identity_at_any_gain_falloff(self):
        """Alle Räder neutral → Identität, egal wie groß die Zonen sind."""
        torch.manual_seed(123)
        img = torch.rand(1, 8, 8, 3, device=_DEVICE, dtype=torch.float32)
        vals = _defaults()
        vals["wheel_gain"] = 0.5
        vals["wheel_falloff"] = 0.5
        out = grade(img, vals)
        diff = (out - img).abs().max().item()
        self.assertLess(diff, 1e-6, f"Identität verletzt: max Δ={diff}")


class TestMidtonesUntouched(unittest.TestCase):
    """Nutzer 2026-10-12: Lichter gelb, Schatten blau — die Mitten müssen unberührt bleiben."""

    def _yellow_hi_blue_lo(self, gain: float = 0.1, falloff: float = 0.1) -> dict:
        vals = _defaults()
        vals["wheel_gain"], vals["wheel_falloff"] = gain, falloff
        vals["lift_b"] = 0.2                       # Schatten → blau
        vals["gain_r"], vals["gain_g"], vals["gain_b"] = 1.2, 1.2, 0.8   # Lichter → gelb
        return vals

    def test_midtones_unchanged_with_narrow_zones(self):
        """Enge Zonen (0.1/0.1): Lichter gelb, Schatten blau — Mitten unberührt."""
        vals = self._yellow_hi_blue_lo()
        for y in (0.42, 0.5, 0.58):
            out = grade(_make_pixel(y), vals)
            d = (out - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Mitte Y={y} verändert um {d}")

    def test_shadows_blue_highlights_yellow(self):
        vals = self._yellow_hi_blue_lo()
        dark = grade(_make_pixel(0.05), vals)[0, 0, 0]
        light = grade(_make_pixel(0.95), vals)[0, 0, 0]
        self.assertGreater(dark[2].item() - dark[0].item(), 0.1, "Schatten blau")
        self.assertGreater(light[0].item() - light[2].item(), 0.2, "Lichter gelb")

    def test_master_slider_stays_in_its_zone(self):
        """Nutzer 2026-10-12: Highlights-Regler runter → nur die Lichter dunkler,
        Mitten und Schatten unverändert (mit engen Zonen 0.1/0.1)."""
        vals = _defaults()
        vals["wheel_gain"], vals["wheel_falloff"] = 0.1, 0.1
        vals["gain_m"] = 0.7
        for y in (0.1, 0.5):
            d = (grade(_make_pixel(y), vals) - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Y={y} verändert um {d}")
        self.assertLess(grade(_make_pixel(0.95), vals)[0, 0, 0, 0].item(), 0.8)
        vals = _defaults()
        vals["wheel_gain"], vals["wheel_falloff"] = 0.1, 0.1
        vals["gamma_m"] = 0.7
        for y in (0.1, 0.9):
            d = (grade(_make_pixel(y), vals) - y).abs().max().item()
            self.assertLess(d, 1e-4, f"Midtones-Regler: Y={y} verändert um {d}")

    def test_wider_falloff_reaches_further(self):
        a = grade(_make_pixel(0.3), self._yellow_hi_blue_lo(gain=0.1, falloff=0.0))
        b = grade(_make_pixel(0.3), self._yellow_hi_blue_lo(gain=0.1, falloff=0.3))
        self.assertGreater((b - 0.3).abs().max().item(), (a - 0.3).abs().max().item())


class TestSections(unittest.TestCase):
    """FITTABLE schließt beide aus; VISIBLE +2."""

    def test_fittable_excludes(self):
        self.assertNotIn("wheel_gain", FITTABLE)
        self.assertNotIn("wheel_falloff", FITTABLE)

    def test_visible_count(self):
        # 62 (vor K28d) + 2 (texture, dehaze) + 1 (highlight_rolloff, K39a) + 1 (highlight_reconstruct, K39b)
        # + 8 (cr_{i}_sat_floor, 2026-10-18)
        self.assertEqual(len(VISIBLE), 139)
        self.assertIn("wheel_gain", VISIBLE)
        self.assertIn("wheel_falloff", VISIBLE)


class TestWindow(unittest.TestCase):
    """Gain/Falloff im Preview-Pipeline: kein NaN, Regler wirkt sichtbar."""

    def test_gain_falloff_in_preview(self):
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
        vals["wheel_gain"] = 0.3
        vals["wheel_falloff"] = 0.2
        vals["gain_r"] = 1.3  # damit der Regler sichtbar wirkt

        result, _ = preview(sess, [_REF], "reinhard", vals, 256)
        self.assertTrue(torch.isfinite(result).all(), "NaN/Inf in Preview")
        diff = (result - img).abs().max().item()
        self.assertGreater(diff, 1e-6, "Gain/Falloff haben keine Auswirkung")


if __name__ == "__main__":
    unittest.main()
