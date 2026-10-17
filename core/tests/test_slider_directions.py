"""Part A: Richtungs-Tests für alle 47 sichtbaren Regler.

Jeder Regler steht auf einem deutlichen Wert, alle anderen auf Standard.
Auswertung an festen Testfarben in Lab/HSV. Weicht ein Regler ab, ist das
ein Fehler im Code, nicht im Test.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refgrade.color import luma, rgb_to_lab
from refgrade.grade import grade
from refgrade.protect import protect
from refgrade.sections import VISIBLE
from refgrade.sliders import SLIDERS
from refgrade import zonal

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Testfarben (sRGB 0..1) ---
GREY_15 = (0.15, 0.15, 0.15)
GREY_50 = (0.5, 0.5, 0.5)
GREY_85 = (0.85, 0.85, 0.85)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
SKIN = (0.8, 0.6, 0.5)
LESS_SAT = (0.7, 0.5, 0.4)
SAT = (0.9, 0.2, 0.2)

# --- Band-Zentren (Grad) ---
BAND_CENTERS = {
    "red": 0.0, "yellow": 60.0, "green": 120.0,
    "cyan": 180.0, "blue": 240.0, "magenta": 300.0,
}
OPPOSITE = {
    "red": "cyan", "yellow": "blue", "green": "magenta",
    "cyan": "red", "blue": "yellow", "magenta": "green",
}


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _pixel(r: float, g: float, b: float) -> torch.Tensor:
    return torch.tensor([[[[r, g, b]]]], device=_DEVICE, dtype=torch.float32)


def _lab(p: torch.Tensor) -> tuple[float, float, float]:
    lab = rgb_to_lab(p)
    return float(lab[0, 0, 0, 0]), float(lab[0, 0, 0, 1]), float(lab[0, 0, 0, 2])


def _grade(r: float, g: float, b: float, vals: dict) -> torch.Tensor:
    return grade(_pixel(r, g, b), vals)


def _hue_deg(r: float, g: float, b: float) -> float:
    """HSV-Hue in Grad (0..360)."""
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h = (g - b) / d
    elif mx == g:
        h = 2.0 + (b - r) / d
    else:
        h = 4.0 + (r - g) / d
    return (h * 60.0) % 360.0


def _sat(r: float, g: float, b: float) -> float:
    """HSV-Sättigung 0..1."""
    mx, mn = max(r, g, b), min(r, g, b)
    return 0.0 if mx == 0 else (mx - mn) / mx


def _band_color(band: str, v: float = 1.0) -> tuple[float, float, float]:
    """Geseigte Farbe am Bandzentrum (HSV: h=center, s=1, v=angewählt)."""
    h = BAND_CENTERS[band] / 360.0
    s = 1.0
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)


def _opposite_color(band: str) -> tuple[float, float, float]:
    return _band_color(OPPOSITE[band])


class TestToneSection(unittest.TestCase):
    """Tone-Bereich: exposure, contrast, highlight_rolloff."""

    def test_exposure_up(self):
        """exposure +1 → L* steigt bei allen Grautönen, a*/b* ≈ 0."""
        vals = _defaults()
        vals["exposure"] = 1.0
        for c in (GREY_15, GREY_50, GREY_85):
            L0, a0, b0 = _lab(_pixel(*c))
            L1, a1, b1 = _lab(_grade(*c, vals))
            self.assertGreater(L1, L0, f"exposure=+1: L* nicht gestiegen bei {c}")
            self.assertLess(abs(a1), 1.0, f"exposure=+1: a* ≠ 0 bei {c}")
            self.assertLess(abs(b1), 1.0, f"exposure=+1: b* ≠ 0 bei {c}")

    def test_contrast_up(self):
        """contrast 1.5 → dunkles Grau dunkler, helles Grau heller."""
        vals = _defaults()
        vals["contrast"] = 1.5
        L0_dark, _, _ = _lab(_pixel(*GREY_15))
        L1_dark, _, _ = _lab(_grade(*GREY_15, vals))
        L0_light, _, _ = _lab(_pixel(*GREY_85))
        L1_light, _, _ = _lab(_grade(*GREY_85, vals))
        self.assertLess(L1_dark, L0_dark, f"contrast=1.5: dunkles Grau nicht dunkler")
        self.assertGreater(L1_light, L0_light, f"contrast=1.5: helles Grau nicht heller")

    def test_highlight_rolloff_up(self):
        """highlight_rolloff 0.5 → helles Grau sinkt, dunkles Grau unverändert."""
        vals = _defaults()
        vals["highlight_rolloff"] = 0.5
        L0_dark, _, _ = _lab(_pixel(*GREY_15))
        L1_dark, _, _ = _lab(_grade(*GREY_15, vals))
        L0_light, _, _ = _lab(_pixel(*GREY_85))
        L1_light, _, _ = _lab(_grade(*GREY_85, vals))
        self.assertLess(abs(L1_dark - L0_dark), 0.5,
                        f"rolloff=0.5: dunkles Grau verändert (ΔL={L1_dark - L0_dark:.2f})")
        self.assertLess(L1_light, L0_light, f"rolloff=0.5: helles Grau nicht gesunken")


class TestColorSection(unittest.TestCase):
    """Color-Bereich: temperature, tint, saturation, vibrance."""

    def test_temperature_up(self):
        """temperature +0.5 → b* steigt, L* unverändert (±0.5)."""
        vals = _defaults()
        vals["temperature"] = 0.5
        L0, a0, b0 = _lab(_pixel(*GREY_50))
        L1, a1, b1 = _lab(_grade(*GREY_50, vals))
        self.assertGreater(b1, b0, f"temperature=+0.5: b* nicht gestiegen")
        self.assertLess(abs(L1 - L0), 0.5,
                        f"temperature=+0.5: L* verändert (ΔL={L1 - L0:.2f})")

    def test_tint_up(self):
        """tint +0.5 → a* steigt, L* unverändert (±0.5)."""
        vals = _defaults()
        vals["tint"] = 0.5
        L0, a0, b0 = _lab(_pixel(*GREY_50))
        L1, a1, b1 = _lab(_grade(*GREY_50, vals))
        self.assertGreater(a1, a0, f"tint=+0.5: a* nicht gestiegen")
        self.assertLess(abs(L1 - L0), 0.5,
                        f"tint=+0.5: L* verändert (ΔL={L1 - L0:.2f})")

    def test_saturation_up(self):
        """saturation 1.5 → Chroma von gesättigten Farben steigt, Grau unverändert."""
        vals = _defaults()
        vals["saturation"] = 1.5
        L0, a0, b0 = _lab(_pixel(*RED))
        L1, a1, b1 = _lab(_grade(*RED, vals))
        c0 = math.sqrt(a0 * a0 + b0 * b0)
        c1 = math.sqrt(a1 * a1 + b1 * b1)
        self.assertGreater(c1, c0, f"saturation=1.5: Chroma von Rot nicht gestiegen")
        Lg0, ag0, bg0 = _lab(_pixel(*GREY_50))
        Lg1, ag1, bg1 = _lab(_grade(*GREY_50, vals))
        cg0 = math.sqrt(ag0 * ag0 + bg0 * bg0)
        cg1 = math.sqrt(ag1 * ag1 + bg1 * bg1)
        self.assertLess(abs(cg1 - cg0), 1.0,
                        f"saturation=1.5: Grau-Chroma verändert (ΔC={cg1 - cg0:.2f})")

    def test_vibrance_up(self):
        """vibrance +0.5 → Chroma steigt mehr bei wenig gesättigter Farbe."""
        vals = _defaults()
        vals["vibrance"] = 0.5
        L0, a0, b0 = _lab(_pixel(*LESS_SAT))
        L1, a1, b1 = _lab(_grade(*LESS_SAT, vals))
        c0_less = math.sqrt(a0 * a0 + b0 * b0)
        c1_less = math.sqrt(a1 * a1 + b1 * b1)
        Ls0, as0, bs0 = _lab(_pixel(*SAT))
        Ls1, as1, bs1 = _lab(_grade(*SAT, vals))
        c0_sat = math.sqrt(as0 * as0 + bs0 * bs0)
        c1_sat = math.sqrt(as1 * as1 + bs1 * bs1)
        self.assertGreater(c1_less - c0_less, c1_sat - c0_sat,
                           f"vibrance=+0.5: wenig gesättigte Farbe gewinnt nicht mehr Chroma "
                           f"(ΔC_less={c1_less - c0_less:.3f}, ΔC_sat={c1_sat - c0_sat:.3f})")


class TestWheelsSection(unittest.TestCase):
    """Wheels-Bereich: lift/gamma/gain (m + r/g/b), blending, balance."""

    def test_lift_m_up(self):
        """lift_m +0.2 → dunkles Grau steigt deutlich, helles Grau weniger."""
        vals = _defaults()
        vals["lift_m"] = 0.2
        L0_dark, _, _ = _lab(_pixel(*GREY_15))
        L1_dark, _, _ = _lab(_grade(*GREY_15, vals))
        L0_light, _, _ = _lab(_pixel(*GREY_85))
        L1_light, _, _ = _lab(_grade(*GREY_85, vals))
        d_dark = L1_dark - L0_dark
        d_light = L1_light - L0_light
        self.assertGreater(d_dark, 2.0,
                           f"lift_m=+0.2: dunkles Grau steigt nicht deutlich (ΔL={d_dark:.2f})")
        self.assertGreater(d_dark, d_light,
                           f"lift_m=+0.2: helles Grau steigt mehr als dunkles (ΔL_light={d_light:.2f})")

    def test_gain_m_up(self):
        """gain_m 1.5 → helles Grau steigt deutlich, dunkles Grau weniger."""
        vals = _defaults()
        vals["gain_m"] = 1.5
        L0_dark, _, _ = _lab(_pixel(*GREY_15))
        L1_dark, _, _ = _lab(_grade(*GREY_15, vals))
        L0_light, _, _ = _lab(_pixel(*GREY_85))
        L1_light, _, _ = _lab(_grade(*GREY_85, vals))
        d_dark = L1_dark - L0_dark
        d_light = L1_light - L0_light
        self.assertGreater(d_light, 2.0,
                           f"gain_m=1.5: helles Grau steigt nicht deutlich (ΔL={d_light:.2f})")
        self.assertGreater(d_light, d_dark,
                           f"gain_m=1.5: dunkles Grau steigt mehr als helles (ΔL_dark={d_dark:.2f})")

    def test_gamma_m_up(self):
        """gamma_m 1.5 → Grau 0.5 wird heller."""
        vals = _defaults()
        vals["gamma_m"] = 1.5
        L0, _, _ = _lab(_pixel(*GREY_50))
        L1, _, _ = _lab(_grade(*GREY_50, vals))
        self.assertGreater(L1, L0, f"gamma_m=1.5: Grau 0.5 nicht heller (ΔL={L1 - L0:.2f})")

    def test_lift_r_up(self):
        """lift_r +0.2 → R-Kanal steigt bei Grau."""
        vals = _defaults()
        vals["lift_r"] = 0.2
        out = _grade(*GREY_50, vals)
        r_out = float(out[0, 0, 0, 0])
        self.assertGreater(r_out, 0.5, f"lift_r=+0.2: R-Kanal nicht gestiegen (R={r_out:.3f})")

    def test_lift_g_up(self):
        """lift_g +0.2 → G-Kanal steigt bei Grau."""
        vals = _defaults()
        vals["lift_g"] = 0.2
        out = _grade(*GREY_50, vals)
        g_out = float(out[0, 0, 0, 1])
        self.assertGreater(g_out, 0.5, f"lift_g=+0.2: G-Kanal nicht gestiegen (G={g_out:.3f})")

    def test_lift_b_up(self):
        """lift_b +0.2 → B-Kanal steigt bei Grau."""
        vals = _defaults()
        vals["lift_b"] = 0.2
        out = _grade(*GREY_50, vals)
        b_out = float(out[0, 0, 0, 2])
        self.assertGreater(b_out, 0.5, f"lift_b=+0.2: B-Kanal nicht gestiegen (B={b_out:.3f})")

    def test_gamma_r_up(self):
        """gamma_r 1.5 → R-Kanal von Grau 0.5 steigt (LGG: Gamma > 1 = heller)."""
        vals = _defaults()
        vals["gamma_r"] = 1.5
        out = _grade(*GREY_50, vals)
        r_out = float(out[0, 0, 0, 0])
        self.assertGreater(r_out, 0.5, f"gamma_r=1.5: R-Kanal nicht gestiegen (R={r_out:.3f})")

    def test_gain_r_up(self):
        """gain_r 1.5 → R-Kanal steigt bei hellem Grau."""
        vals = _defaults()
        vals["gain_r"] = 1.5
        out = _grade(*GREY_85, vals)
        r_out = float(out[0, 0, 0, 0])
        self.assertGreater(r_out, 0.85, f"gain_r=1.5: R-Kanal nicht gestiegen (R={r_out:.3f})")

    def test_wheel_blending_zero(self):
        """wheel_blending=0, gain_r=1.5: dunkles Grau R < 0.01, helles Grau R > 0.1."""
        vals = _defaults()
        vals["wheel_blending"] = 0.0
        vals["wheel_balance"] = 0.0
        vals["gain_r"] = 1.5
        out_dark = _grade(*GREY_15, vals)
        out_light = _grade(*GREY_85, vals)
        dark_r = abs(float(out_dark[0, 0, 0, 0]) - 0.15)
        light_r = abs(float(out_light[0, 0, 0, 0]) - 0.85)
        self.assertLess(dark_r, 0.01, f"blend=0: dunkles Grau R-Änderung {dark_r} >= 0.01")
        self.assertGreater(light_r, 0.1, f"blend=0: helles Grau R-Änderung {light_r} <= 0.1")

    def test_wheel_balance_shifts(self):
        """wheel_balance +1 → mehr Wirkung in Highlights als -1."""
        vals_pos = _defaults()
        vals_pos["wheel_blending"] = 0.0
        vals_pos["wheel_balance"] = 1.0
        vals_pos["gain_r"] = 1.5
        vals_neg = _defaults()
        vals_neg["wheel_blending"] = 0.0
        vals_neg["wheel_balance"] = -1.0
        vals_neg["gain_r"] = 1.5
        out_pos = _grade(*GREY_50, vals_pos)
        out_neg = _grade(*GREY_50, vals_neg)
        r_pos = abs(float(out_pos[0, 0, 0, 0]) - 0.5)
        r_neg = abs(float(out_neg[0, 0, 0, 0]) - 0.5)
        self.assertGreater(r_pos, r_neg,
                           f"balance=+1 ({r_pos}) nicht > balance=-1 ({r_neg})")


class TestHSLSection(unittest.TestCase):
    """HSL-Bereich: sel_<band>_hue/sat/light für 6 Bänder."""

    def _test_band_hue(self, band: str):
        """sel_<band>_hue +30 → Hue der Bandfarbe verschiebt sich."""
        vals = _defaults()
        vals[f"sel_{band}_hue"] = 30.0
        r, g, b = _band_color(band)
        h0 = _hue_deg(r, g, b)
        out = _grade(r, g, b, vals)
        h1 = _hue_deg(float(out[0, 0, 0, 0]), float(out[0, 0, 0, 1]), float(out[0, 0, 0, 2]))
        dh = (h1 - h0 + 180) % 360 - 180
        self.assertGreater(abs(dh), 15.0,
                           f"sel_{band}_hue=+30: Hue-Verschiebung nur {dh:.1f}° (h0={h0:.1f}, h1={h1:.1f})")

    def _test_band_sat(self, band: str):
        """sel_<band>_sat -0.5 → Sättigung der Bandfarbe sinkt, Gegenfarbe unverändert."""
        vals = _defaults()
        vals[f"sel_{band}_sat"] = -0.5
        r, g, b = _band_color(band)
        s0 = _sat(r, g, b)
        out = _grade(r, g, b, vals)
        s1 = _sat(float(out[0, 0, 0, 0]), float(out[0, 0, 0, 1]), float(out[0, 0, 0, 2]))
        self.assertLess(s1, s0, f"sel_{band}_sat=-0.5: Sättigung nicht gesunken ({s0:.3f}→{s1:.3f})")
        ro, go, bo = _opposite_color(band)
        so0 = _sat(ro, go, bo)
        out_o = _grade(ro, go, bo, vals)
        so1 = _sat(float(out_o[0, 0, 0, 0]), float(out_o[0, 0, 0, 1]), float(out_o[0, 0, 0, 2]))
        self.assertLess(abs(so1 - so0), 0.1,
                        f"sel_{band}_sat=-0.5: Gegenfarbe {OPPOSITE[band]} Sättigung verändert (ΔS={so1 - so0:.3f})")

    def _test_band_light(self, band: str):
        """sel_<band>_light +0.2 → Helligkeit der Bandfarbe steigt, Grau unverändert."""
        vals = _defaults()
        vals[f"sel_{band}_light"] = 0.2
        r, g, b = _band_color(band, v=0.8)  # V<1 damit +0.2 nicht clamped wird
        L0 = float(luma(_pixel(r, g, b))[0, 0, 0])
        out = _grade(r, g, b, vals)
        L1 = float(luma(out)[0, 0, 0])
        self.assertGreater(L1, L0, f"sel_{band}_light=+0.2: Helligkeit nicht gestiegen ({L0:.3f}→{L1:.3f})")
        Lg0 = float(luma(_pixel(*GREY_50))[0, 0, 0])
        out_g = _grade(*GREY_50, vals)
        Lg1 = float(luma(out_g)[0, 0, 0])
        self.assertLess(abs(Lg1 - Lg0), 0.05,
                        f"sel_{band}_light=+0.2: Grau Helligkeit verändert (ΔL={Lg1 - Lg0:.3f})")

    def test_red_hue(self):
        self._test_band_hue("red")

    def test_yellow_hue(self):
        self._test_band_hue("yellow")

    def test_green_hue(self):
        self._test_band_hue("green")

    def test_cyan_hue(self):
        self._test_band_hue("cyan")

    def test_blue_hue(self):
        self._test_band_hue("blue")

    def test_magenta_hue(self):
        self._test_band_hue("magenta")

    def test_red_sat(self):
        self._test_band_sat("red")

    def test_yellow_sat(self):
        self._test_band_sat("yellow")

    def test_green_sat(self):
        self._test_band_sat("green")

    def test_cyan_sat(self):
        self._test_band_sat("cyan")

    def test_blue_sat(self):
        self._test_band_sat("blue")

    def test_magenta_sat(self):
        self._test_band_sat("magenta")

    def test_red_light(self):
        self._test_band_light("red")

    def test_yellow_light(self):
        self._test_band_light("yellow")

    def test_green_light(self):
        self._test_band_light("green")

    def test_cyan_light(self):
        self._test_band_light("cyan")

    def test_blue_light(self):
        self._test_band_light("blue")

    def test_magenta_light(self):
        self._test_band_light("magenta")


class TestReferenceSection(unittest.TestCase):
    """Reference-Bereich: light_strength, color_strength, protect_shadows/highlights."""

    def _make_ref_pair(self, source_rgb, ref_rgb):
        """Erstelle Source und Reference als [1,4,4,3] Tensoren."""
        src = torch.tensor([[list(source_rgb) for _ in range(4)] for _ in range(4)],
                           device=_DEVICE, dtype=torch.float32).unsqueeze(0)
        ref = torch.tensor([[list(ref_rgb) for _ in range(4)] for _ in range(4)],
                           device=_DEVICE, dtype=torch.float32).unsqueeze(0)
        return src, ref

    def test_light_strength_zero(self):
        """light_strength=0 → L* der Source unverändert."""
        from refgrade.color import lab_to_rgb
        src, ref = self._make_ref_pair(GREY_50, GREY_85)
        params = zonal.analyze(ref, src)
        fn = zonal.apply_params(params, light_strength=0.0, color_strength=0.0)
        lab_src = rgb_to_lab(src)
        lab_out = fn(lab_src)
        L0 = float(lab_src[0, 0, 0, 0])
        L1 = float(lab_out[0, 0, 0, 0])
        self.assertLess(abs(L1 - L0), 0.5,
                        f"light_strength=0: L* verändert (ΔL={L1 - L0:.2f})")

    def test_color_strength_zero(self):
        """color_strength=0 → a*/b* der Source unverändert."""
        from refgrade.color import lab_to_rgb
        src, ref = self._make_ref_pair(GREY_50, (0.9, 0.5, 0.3))
        params = zonal.analyze(ref, src)
        fn = zonal.apply_params(params, light_strength=0.0, color_strength=0.0)
        lab_src = rgb_to_lab(src)
        lab_out = fn(lab_src)
        da = float(lab_out[0, 0, 0, 1] - lab_src[0, 0, 0, 1])
        db = float(lab_out[0, 0, 0, 2] - lab_src[0, 0, 0, 2])
        self.assertLess(abs(da), 1.0, f"color_strength=0: a* verändert (Δa={da:.2f})")
        self.assertLess(abs(db), 1.0, f"color_strength=0: b* verändert (Δb={db:.2f})")

    def test_protect_shadows(self):
        """protect_shadows=1.0 → dunkle Pixel bleiben unverändert."""
        dark_src = torch.full((1, 4, 4, 3), 0.1, device=_DEVICE, dtype=torch.float32)
        dark_matched = torch.full((1, 4, 4, 3), 0.3, device=_DEVICE, dtype=torch.float32)
        vals = _defaults()
        vals["protect_shadows"] = 1.0
        vals["protect_highlights"] = 0.0
        out = protect(dark_src, dark_matched, vals)
        diff = (out - dark_src).abs().max().item()
        self.assertLess(diff, 0.01,
                        f"protect_shadows=1.0: dunkles Pixel verändert (Δ={diff:.4f})")

    def test_protect_highlights(self):
        """protect_highlights=1.0 → helle Pixel bleiben unverändert."""
        light_src = torch.full((1, 4, 4, 3), 0.9, device=_DEVICE, dtype=torch.float32)
        light_matched = torch.full((1, 4, 4, 3), 0.6, device=_DEVICE, dtype=torch.float32)
        vals = _defaults()
        vals["protect_shadows"] = 0.0
        vals["protect_highlights"] = 1.0
        out = protect(light_src, light_matched, vals)
        diff = (out - light_src).abs().max().item()
        self.assertLess(diff, 0.01,
                        f"protect_highlights=1.0: helles Pixel verändert (Δ={diff:.4f})")


class TestToneZones(unittest.TestCase):
    """Lightroom-Zonen: highlights, shadows, whites, blacks. Graukeil L* 0..100, je Regler +1."""

    def _grey_at_L(self, L: float) -> tuple[float, float, float]:
        """sRGB-Grauwert für gegebenes L* (0..100)."""
        # L* → Y → sRGB
        Y = ((L + 16) / 116) ** 3 if L > 8 else L / 903.3
        # sRGB linear → gamma
        if Y <= 0.0031308:
            c = 12.92 * Y
        else:
            c = 1.055 * Y ** (1 / 2.4) - 0.055
        c = max(0.0, min(1.0, c))
        return (c, c, c)

    def _L_at(self, L_target: float, vals: dict) -> float:
        """L* des Grades an einem Graupunkt mit L*=L_target."""
        r, g, b = self._grey_at_L(L_target)
        _, a, b_ = _lab(_pixel(r, g, b))
        L0 = float(rgb_to_lab(_pixel(r, g, b))[0, 0, 0, 0])
        L1 = float(rgb_to_lab(_grade(r, g, b, vals))[0, 0, 0, 0])
        return L1 - L0

    def test_shadows_plus1(self):
        """shadows +1: L*-Anstieg bei L*25 > 6, bei L*75 < 2; L*0 und L*100 < 0,5."""
        vals = _defaults()
        vals["shadows"] = 1.0
        d25 = self._L_at(25, vals)
        d75 = self._L_at(75, vals)
        d0 = self._L_at(0, vals)
        d100 = self._L_at(100, vals)
        self.assertGreater(d25, 6.0, f"shadows=+1: ΔL*(25)={d25:.2f} <= 6")
        self.assertLess(abs(d75), 2.0, f"shadows=+1: |ΔL*(75)|={abs(d75):.2f} >= 2")
        self.assertLess(abs(d0), 0.5, f"shadows=+1: |ΔL*(0)|={abs(d0):.2f} >= 0.5")
        self.assertLess(abs(d100), 0.5, f"shadows=+1: |ΔL*(100)|={abs(d100):.2f} >= 0.5")

    def test_highlights_plus1(self):
        """highlights +1: spiegelbildlich zu shadows."""
        vals = _defaults()
        vals["highlights"] = 1.0
        d75 = self._L_at(75, vals)
        d25 = self._L_at(25, vals)
        d0 = self._L_at(0, vals)
        d100 = self._L_at(100, vals)
        self.assertGreater(d75, 6.0, f"highlights=+1: ΔL*(75)={d75:.2f} <= 6")
        self.assertLess(abs(d25), 2.0, f"highlights=+1: |ΔL*(25)|={abs(d25):.2f} >= 2")
        self.assertLess(abs(d0), 0.5, f"highlights=+1: |ΔL*(0)|={abs(d0):.2f} >= 0.5")
        self.assertLess(abs(d100), 0.5, f"highlights=+1: |ΔL*(100)|={abs(d100):.2f} >= 0.5")

    def test_blacks_plus1(self):
        """blacks +1: L*5 steigt um > 5, L*95 um < 1."""
        vals = _defaults()
        vals["blacks"] = 1.0
        d5 = self._L_at(5, vals)
        d95 = self._L_at(95, vals)
        self.assertGreater(d5, 5.0, f"blacks=+1: ΔL*(5)={d5:.2f} nicht > 5")
        self.assertLess(abs(d95), 1.0, f"blacks=+1: |ΔL*(95)|={abs(d95):.2f} >= 1")

    def test_whites_plus1(self):
        """whites +1: spiegelbildlich zu blacks."""
        vals = _defaults()
        vals["whites"] = 1.0
        d95 = self._L_at(95, vals)
        d5 = self._L_at(5, vals)
        # +5 is the physical ceiling (L*95 -> 100); float32 Lab round trip leaves ~1e-4 of slack.
        self.assertGreaterEqual(d95, 5.0 - 1e-3, f"whites=+1: ΔL*(95)={d95:.2f} < 5")
        self.assertLess(abs(d5), 1.0, f"whites=+1: |ΔL*(5)|={abs(d5):.2f} >= 1")

    def test_shadows_minus1(self):
        """shadows −1: Gegenrichtung zu +1."""
        vals = _defaults()
        vals["shadows"] = -1.0
        d25 = self._L_at(25, vals)
        self.assertLess(d25, -6.0, f"shadows=-1: ΔL*(25)={d25:.2f} nicht < -6")

    def test_highlights_minus1(self):
        """highlights −1: Gegenrichtung zu +1."""
        vals = _defaults()
        vals["highlights"] = -1.0
        d75 = self._L_at(75, vals)
        self.assertLess(d75, -6.0, f"highlights=-1: ΔL*(75)={d75:.2f} nicht < -6")

    def test_blacks_minus1(self):
        """blacks −1: Gegenrichtung zu +1 (L*5 sinkt)."""
        vals = _defaults()
        vals["blacks"] = -1.0
        d5 = self._L_at(5, vals)
        # -5 is the physical floor (L*5 -> 0); float32 Lab round trip leaves ~1e-4 of slack.
        self.assertLessEqual(d5, -5.0 + 1e-3, f"blacks=-1: ΔL*(5)={d5:.2f} nicht <= -5")

    def test_whites_minus1(self):
        """whites −1: Gegenrichtung zu +1 (L*95 sinkt)."""
        vals = _defaults()
        vals["whites"] = -1.0
        d95 = self._L_at(95, vals)
        self.assertLess(d95, -5.0, f"whites=-1: ΔL*(95)={d95:.2f} nicht < -5")

    def test_identity_at_zero(self):
        """Alle Zonen auf 0 → Identität (bitgleich)."""
        vals = _defaults()
        out = _grade(*GREY_50, vals)
        self.assertEqual(float(out[0, 0, 0, 0]), 0.5)
        self.assertEqual(float(out[0, 0, 0, 1]), 0.5)
        self.assertEqual(float(out[0, 0, 0, 2]), 0.5)


class TestRGBSection(unittest.TestCase):
    """Richtungs-Tests für die 15 RGB-Regler (rgb_m_*, rgb_r_*, rgb_g_*, rgb_b_*, rgb_use_*)."""

    def _rgb(self, r: float, g: float, b: float, **kw: float) -> tuple[float, float, float]:
        from refgrade.rgb import rgb_grade
        x = _pixel(r, g, b)
        out = rgb_grade(x, **kw)
        return (float(out[0, 0, 0, 0]), float(out[0, 0, 0, 1]), float(out[0, 0, 0, 2]))

    # ── Master ──

    def test_rgb_m_gamma_up(self):
        """rgb_m_gamma=2 → alle Kanäle heller (ch^(1/2) > ch für ch<1)."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, master_gamma=2.0)
        self.assertGreater(r, 0.5, f"rgb_m_gamma=2: R={r:.4f} nicht > 0.5")
        self.assertGreater(g, 0.5, f"rgb_m_gamma=2: G={g:.4f} nicht > 0.5")
        self.assertGreater(b, 0.5, f"rgb_m_gamma=2: B={b:.4f} nicht > 0.5")

    def test_rgb_m_pedestal_up(self):
        """rgb_m_pedestal=+0.2 → alle Kanäle heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, master_pedestal=0.2)
        self.assertGreater(r, 0.5, f"rgb_m_pedestal=0.2: R={r:.4f} nicht > 0.5")
        self.assertGreater(g, 0.5, f"rgb_m_pedestal=0.2: G={g:.4f} nicht > 0.5")
        self.assertGreater(b, 0.5, f"rgb_m_pedestal=0.2: B={b:.4f} nicht > 0.5")

    def test_rgb_m_gain_up(self):
        """rgb_m_gain=2 → alle Kanäle heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, master_gain=2.0)
        self.assertGreater(r, 0.5, f"rgb_m_gain=2: R={r:.4f} nicht > 0.5")
        self.assertGreater(g, 0.5, f"rgb_m_gain=2: G={g:.4f} nicht > 0.5")
        self.assertGreater(b, 0.5, f"rgb_m_gain=2: B={b:.4f} nicht > 0.5")

    # ── R-Kanal ──

    def test_rgb_r_gamma_up(self):
        """rgb_r_gamma=2 → nur R heller (ch^(1/2) > ch für ch<1)."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, r_gamma=2.0)
        self.assertGreater(r, 0.5, f"rgb_r_gamma=2: R={r:.4f} nicht > 0.5")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_rgb_r_pedestal_up(self):
        """rgb_r_pedestal=+0.2 → nur R heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, r_pedestal=0.2)
        self.assertGreater(r, 0.5, f"rgb_r_pedestal=0.2: R={r:.4f} nicht > 0.5")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_rgb_r_gain_up(self):
        """rgb_r_gain=2 → nur R heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, r_gain=2.0)
        self.assertGreater(r, 0.5, f"rgb_r_gain=2: R={r:.4f} nicht > 0.5")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    # ── G-Kanal ──

    def test_rgb_g_gamma_up(self):
        """rgb_g_gamma=2 → nur G heller (ch^(1/2) > ch für ch<1)."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, g_gamma=2.0)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertGreater(g, 0.5, f"rgb_g_gamma=2: G={g:.4f} nicht > 0.5")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_rgb_g_pedestal_up(self):
        """rgb_g_pedestal=+0.2 → nur G heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, g_pedestal=0.2)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertGreater(g, 0.5, f"rgb_g_pedestal=0.2: G={g:.4f} nicht > 0.5")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_rgb_g_gain_up(self):
        """rgb_g_gain=2 → nur G heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, g_gain=2.0)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertGreater(g, 0.5, f"rgb_g_gain=2: G={g:.4f} nicht > 0.5")
        self.assertAlmostEqual(b, 0.5, places=4, msg=f"B={b:.4f} soll unverändert bleiben")

    # ── B-Kanal ──

    def test_rgb_b_gamma_up(self):
        """rgb_b_gamma=2 → nur B heller (ch^(1/2) > ch für ch<1)."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, b_gamma=2.0)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertGreater(b, 0.5, f"rgb_b_gamma=2: B={b:.4f} nicht > 0.5")

    def test_rgb_b_pedestal_up(self):
        """rgb_b_pedestal=+0.2 → nur B heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, b_pedestal=0.2)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertGreater(b, 0.5, f"rgb_b_pedestal=0.2: B={b:.4f} nicht > 0.5")

    def test_rgb_b_gain_up(self):
        """rgb_b_gain=2 → nur B heller."""
        r, g, b = self._rgb(0.5, 0.5, 0.5, b_gain=2.0)
        self.assertAlmostEqual(r, 0.5, places=4, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.5, places=4, msg=f"G={g:.4f} soll unverändert bleiben")
        self.assertGreater(b, 0.5, f"rgb_b_gain=2: B={b:.4f} nicht > 0.5")

    # ── Zonen ──

    def test_rgb_use_shadows_zero(self):
        """rgb_use_shadows=0 → kaum Effekt in tiefen Schatten."""
        from refgrade.rgb import rgb_grade
        x = _pixel(0.01, 0.01, 0.01)
        out = rgb_grade(x, r_gain=2.0, use_shadows=0.0, use_midtones=1.0, use_highlights=1.0)
        r = float(out[0, 0, 0, 0])
        self.assertAlmostEqual(r, 0.01, places=4,
                               msg=f"rgb_use_shadows=0: R={r:.6f} soll in Schatten ≈ 0.01 bleiben")

    def test_rgb_use_midtones_zero(self):
        """rgb_use_midtones=0 → kein Effekt in reinen Mitteltönen (0.5 ist pure Midtone)."""
        from refgrade.rgb import rgb_grade
        x = _pixel(0.5, 0.5, 0.5)
        out = rgb_grade(x, r_gain=2.0, use_shadows=1.0, use_midtones=0.0, use_highlights=1.0)
        r = float(out[0, 0, 0, 0])
        self.assertAlmostEqual(r, 0.5, places=4,
                               msg=f"rgb_use_midtones=0: R={r:.6f} soll in Mitteltönen ≈ 0.5 bleiben")

    def test_rgb_use_highlights_zero(self):
        """rgb_use_highlights=0 → kaum Effekt in Highlights (kleine Midtone-Überlappung ok)."""
        from refgrade.rgb import rgb_grade
        x = _pixel(0.99, 0.99, 0.99)
        out = rgb_grade(x, r_gain=2.0, use_shadows=1.0, use_midtones=1.0, use_highlights=0.0)
        r = float(out[0, 0, 0, 0])
        self.assertLess(r - 0.99, 0.005,
                        msg=f"rgb_use_highlights=0: R={r:.6f} weicht > 0.005 von 0.99 ab")


class TestClarity(unittest.TestCase):
    """Richtungs-Tests für clarity (lokaler Kontrast, braucht >1×1 Bild)."""

    @staticmethod
    def _test_img() -> torch.Tensor:
        """32×32: heller Fleck (0.8) in dunklem Feld (0.3) → lokaler Kontrast."""
        x = torch.full((1, 32, 32, 3), 0.3, dtype=torch.float32)
        x[:, 12:20, 12:20, :] = 0.8
        return x

    def test_clarity_positive(self):
        """clarity=+0.5 → lokale Kontraststeigerung (Std ↑)."""
        from refgrade.clarity import apply
        x = self._test_img()
        out = apply(x, 0.5, full_size=32)
        self.assertGreater(float(out.std()), float(x.std()),
                           "clarity=0.5: Std soll steigen (lokaler Kontrast ↑)")

    def test_clarity_negative(self):
        """clarity=-0.5 → lokale Kontrastverringerung (Std ↓)."""
        from refgrade.clarity import apply
        x = self._test_img()
        out = apply(x, -0.5, full_size=32)
        self.assertLess(float(out.std()), float(x.std()),
                        "clarity=-0.5: Std soll fallen (lokaler Kontrast ↓)")

    def test_clarity_zero_identity(self):
        """clarity=0 → Identität."""
        from refgrade.clarity import apply
        x = self._test_img()
        out = apply(x, 0.0, full_size=32)
        self.assertTrue(torch.equal(out, x), "clarity=0 muss bitgleich Identität sein")


class TestTexture(unittest.TestCase):
    """Richtungs-Tests für texture (feiner Detailkontrast, braucht >1×1 Bild)."""

    @staticmethod
    def _test_img() -> torch.Tensor:
        """64×64: gleichmäßiger Grund + feines Rauschen."""
        base = 0.5
        noise = torch.randn(1, 64, 64, 3, dtype=torch.float32) * 0.05
        return (base + noise).clamp(0, 1)

    def test_texture_positive(self):
        """texture=+0.5 → Detailkontrast steigt (Std ↑)."""
        from refgrade.texture import apply
        x = self._test_img()
        out = apply(x, 0.5, full_size=64)
        self.assertGreater(float(out.std()), float(x.std()),
                           "texture=0.5: Std soll steigen (Detailkontrast ↑)")

    def test_texture_negative(self):
        """texture=-0.5 → Detailkontrast sinkt (Std ↓)."""
        from refgrade.texture import apply
        x = self._test_img()
        out = apply(x, -0.5, full_size=64)
        self.assertLess(float(out.std()), float(x.std()),
                        "texture=-0.5: Std soll fallen (Detailkontrast ↓)")

    def test_texture_zero_identity(self):
        """texture=0 → Identität."""
        from refgrade.texture import apply
        x = self._test_img()
        out = apply(x, 0.0, full_size=64)
        self.assertTrue(torch.equal(out, x), "texture=0 muss bitgleich Identität sein")


class TestDehaze(unittest.TestCase):
    """Richtungs-Tests für dehaze (Dark Channel, braucht >1×1 Bild)."""

    @staticmethod
    def _clear_img() -> torch.Tensor:
        """64×64: helle Region (0.9) + dunkle Region (0.2)."""
        J = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        J[:, :32, :, :] = 0.9
        J[:, 32:, :, :] = 0.2
        return J

    @staticmethod
    def _hazed_img() -> torch.Tensor:
        """64×64: Dunst auf klarem Bild (t=0.6, A=0.9)."""
        J = TestDehaze._clear_img()
        t, A = 0.6, 0.9
        return J * t + A * (1 - t)

    def test_dehaze_positive(self):
        """dehaze=+0.5 → Dunst wird entfernt (Abstand zum klaren Bild sinkt)."""
        from refgrade.dehaze import apply
        x = self._hazed_img()
        J = self._clear_img()
        out = apply(x, 0.5, full_size=64)
        self.assertLess((out - J).abs().mean().item(), (x - J).abs().mean().item(),
                        "dehaze=0.5: Abstand zum klaren Bild soll sinken")

    def test_dehaze_negative(self):
        """dehaze=-0.5 → Kontrast sinkt (Dunst hinzugefügt)."""
        from refgrade.dehaze import apply
        x = self._hazed_img()
        out = apply(x, -0.5, full_size=64)
        self.assertLess(float(out.std()), float(x.std()),
                        "dehaze=-0.5: Std soll fallen (Kontrast ↓)")

    def test_dehaze_zero_identity(self):
        """dehaze=0 → Identität."""
        from refgrade.dehaze import apply
        x = self._hazed_img()
        out = apply(x, 0.0, full_size=64)
        self.assertTrue(torch.equal(out, x), "dehaze=0 muss bitgleich Identität sein")


class TestWheelsMissing(unittest.TestCase):
    """Richtungs-Tests für fehlende Wheel-Regler: gain_g, gain_b, gamma_g, gamma_b, lift_g, lift_b."""

    def _rgb(self, r: float, g: float, b: float, vals: dict) -> tuple[float, float, float]:
        out = _grade(r, g, b, vals)
        return (float(out[0, 0, 0, 0]), float(out[0, 0, 0, 1]), float(out[0, 0, 0, 2]))

    def test_gain_g_up(self):
        """gain_g=2 → nur G heller (Highlights)."""
        vals = _defaults()
        vals["gain_g"] = 2.0
        r, g, b = self._rgb(0.8, 0.8, 0.8, vals)
        self.assertGreater(g, 0.8, f"gain_g=2: G={g:.4f} nicht > 0.8")
        self.assertAlmostEqual(r, 0.8, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.8, places=3, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_gain_b_up(self):
        """gain_b=2 → nur B heller (Highlights)."""
        vals = _defaults()
        vals["gain_b"] = 2.0
        r, g, b = self._rgb(0.8, 0.8, 0.8, vals)
        self.assertGreater(b, 0.8, f"gain_b=2: B={b:.4f} nicht > 0.8")
        self.assertAlmostEqual(r, 0.8, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.8, places=3, msg=f"G={g:.4f} soll unverändert bleiben")

    def test_gamma_g_up(self):
        """gamma_g=2 → nur G heller (Midtone, ch^(1/(1+γ-1)) = ch^0.5)."""
        vals = _defaults()
        vals["gamma_g"] = 2.0
        r, g, b = self._rgb(0.5, 0.5, 0.5, vals)
        self.assertGreater(g, 0.5, f"gamma_g=2: G={g:.4f} nicht > 0.5")
        self.assertAlmostEqual(r, 0.5, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.5, places=3, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_gamma_b_up(self):
        """gamma_b=2 → nur B heller (Midtone)."""
        vals = _defaults()
        vals["gamma_b"] = 2.0
        r, g, b = self._rgb(0.5, 0.5, 0.5, vals)
        self.assertGreater(b, 0.5, f"gamma_b=2: B={b:.4f} nicht > 0.5")
        self.assertAlmostEqual(r, 0.5, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.5, places=3, msg=f"G={g:.4f} soll unverändert bleiben")

    def test_lift_g_up(self):
        """lift_g=+0.2 → nur G in Schatten heller."""
        vals = _defaults()
        vals["lift_g"] = 0.2
        r, g, b = self._rgb(0.15, 0.15, 0.15, vals)
        self.assertGreater(g, 0.15, f"lift_g=0.2: G={g:.4f} nicht > 0.15")
        self.assertAlmostEqual(r, 0.15, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(b, 0.15, places=3, msg=f"B={b:.4f} soll unverändert bleiben")

    def test_lift_b_up(self):
        """lift_b=+0.2 → nur B in Schatten heller."""
        vals = _defaults()
        vals["lift_b"] = 0.2
        r, g, b = self._rgb(0.15, 0.15, 0.15, vals)
        self.assertGreater(b, 0.15, f"lift_b=0.2: B={b:.4f} nicht > 0.15")
        self.assertAlmostEqual(r, 0.15, places=3, msg=f"R={r:.4f} soll unverändert bleiben")
        self.assertAlmostEqual(g, 0.15, places=3, msg=f"G={g:.4f} soll unverändert bleiben")


class TestNeutralIdentity(unittest.TestCase):
    """Alle VISIBLE-Regler auf Neutral → Bild unverändert (max. Abweichung ≤ 1/255)."""

    def test_all_visible_neutral(self):
        """Jeder VISIBLE-Regler einzeln auf Default → Identität."""
        for key in VISIBLE:
            with self.subTest(key=key):
                vals = _defaults()
                slider = next((s for s in SLIDERS if s.name == key), None)
                if slider is None:
                    continue  # Wheel-Key ohne SLIDERS-Eintrag
                vals[key] = slider.default
                out = _grade(0.5, 0.5, 0.5, vals)
                r = float(out[0, 0, 0, 0])
                g = float(out[0, 0, 0, 1])
                b = float(out[0, 0, 0, 2])
                self.assertAlmostEqual(r, 0.5, delta=1.0 / 255.0,
                                       msg=f"{key}={slider.default}: R={r:.6f} ≠ 0.5")
                self.assertAlmostEqual(g, 0.5, delta=1.0 / 255.0,
                                       msg=f"{key}={slider.default}: G={g:.6f} ≠ 0.5")
                self.assertAlmostEqual(b, 0.5, delta=1.0 / 255.0,
                                       msg=f"{key}={slider.default}: B={b:.6f} ≠ 0.5")


class TestVisibleCount(unittest.TestCase):
    """Sicherstellen, dass alle 64 sichtbaren Regler in VISIBLE sind."""

    def test_visible_count(self):
        self.assertEqual(len(VISIBLE), 131, f"VISIBLE hat {len(VISIBLE)} statt 131 Regler")


if __name__ == "__main__":
    unittest.main()
