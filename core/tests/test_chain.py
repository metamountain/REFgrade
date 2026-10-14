"""Bereiche als Kette (Nutzer 2026-10-12): jeder Bereich arbeitet auf dem Ergebnis des vorigen.

Vorher additiv parallel — die Farbräder bestimmten ihre Zonen aus dem Original, eine mit TONE
abgedunkelte Lichterpartie bekam weiter die Lichterfarbe. Reihenfolge sections.DEFAULT_ORDER,
überschreibbar per mix["order"] (später Drag & Drop der Kapitel).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade import session as sess  # noqa: E402
from refgrade.lut import apply as lut_apply, build as lut_build  # noqa: E402
from refgrade.grade import grade  # noqa: E402
from refgrade.sections import DEFAULT_ORDER, SECTIONS, chain_fn, chain_order  # noqa: E402
from refgrade.session import Session  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402

_D = {s.name: s.default for s in SLIDERS}


def _session(img: torch.Tensor) -> Session:
    s = Session.__new__(Session)
    s.source = img
    s.source_full = img
    s.alpha = torch.zeros(1, img.shape[1], img.shape[2])
    s.ignore = torch.zeros(1, img.shape[1], img.shape[2])
    s.hists, s.previews, s.luts, s.method_fns, s.cube_cache = {}, {}, {}, {}, {}
    s.section_grid_cache = {}
    s.refs, s.pair_luts, s.pair_shifts = {}, {}, {}
    s.depth = 8
    return s


def _grey(v: float) -> torch.Tensor:
    return torch.full((1, 1, 1, 3), v)


def _mix(**kw) -> dict:
    m = {"weights": {k: 1.0 for k in SECTIONS}, "master": 1.0, "solo": None}
    m.update(kw)
    return m


class TestOrder(unittest.TestCase):
    def test_default_order(self):
        self.assertEqual(DEFAULT_ORDER, ("reference", "color", "tone", "rgb", "curves", "wheels",
                                         "hsl", "color_replace", "lut"))
        self.assertEqual(set(DEFAULT_ORDER), set(SECTIONS))
        self.assertEqual(chain_order({}), DEFAULT_ORDER)

    def test_custom_order_is_completed(self):
        self.assertEqual(chain_order({"order": ["wheels", "tone"]})[:3], ("wheels", "tone", "reference"))
        self.assertEqual(len(chain_order({"order": ["wheels", "tone"]})), len(DEFAULT_ORDER))

    def test_bad_order_raises(self):
        with self.assertRaises(KeyError):
            chain_order({"order": ["nope"]})
        with self.assertRaises(KeyError):
            chain_order({"order": ["tone", "tone"]})


class TestWheelsSeeToneBrightness(unittest.TestCase):
    """Kern der Nutzerforderung, über den echten LUT-Weg (build_lut)."""

    def _vals(self) -> dict:
        v = dict(_D)
        v["exposure"] = -1.0            # 0,85 → ~0,6 sRGB: aus den Lichtern in die Mitten
        v["wheel_blending"], v["wheel_balance"] = 0.0, 0.0
        v["gain_b"] = 1.4               # Lichter blau
        v["gamma_r"] = 1.4              # Mitten rot
        return v

    def test_darkened_highlight_gets_midtone_colour(self):
        s = _session(_grey(0.85))
        lut = sess.build_lut(s, [], "reinhard", self._vals(), _mix(weights={"tone": 1.0, "wheels": 1.0}))
        out = lut_apply(_grey(0.85), lut)[0, 0, 0]
        self.assertGreater(out[0].item(), out[2].item() + 0.02, f"Mittenfarbe erwartet: {out.tolist()}")

    def test_order_swapped_gives_highlight_colour(self):
        """Gegenprobe: Räder vor TONE → Zonen aus dem hellen Original → Lichterfarbe (blau)."""
        s = _session(_grey(0.85))
        mix = _mix(weights={"tone": 1.0, "wheels": 1.0}, order=["wheels", "tone"])
        out = lut_apply(_grey(0.85), sess.build_lut(s, [], "reinhard", self._vals(), mix))[0, 0, 0]
        self.assertGreater(out[2].item(), out[0].item() + 0.02, f"Lichterfarbe erwartet: {out.tolist()}")


class TestCacheEqualsChain(unittest.TestCase):
    def test_lut_equals_chain_fn(self):
        """Präfix-Cache-Weg == sections.chain_fn am Gitter (≤ 1e-5), auch mit Gewichten < 1,
        Master < 1, eigener Reihenfolge und Werten über 1 hinaus."""
        torch.manual_seed(0)
        img = torch.rand(1, 16, 16, 3)
        v = dict(_D)
        v.update(exposure=2.5, contrast=1.8, temperature=0.5, gain_r=1.3, lift_b=0.1, saturation=1.4)
        full = {"tone": 1.0, "color": 1.0, "wheels": 1.0, "hsl": 1.0}
        for mix in (_mix(weights=full), _mix(weights={"tone": 0.6, "color": 1.0, "wheels": 0.3}, master=0.7),
                    _mix(weights=full, order=["wheels", "color", "tone"])):
            with self.subTest(mix=mix):
                s = _session(img)
                lut = sess.build_lut(s, [], "reinhard", v, mix)
                fns = {sec: (lambda x, _v=sess.section_values(v, sec): grade(x, _v))
                       for sec in ("tone", "color", "wheels", "hsl")}
                ref = lut_build(chain_fn(fns, {**mix, "weights": {k: w for k, w in mix["weights"].items()
                                                                   if k in fns}}),
                                size=lut["size"], device=img.device)
                d = (lut["table"].cpu() - ref["table"].cpu()).abs().max().item()
                self.assertLess(d, 1e-5, f"Abweichung {d}")


class TestPrefixCache(unittest.TestCase):
    def test_late_change_reuses_early_steps(self):
        """Nur ein Rad ändern → TONE wird nicht neu gerechnet (Präfix aus dem Cache)."""
        s = _session(_grey(0.5))
        v = dict(_D)
        v["exposure"] = 0.5
        mix = _mix(weights={"tone": 1.0, "wheels": 1.0})
        sess.build_lut(s, [], "reinhard", v, mix)
        calls = {"n": 0}
        orig = sess.grade

        def counting(x, vals):
            if vals.get("exposure", 0.0) == 0.5:
                calls["n"] += 1
            return orig(x, vals)

        sess.grade = counting
        try:
            v2 = dict(v)
            v2["gain_r"] = 1.2
            sess.build_lut(s, [], "reinhard", v2, mix)
        finally:
            sess.grade = orig
        self.assertEqual(calls["n"], 0, "TONE wurde neu gerechnet")

    def test_cache_bounded(self):
        s = _session(_grey(0.5))
        for i in range(60):
            v = dict(_D)
            v["exposure"] = i * 0.05
            sess.build_lut(s, [], "reinhard", v, _mix(weights={"tone": 1.0, "wheels": 1.0}))
        self.assertLessEqual(len(s.section_grid_cache), sess._CHAIN_CACHE_MAX)


class TestChainInput(unittest.TestCase):
    """AUTO-Eingang: das Bild, wie es beim Bereich ankommt."""

    def test_wheels_input_contains_tone_color_input_does_not(self):
        img = _grey(0.4)
        s = _session(img)
        v = dict(_D)
        v["exposure"] = 1.0
        mix = _mix(weights={"tone": 1.0, "wheels": 1.0, "color": 1.0})
        at_wheels = sess.chain_input(s, [], "reinhard", v, mix, "wheels", img)
        at_color = sess.chain_input(s, [], "reinhard", v, mix, "color", img)
        expect = grade(img, sess.section_values(v, "tone")).clamp(0, 1)
        self.assertLess((at_wheels - expect).abs().max().item(), 1e-5)
        self.assertLess((at_color - img).abs().max().item(), 1e-6)

    def test_amount_half_is_half_step(self):
        img = _grey(0.4)
        s = _session(img)
        v = dict(_D)
        v["exposure"] = 1.0
        full = sess.chain_input(s, [], "reinhard", v, _mix(weights={"tone": 1.0}), "wheels", img)
        half = sess.chain_input(s, [], "reinhard", v, _mix(weights={"tone": 0.5}), "wheels", img)
        self.assertAlmostEqual((half - img).mean().item(), 0.5 * (full - img).mean().item(), places=5)


class TestAmountAndSolo(unittest.TestCase):
    def test_amount_zero_skips_section(self):
        s = _session(_grey(0.5))
        v = dict(_D)
        v["exposure"] = 1.0
        lut = sess.build_lut(s, [], "reinhard", v, _mix(weights={"tone": 0.0}))
        self.assertLess((lut_apply(_grey(0.5), lut) - 0.5).abs().max().item(), 1e-4)

    def test_solo_is_section_on_original(self):
        s = _session(_grey(0.85))
        v = TestWheelsSeeToneBrightness()._vals()
        lut = sess.build_lut(s, [], "reinhard", v, _mix(solo="wheels"))
        out = lut_apply(_grey(0.85), lut)[0, 0, 0]
        self.assertGreater(out[2].item(), out[0].item(), "SOLO Räder: Zonen aus dem Original (Lichter)")


if __name__ == "__main__":
    unittest.main()
