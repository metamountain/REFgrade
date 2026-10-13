"""Tests für refgrade.zonal: zone_weights, analyze, apply_params."""

import sys
import os
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.zonal import zone_weights, analyze, apply_params
from refgrade.color import rgb_to_lab, lab_to_rgb


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestZoneWeights(unittest.TestCase):
    """zone_weights: Summe = 1, korrekte Verteilung."""

    def test_sum_to_one(self):
        dev = _get_device()
        L = torch.rand(1000) * 100.0  # [0, 100]
        w_sh, w_hi, w_mid = zone_weights(L)
        total = w_sh + w_hi + w_mid
        self.assertLess((total - 1.0).abs().max().item(), 1e-5,
                        f"Zonen-Gewichte summieren nicht zu 1: max_err={(total-1).abs().max().item()}")

    def test_shadows_low_L(self):
        dev = _get_device()
        L = torch.tensor([5.0, 10.0, 15.0])
        w_sh, w_hi, w_mid = zone_weights(L)
        self.assertGreater(w_sh[0].item(), 0.8)
        self.assertLess(w_hi[0].item(), 0.1)

    def test_highlights_high_L(self):
        dev = _get_device()
        L = torch.tensor([85.0, 90.0, 95.0])
        w_sh, w_hi, w_mid = zone_weights(L)
        self.assertGreater(w_hi[0].item(), 0.8)
        self.assertLess(w_sh[0].item(), 0.1)

    def test_midrange(self):
        dev = _get_device()
        L = torch.tensor([50.0])
        w_sh, w_hi, w_mid = zone_weights(L)
        self.assertGreater(w_mid.item(), 0.5)


class TestAnalyze(unittest.TestCase):
    """analyze: JSON-fähig, korrekte Struktur."""

    def test_structure(self):
        dev = _get_device()
        torch.manual_seed(11)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        params = analyze(img, ref)

        self.assertEqual(params["version"], 1)
        self.assertIn("L_curve", params)
        self.assertEqual(len(params["L_curve"]["src"]), 17)
        self.assertEqual(len(params["L_curve"]["dst"]), 17)
        self.assertIn("zones", params)
        for zone in ("shadows", "mids", "highlights"):
            self.assertIn(zone, params["zones"])
            self.assertEqual(len(params["zones"][zone]), 2)
        self.assertIn("chroma_scale", params)
        self.assertGreaterEqual(params["chroma_scale"], 0.25)
        self.assertLessEqual(params["chroma_scale"], 4.0)

    def test_json_serializable(self):
        dev = _get_device()
        torch.manual_seed(22)
        img = torch.rand(1, 24, 24, 3, device=dev)
        ref = torch.rand(1, 24, 24, 3, device=dev)
        params = analyze(img, ref)
        # Muss serialisierbar sein
        s = json.dumps(params)
        parsed = json.loads(s)
        self.assertEqual(parsed["version"], 1)

    def test_identity_image(self):
        """Bild == Vorlage → L-Kurve ≈ Identität, Δab ≈ 0."""
        dev = _get_device()
        torch.manual_seed(33)
        img = torch.rand(2, 32, 48, 3, device=dev)
        params = analyze(img, img)
        # L-Kurve: src ≈ dst
        src = torch.tensor(params["L_curve"]["src"])
        dst = torch.tensor(params["L_curve"]["dst"])
        err = (src - dst).abs().max().item()
        self.assertLess(err, 1.0, f"L-Kurve bei Identität: Fehler {err}")
        # Δab ≈ 0
        for zone in ("shadows", "mids", "highlights"):
            da, db = params["zones"][zone]
            self.assertLess(abs(da), 0.5, f"Zone {zone} Δa: {da}")
            self.assertLess(abs(db), 0.5, f"Zone {zone} Δb: {db}")


class TestApplyParams(unittest.TestCase):
    """apply_params: Stärken, L-Quantile, a/b."""

    def test_color_strength_zero(self):
        """color_strength=0 → a/b unverändert (< 1e-3)."""
        dev = _get_device()
        torch.manual_seed(44)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        params = analyze(img, ref)
        fn = apply_params(params, light_strength=1.0, color_strength=0.0)

        lab = rgb_to_lab(img)
        lab_out = fn(lab)
        # a/b unverändert
        da = (lab_out[..., 1] - lab[..., 1]).abs().max().item()
        db = (lab_out[..., 2] - lab[..., 2]).abs().max().item()
        self.assertLess(da, 1e-3, f"a verändert: {da}")
        self.assertLess(db, 1e-3, f"b verändert: {db}")

    def test_light_strength_zero(self):
        """light_strength=0 → L unverändert."""
        dev = _get_device()
        torch.manual_seed(55)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        params = analyze(img, ref)
        fn = apply_params(params, light_strength=0.0, color_strength=1.0)

        lab = rgb_to_lab(img)
        lab_out = fn(lab)
        dL = (lab_out[..., 0] - lab[..., 0]).abs().max().item()
        self.assertLess(dL, 1e-5, f"L verändert: {dL}")

    def test_L_quantiles(self):
        """L-Quantile des Ergebnisses ±2 L-Einheiten der Vorlage (vor clamp)."""
        dev = _get_device()
        torch.manual_seed(66)
        img = torch.rand(2, 48, 64, 3, device=dev)
        ref = torch.rand(3, 40, 40, 3, device=dev)
        params = analyze(img, ref)
        fn = apply_params(params, light_strength=1.0, color_strength=1.0)

        lab_img = rgb_to_lab(img)
        lab_ref = rgb_to_lab(ref)
        lab_out = fn(lab_img)

        # 17 Quantile von L
        from refgrade.stats import sample, weighted_quantile
        q_positions = (torch.arange(17, dtype=torch.float64) + 0.5) / 17

        pix_out, _ = sample(lab_out, None)
        pix_ref, _ = sample(lab_ref, None)

        L_out_q = weighted_quantile(pix_out[:, 0].double(), torch.ones(len(pix_out), dtype=torch.float64), q_positions)
        L_ref_q = weighted_quantile(pix_ref[:, 0].double(), torch.ones(len(pix_ref), dtype=torch.float64), q_positions)

        err = (L_out_q - L_ref_q).abs().max().item()
        self.assertLess(err, 2.0, f"L-Quantil-Fehler: {err:.2f} (erwartet < 2)")


class TestZonalDeterminism(unittest.TestCase):
    """Determinismus: gleiche Eingabe → gleiches Ergebnis."""

    def test_analyze_determinism(self):
        dev = _get_device()
        torch.manual_seed(77)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        p1 = analyze(img, ref)
        p2 = analyze(img, ref)
        self.assertEqual(json.dumps(p1, sort_keys=True), json.dumps(p2, sort_keys=True))

    def test_apply_determinism(self):
        dev = _get_device()
        torch.manual_seed(88)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        params = analyze(img, ref)
        fn1 = apply_params(params)
        fn2 = apply_params(params)
        lab = rgb_to_lab(img)
        out1 = fn1(lab)
        out2 = fn2(lab)
        self.assertTrue(torch.equal(out1, out2), "apply_params nicht deterministisch")


if __name__ == "__main__":
    unittest.main()
