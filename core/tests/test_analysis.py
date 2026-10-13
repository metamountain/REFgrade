"""Tests für refgrade.analysis: analyze_image, compare."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from skimage import data as skdata

from refgrade.analysis import analyze_image, compare
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from refgrade.color import rgb_to_lab
from refgrade.auto_ignore import suggest
from refgrade.reference_rank import features, rank, _load_image_tensor


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _skimage_to_tensor(name: str, device=None) -> torch.Tensor:
    """skimage-Bild → [1,H,W,3] float32 0..1."""
    if device is None:
        device = _get_device()
    img = getattr(skdata, name)()[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).to(device)


class TestAnalyzeImage(unittest.TestCase):
    """analyze_image: Struktur und Grundverhalten."""

    def test_structure(self):
        dev = _get_device()
        img = _skimage_to_tensor("coffee", dev)
        result = analyze_image(img)
        self.assertIn("tone", result)
        self.assertIn("color", result)
        self.assertIn("palette", result)

        tone = result["tone"]
        for key in ("black_point", "white_point", "median_l", "contrast",
                    "clipped_shadows", "clipped_highlights"):
            self.assertIn(key, tone)

        color = result["color"]
        self.assertIn("cast", color)
        self.assertIn("zones", color)
        self.assertIn("saturation", color)
        for zone in ("shadows", "mids", "highlights"):
            self.assertIn(zone, color["zones"])
            self.assertIn("hue", color["zones"][zone])
            self.assertIn("strength", color["zones"][zone])

        self.assertEqual(len(result["palette"]), 6)
        total_frac = sum(p["fraction"] for p in result["palette"])
        self.assertAlmostEqual(total_frac, 1.0, delta=0.01)

    def test_deterministic(self):
        """Gleiche Eingabe → gleiches Ergebnis (Seed 0)."""
        dev = _get_device()
        img = _skimage_to_tensor("astronaut", dev)
        r1 = analyze_image(img)
        r2 = analyze_image(img)
        self.assertEqual(r1, r2)

    def test_weights_change_analysis(self):
        """Ignore-Gewichte ändern die Analyse."""
        dev = _get_device()
        img = _skimage_to_tensor("coffee", dev)
        full = analyze_image(img)

        # Rechte Hälfte ignorieren
        h, w = img.shape[1], img.shape[2]
        weights = torch.ones(1, h, w, device=dev)
        weights[:, :, w // 2:] = 0.0
        partial = analyze_image(img, weights=weights)

        # Mindestens ein Wert muss sich unterscheiden
        changed = (
            full["tone"]["median_l"] != partial["tone"]["median_l"]
            or full["tone"]["contrast"] != partial["tone"]["contrast"]
            or full["color"]["mean_a"] != partial["color"]["mean_a"]
            or full["color"]["mean_b"] != partial["color"]["mean_b"]
            or full["color"]["saturation"]["chroma_median"] != partial["color"]["saturation"]["chroma_median"]
        )
        self.assertTrue(changed, "Gewichte haben die Analyse nicht verändert")

    def test_flat_image(self):
        """Flaches Bild → keine NaN."""
        dev = _get_device()
        img = torch.full((1, 16, 16, 3), 0.5, device=dev)
        result = analyze_image(img)
        # Alle Werte sollten finite sein
        tone = result["tone"]
        for v in tone.values():
            self.assertTrue(np.isfinite(v))


class TestCompare(unittest.TestCase):
    """compare: Regler-Schätzung aus bekannten Graduierungen."""

    def _run_compare_test(self, slider_values: dict, seed: int = 0,
                          crop: str | None = None):
        """Bekanntes Grade anwenden, compare schätzt die Werte zurück."""
        dev = _get_device()
        img = _skimage_to_tensor("astronaut", dev)

        # Gradieren
        graded = grade(img, slider_values)

        # Vorlage: ggf. Ausschnitt
        if crop == "tl40":
            h, w = img.shape[1], img.shape[2]
            ref = graded[:, :h * 40 // 100, :w * 40 // 100, :]
        elif crop == "br40":
            h, w = img.shape[1], img.shape[2]
            ref = graded[:, h * 60 // 100:, w * 60 // 100:, :]
        else:
            ref = graded

        src_analysis = analyze_image(img)
        ref_analysis = analyze_image(ref)
        result = compare(src_analysis, ref_analysis)
        return result["sliders"]

    def test_exposure(self):
        """Exposure ±0.1 Blende."""
        sliders = self._run_compare_test({"exposure": 0.5})
        self.assertAlmostEqual(sliders["exposure"], 0.5, delta=0.1,
                               msg=f"exposure: {sliders['exposure']}")

    def test_temperature_tint(self):
        """Temperature/Tint ±0.05."""
        sliders = self._run_compare_test({"temperature": 0.3, "tint": 0.2})
        self.assertAlmostEqual(sliders["temperature"], 0.3, delta=0.05,
                               msg=f"temperature: {sliders['temperature']}")
        self.assertAlmostEqual(sliders["tint"], 0.2, delta=0.05,
                               msg=f"tint: {sliders['tint']}")

    def test_saturation(self):
        """Saturation ±0.05."""
        sliders = self._run_compare_test({"saturation": 0.7})
        self.assertAlmostEqual(sliders["saturation"], 0.7, delta=0.05,
                               msg=f"saturation: {sliders['saturation']}")

    def test_shadow_hue(self):
        """Shadow hue ±15° bei shadow_amount ≥ 0.2."""
        sliders = self._run_compare_test(
            {"shadow_hue": 30.0, "shadow_amount": 0.3}
        )
        # Hue ist zirkulär
        est_hue = sliders["shadow_hue"]
        true_hue = 30.0
        hue_err = min(abs(est_hue - true_hue), 360 - abs(est_hue - true_hue))
        self.assertLessEqual(hue_err, 15.0,
                             msg=f"shadow_hue: {est_hue} (erwartet ~{true_hue})")

    def test_crop_tl40(self):
        """Ausschnitt oben-links 40 %: Werte protokollieren (keine Schranke)."""
        sliders = self._run_compare_test(
            {"exposure": 0.3, "temperature": 0.2}, crop="tl40"
        )
        # Nur protokollieren, keine Assertion
        print(f"  [T1b tl40] exposure={sliders['exposure']:.2f}, "
              f"temperature={sliders['temperature']:.3f}, "
              f"tint={sliders['tint']:.3f}, saturation={sliders['saturation']:.2f}")

    def test_crop_br40(self):
        """Ausschnitt unten-rechts 40 %: Werte protokollieren (keine Schranke)."""
        sliders = self._run_compare_test(
            {"exposure": 0.3, "temperature": 0.2}, crop="br40"
        )
        print(f"  [T1b br40] exposure={sliders['exposure']:.2f}, "
              f"temperature={sliders['temperature']:.3f}, "
              f"tint={sliders['tint']:.3f}, saturation={sliders['saturation']:.2f}")

    def test_sentences(self):
        """compare liefert deutsche Sätze."""
        dev = _get_device()
        img = _skimage_to_tensor("coffee", dev)
        graded = grade(img, {"temperature": 0.4, "shadow_hue": 30.0, "shadow_amount": 0.3})
        src = analyze_image(img)
        ref = analyze_image(graded)
        result = compare(src, ref)
        self.assertIsInstance(result["sentences"], list)
        self.assertGreater(len(result["sentences"]), 0)
        for s in result["sentences"]:
            self.assertIsInstance(s, str)
            self.assertGreater(len(s), 5)


class TestAutoIgnore(unittest.TestCase):
    """auto_ignore.suggest: findet dominante Fremdfläche."""

    def test_finds_tree(self):
        """T4-Aufbau: Baum links → ≥ 90 % gefunden, ≤ 5 % Fehlmarkierung."""
        dev = _get_device()
        torch.manual_seed(0)

        h, w = 64, 64
        # Basis: bläuliches Bild (begrenzte Farbpalette)
        img = torch.zeros(1, h, w, 3, device=dev)
        img[..., 0] = 0.15 + 0.05 * torch.rand(1, h, w, device=dev)
        img[..., 1] = 0.20 + 0.05 * torch.rand(1, h, w, device=dev)
        img[..., 2] = 0.60 + 0.10 * torch.rand(1, h, w, device=dev)

        # Baum: stark gesättigtes Grün in der linken Hälfte
        tree = torch.zeros(1, h, w, 3, device=dev)
        tree[..., 1] = 0.55 + 0.25 * torch.rand(1, h, w, device=dev)
        tree[..., 0] = 0.10 + 0.10 * torch.rand(1, h, w, device=dev)
        tree[..., 2] = 0.05 + 0.08 * torch.rand(1, h, w, device=dev)
        mask_tree = torch.zeros(1, h, w, device=dev)
        mask_tree[:, :, :w // 2] = 1.0
        src = img.clone()
        src[mask_tree > 0.5] = tree[mask_tree > 0.5]

        # Vorlage: ohne Baum (blau)
        ref = img

        result = suggest(src, ref, chroma_percentile=99, min_area=0.02, feather_px=2)
        self.assertEqual(result.shape, (1, h, w))

        # Bewertung
        tree_region = mask_tree > 0.5
        detected = result > 0.5

        tp = (detected & tree_region).sum().item()
        fp = (detected & ~tree_region).sum().item()
        fn_ = (~detected & tree_region).sum().item()

        total_tree = tp + fn_
        total_non_tree = fp + (~detected & ~tree_region).sum().item()

        recall = tp / max(total_tree, 1)
        false_rate = fp / max(total_non_tree, 1)

        print(f"  [auto_ignore] recall={recall:.3f}, false_rate={false_rate:.3f}")
        self.assertGreaterEqual(recall, 0.90,
                                f"recall {recall:.3f} < 0.90")
        self.assertLessEqual(false_rate, 0.05,
                             f"Fehlmarkierung {false_rate:.3f} > 0.05")


class TestReferenceRank(unittest.TestCase):
    """reference_rank: features und rank."""

    def test_features_structure(self):
        """features liefert erwartete Keys."""
        dev = _get_device()
        img = _skimage_to_tensor("coffee", dev)
        feat = features(img)
        expected_keys = [
            "tone_median_l", "tone_contrast",
            "zone_shadows_hue", "zone_shadows_strength",
            "zone_mids_hue", "zone_mids_strength",
            "zone_highlights_hue", "zone_highlights_strength",
            "chroma_median",
            "content_sky", "content_green", "content_neutral", "content_skin",
            "layout",
        ]
        for key in expected_keys:
            self.assertIn(key, feat, f"Key fehlt: {key}")
        self.assertEqual(len(feat["layout"]), 64)

    def test_features_deterministic(self):
        """Gleiche Eingabe → gleiche Features."""
        dev = _get_device()
        img = _skimage_to_tensor("astronaut", dev)
        f1 = features(img)
        f2 = features(img)
        self.assertEqual(f1, f2)

    def test_rank_returns_list(self):
        """rank liefert sortierte Liste."""
        dev = _get_device()
        img = _skimage_to_tensor("coffee", dev)
        # Nutze den Pool als Test-Ordner
        pool_dir = str(Path(__file__).resolve().parents[2] / "bench" / "pool")
        if not os.path.isdir(pool_dir) or not os.listdir(pool_dir):
            self.skipTest("Pool leer")
        results = rank(img, pool_dir, k=3)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        for path, score in results:
            self.assertIsInstance(path, str)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)
        # Sortiert absteigend
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
