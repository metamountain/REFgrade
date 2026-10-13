"""Tests für Paket 10: Presets gültig, classify_luts korrekt."""

import sys
import os
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.presets import PRESETS
from refgrade.sliders import SLIDERS
from refgrade.grade import grade
from refgrade.lut import build, apply as lut_apply


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _slider_map() -> dict[str, tuple[float, float]]:
    """name → (min, max)."""
    return {s.name: (s.min, s.max) for s in SLIDERS}


def _default_for(name: str) -> float:
    """Standardwert eines Reglers."""
    for s in SLIDERS:
        if s.name == name:
            return s.default
    return 0.0


class TestPresetValidity(unittest.TestCase):
    """Jeder Preset-Wert gültig im SLIDERS-Bereich."""

    def test_all_preset_values_in_range(self):
        smap = _slider_map()
        for name, vals in PRESETS.items():
            with self.subTest(preset=name):
                for key, val in vals.items():
                    self.assertIn(key, smap,
                                  f"Preset {name}: unbekannter Regler {key}")
                    lo, hi = smap[key]
                    self.assertGreaterEqual(val, lo,
                                            f"Preset {name}: {key}={val} < min={lo}")
                    self.assertLessEqual(val, hi,
                                         f"Preset {name}: {key}={val} > max={hi}")

    def test_film_presets_exist(self):
        expected = [
            "film_portra_400", "film_portra_800", "film_velvia_50",
            "film_velvia_100", "film_vision3_500t", "film_gold_200",
            "film_pro_400h", "film_trix_400", "film_hp5",
        ]
        for name in expected:
            self.assertIn(name, PRESETS, f"Film-Preset fehlt: {name}")

    def test_deep_blacks_sepia_neutral_name(self):
        """deep_blacks_sepia bleibt neutral benannt."""
        self.assertIn("deep_blacks_sepia", PRESETS)


class TestClassifyRules(unittest.TestCase):
    """_classify-Kategorien aus bekannten Messwerten."""

    def _classify(self, **kwargs):
        from tools.classify_luts import _classify
        return _classify(**kwargs)

    def test_identity_is_other(self):
        """Alle Werte ≈ Default → other."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0, "temperature": 0.0,
                          "tint": 0.0, "saturation": 1.0, "lift_m": 0.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.0,
                         "lift_m": 0.0},
            green_sat=0.0,
            shadow_hue_val=180.0,
        )
        self.assertEqual(cats, ["other"])

    def test_bleach_bypass(self):
        """Sättigung −0.4, Kontrast +0.2 → bleach_bypass."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 0.6, "contrast": 1.2, "temperature": 0.0,
                         "lift_m": 0.0},
            green_sat=0.0,
            shadow_hue_val=180.0,
        )
        self.assertIn("bleach_bypass", cats)

    def test_warm(self):
        """temperature +0.2 → warm."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.2,
                         "lift_m": 0.0},
            green_sat=0.0,
            shadow_hue_val=180.0,
        )
        self.assertIn("warm", cats)

    def test_cold(self):
        """temperature −0.2 → cold."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": -0.2,
                         "lift_m": 0.0},
            green_sat=0.0,
            shadow_hue_val=180.0,
        )
        self.assertIn("cold", cats)

    def test_nature(self):
        """Grün-Sättigung +0.15 → nature."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.0,
                         "lift_m": 0.0},
            green_sat=0.15,
            shadow_hue_val=180.0,
        )
        self.assertIn("nature", cats)

    def test_deep_blacks(self):
        """lift_m +0.05 (dunklere Blacks) + Schattentönung 30° → deep_blacks."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.0,
                         "lift_m": 0.05},
            green_sat=0.0,
            shadow_hue_val=30.0,
        )
        self.assertIn("deep_blacks", cats)

    def test_log_input(self):
        """Graurampe: exposure +2, contrast 2.0 → log_input."""
        cats = self._classify(
            ramp_metrics={"exposure": 2.0, "contrast": 2.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.0,
                         "lift_m": 0.0},
            green_sat=0.0,
            shadow_hue_val=180.0,
        )
        self.assertIn("log_input", cats)

    def test_multiple_categories(self):
        """warm + nature gleichzeitig möglich."""
        cats = self._classify(
            ramp_metrics={"exposure": 0.0, "contrast": 1.0},
            avg_sliders={"saturation": 1.0, "contrast": 1.0, "temperature": 0.2,
                         "lift_m": 0.0},
            green_sat=0.15,
            shadow_hue_val=180.0,
        )
        self.assertIn("warm", cats)
        self.assertIn("nature", cats)


class TestClassifyLutEndToEnd(unittest.TestCase):
    """classify_lut auf selbst erzeugten LUTs (end-to-end)."""

    def _build_lut(self, vals: dict[str, float],
                   device: torch.device | None = None) -> dict:
        def fn(img: torch.Tensor) -> torch.Tensor:
            return grade(img, vals)
        return build(fn, size=33, device=device)

    def test_identity_lut_is_other(self):
        """Identitäts-LUT → Kategorie 'other', structure ≈ 1, banding ≈ 0."""
        from tools.classify_luts import classify_lut
        dev = _get_device()
        lut = self._build_lut({}, device=dev)
        result = classify_lut(lut, dev)

        self.assertIn("other", result["categories"])
        self.assertGreater(result["structure_mean"], 0.99,
                           f"Identitäts-LUT: structure={result['structure_mean']}")
        self.assertLess(result["banding_mean"], 0.01,
                        f"Identitäts-LUT: banding={result['banding_mean']}")

    def test_warm_lut_category(self):
        """LUT mit temperature=0.3 → Kategorie 'warm'."""
        from tools.classify_luts import classify_lut
        dev = _get_device()
        lut = self._build_lut({"temperature": 0.3}, device=dev)
        result = classify_lut(lut, dev)
        self.assertIn("warm", result["categories"],
                      f"Erwartet 'warm', bekommen: {result['categories']}")

    def test_cold_lut_category(self):
        """LUT mit temperature=-0.3 → Kategorie 'cold'."""
        from tools.classify_luts import classify_lut
        dev = _get_device()
        lut = self._build_lut({"temperature": -0.3}, device=dev)
        result = classify_lut(lut, dev)
        self.assertIn("cold", result["categories"],
                      f"Erwartet 'cold', bekommen: {result['categories']}")

    def test_bleach_bypass_category(self):
        """LUT mit Sättigung 0.5 + Kontrast 1.3 → 'bleach_bypass'."""
        from tools.classify_luts import classify_lut
        dev = _get_device()
        lut = self._build_lut({"saturation": 0.5, "contrast": 1.3}, device=dev)
        result = classify_lut(lut, dev)
        self.assertIn("bleach_bypass", result["categories"],
                      f"Erwartet 'bleach_bypass', bekommen: {result['categories']}")

    def test_deep_blacks_category(self):
        """LUT mit lift_m=-0.08, shadow_hue=30, shadow_amount=0.5 → 'deep_blacks'."""
        from tools.classify_luts import classify_lut
        dev = _get_device()
        vals = {"lift_m": -0.08, "shadow_hue": 30, "shadow_amount": 0.5}
        lut = self._build_lut(vals, device=dev)
        result = classify_lut(lut, dev)
        self.assertIn("deep_blacks", result["categories"],
                      f"Erwartet 'deep_blacks', bekommen: {result['categories']}")


if __name__ == "__main__":
    unittest.main()
