"""PRESETS: Name → Reglerwerte (nur Daten)."""

from __future__ import annotations

PRESETS: dict[str, dict[str, float]] = {
    "none": {},
    "bleach_bypass": {
        "saturation": 0.55,
        "contrast": 1.25,
        "pivot": 0.45,
        "highlight_rolloff": 0.3,
    },
    "warm_light": {
        "temperature": 0.35,
        "highlight_hue": 40,
        "highlight_amount": 0.15,
    },
    "cold_light": {
        "temperature": -0.35,
        "tint": -0.05,
        "shadow_hue": 220,
        "shadow_amount": 0.15,
    },
    "architecture_clean": {
        "contrast": 1.1,
        "saturation": 0.9,
        "vibrance": 0.1,
        "gain_m": 1.03,
    },
    "nature_greens": {
        "sel_green_hue": 6,
        "sel_green_sat": 0.15,
        "vibrance": 0.15,
    },
    "deep_blacks_sepia": {
        "lift_m": -0.04,
        "contrast": 1.15,
        "pivot": 0.45,
        "shadow_hue": 30,
        "shadow_amount": 0.25,
        "saturation": 0.92,
    },
    # Film-Presets: Daten aus EasyColorCorrector FILM_STOCKS (MIT)
    "film_portra_400": {
        "temperature": 0.15,
        "tint": 0.05,
        "contrast": 0.9,
        "saturation": 0.9,
        "gamma_m": 0.95,
    },
    "film_portra_800": {
        "temperature": 0.18,
        "tint": 0.08,
        "contrast": 0.9,
        "saturation": 0.9,
        "gamma_m": 0.92,
    },
    "film_velvia_50": {
        "temperature": -0.1,
        "tint": -0.05,
        "contrast": 1.2,
        "saturation": 1.2,
        "gamma_m": 1.1,
    },
    "film_velvia_100": {
        "temperature": -0.08,
        "tint": -0.03,
        "contrast": 1.1,
        "saturation": 1.2,
        "gamma_m": 1.05,
    },
    "film_vision3_500t": {
        "temperature": 0.25,
        "tint": 0.1,
        "contrast": 1.0,
        "saturation": 1.0,
        "gamma_m": 0.9,
    },
    "film_gold_200": {
        "temperature": 0.2,
        "tint": 0.15,
        "contrast": 1.0,
        "saturation": 0.95,
        "gamma_m": 0.88,
    },
    "film_pro_400h": {
        "temperature": 0.05,
        "tint": 0.02,
        "contrast": 0.8,
        "saturation": 0.85,
        "gamma_m": 0.85,
    },
    "film_trix_400": {
        "temperature": 0.0,
        "tint": 0.0,
        "contrast": 1.2,
        "saturation": 0.0,
        "gamma_m": 1.2,
    },
    "film_hp5": {
        "temperature": 0.0,
        "tint": 0.0,
        "contrast": 1.4,
        "saturation": 0.0,
        "gamma_m": 1.3,
    },
}
