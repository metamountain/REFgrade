"""Shared constants and core-data loaders for RefGrade dialog mixins (K26)."""

from __future__ import annotations

import importlib.util
import sys

from . import config

# Column 1 starts with method + FIT, column 2 ends with LUT…/CANCEL/OK. UI language: English.
COLUMN_1 = (("Reference", ("light_strength", "color_strength", "protect_shadows",
                       "protect_highlights")),
            ("Tone", ("exposure", "contrast", "highlights", "shadows", "whites", "blacks",
                       "clarity", "texture", "dehaze")),
            ("RGB", ("rgb_m_gamma", "rgb_m_pedestal", "rgb_m_gain",
                     "rgb_r_gamma", "rgb_r_pedestal", "rgb_r_gain",
                     "rgb_g_gamma", "rgb_g_pedestal", "rgb_g_gain",
                     "rgb_b_gamma", "rgb_b_pedestal", "rgb_b_gain",
                     "rgb_use_shadows", "rgb_use_midtones", "rgb_use_highlights")))
COLUMN_2 = (("Color", ("temperature", "tint", "saturation", "vibrance")),)
LABELS = {"light_strength": "Light", "color_strength": "Color", "protect_shadows": "Protect shadows",
          "protect_highlights": "Protect highlights", "exposure": "Exposure", "contrast": "Contrast",
          "highlights": "Highlights", "shadows": "Shadows", "whites": "Whites", "blacks": "Blacks",
          "clarity": "Clarity", "texture": "Texture", "dehaze": "Dehaze",
          "temperature": "Temperature", "tint": "Tint",
          "saturation": "Saturation", "vibrance": "Vibrance",
          "wheel_blending": "Blending", "wheel_balance": "Balance",
          "wheel_intensity": "Intensity",
          "rgb_m_gamma": "M Gamma", "rgb_m_pedestal": "M Offset", "rgb_m_gain": "M Gain",
          "rgb_r_gamma": "R Gamma", "rgb_r_pedestal": "R Offset", "rgb_r_gain": "R Gain",
          "rgb_g_gamma": "G Gamma", "rgb_g_pedestal": "G Offset", "rgb_g_gain": "G Gain",
          "rgb_b_gamma": "B Gamma", "rgb_b_pedestal": "B Offset", "rgb_b_gain": "B Gain",
          "rgb_use_shadows": "Shadows", "rgb_use_midtones": "Mids", "rgb_use_highlights": "Highs"}
POP_RECIPE = {"contrast": 1.10, "vibrance": 0.30, "highlights": -0.35, "shadows": 0.35,
              "whites": 0.20, "blacks": -0.20, "clarity": 0.35}
WHEELS = (("gain", "Highlights"), ("gamma", "Midtones"), ("lift", "Shadows"))   # light on top, shadow on bottom
VISIBLE = (tuple(n for col in (COLUMN_1, COLUMN_2) for _t, names in col for n in names)
           + tuple(f"{k}_{c}" for k, _t in WHEELS for c in "mrgb"))
# user 2026-10-11: every section starts OFF and switches ON as soon as something in it changes
# (ON then shows what is actually in play); ↺ switches it OFF again
DEFAULT_ON: tuple[str, ...] = ()
AUTO_SECTIONS = ("color", "tone", "curves")   # K30c: measured — wheels/rgb auto harm good images
CHAIN_ORDER = ("reference", "color", "tone", "rgb", "curves", "wheels", "hsl", "lut")
METHODS = ("REINHARD", "MKL", "ZONAL", "PAIR")
_DEBOUNCE_MS = 30
_CROP_IDLE_MS = 120


def load_sliders() -> list:
    """SLIDERS straight from core (pure dataclasses, no torch), loaded by file path."""
    spec = importlib.util.spec_from_file_location("ref_grade_sliders",
                                                  config.CORE / "refgrade" / "sliders.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses look their module up while loading
    spec.loader.exec_module(mod)
    return mod.SLIDERS


def _load_help_module():
    """Load the help module from core by file path (cached in sys.modules)."""
    if "ref_grade_help" in sys.modules:
        return sys.modules["ref_grade_help"]
    spec = importlib.util.spec_from_file_location("ref_grade_help",
                                                  config.CORE / "refgrade" / "help.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_help() -> dict:
    """HELP straight from core (pure dict, no torch), loaded by file path."""
    return _load_help_module().HELP


def load_auto_help() -> dict:
    """AUTO_HELP straight from core (pure dict, no torch), loaded by file path."""
    return _load_help_module().AUTO_HELP


