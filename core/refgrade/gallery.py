"""Galerie-Logik: Runden mit 9 Kandidaten (3×3-Raster), Anker in der Mitte."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch

from .sliders import SLIDERS
from .analysis import compare as _compare
from .grade import grade
from .color import rgb_to_lab

_EPS = 1e-6
_N_CANDIDATES = 9
_ANCHOR_IDX = 4

# Basis-Schrittweiten für light/color (erste Runde)
_LIGHT_STEPS = {
    "exposure": 0.5,       # ½ Blende
    "contrast": 0.3,
    # 2026-10-18: Lift wirkt jetzt nur noch in der Schattenzone (Bump-Modell, grosse
    # Lücken) → volle Schrittweite, damit die Kandidaten sichtbar bleiben.
    "lift_m": 0.5,
    "lift_r": 0.5,
}
_COLOR_STEPS = {
    "temperature": 0.2,
    "tint": 0.2,
    "saturation": 0.3,
}
# Schattentönung: bidirektional über shadow_amount + shadow_hue
# (shadow_amount ist 0..1, kann also nicht um Default ± bewegt werden)
_SHADOW_TONING_STEP = 0.3
_SHADOW_HUE_WARM = 30.0
_SHADOW_HUE_COOL = 210.0

# Labels für light
_LIGHT_LABELS = {
    "exposure": ("heller", "dunkler"),
    "contrast": ("mehr Kontrast", "weniger Kontrast"),
    "lift_m": ("Schwarz anheben", "Schwarz drücken"),
    "lift_r": ("Rot anheben", "Rot drücken"),
}
_COLOR_LABELS = {
    "temperature": ("wärmer", "kühler"),
    "tint": ("magenta", "grün"),
    "saturation": ("gesättigter", "entfärbt"),
}

# Matrix-Stärken
_MATRIX_STRENGTHS = [0.4, 0.65, 0.9]
_MATRIX_METHODS = ["reinhard", "mkl", "zonal"]

# Strength-Stufen (Default)
_STRENGTH_LEVELS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Candidate:
    values: dict
    label: str


@dataclass
class GalleryContext:
    """Kontext für eine Galerie-Runde."""
    image: torch.Tensor
    reference: torch.Tensor
    image_analysis: dict
    reference_analysis: dict
    image_weights: torch.Tensor | None = None
    reference_weights: torch.Tensor | None = None
    current_method: str = "reinhard"
    reference_paths: list = field(default_factory=list)
    second_candidate: dict | None = None


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _clamp_value(name: str, val: float) -> float:
    """Wert in den SLIDERS-Bereich klemmen."""
    s = next((s for s in SLIDERS if s.name == name), None)
    if s is None:
        return val
    return max(s.min, min(s.max, val))


def _step_width(base: float, step_level: int) -> float:
    """Schrittweite: halbiert pro Runde."""
    return base / (2 ** step_level)


def _compare_values(src: dict, ref: dict, scale: float = 1.0) -> dict[str, float]:
    """analysis.compare mit Skalierung der Abweichung von Default."""
    result = _compare(src, ref)
    d = _defaults()
    vals = dict(d)
    for k, v in result["sliders"].items():
        if k in d:
            # Skalierung: v ist der Zielwert, d[k] ist der Default
            # Skaliert: d[k] + (v - d[k]) * scale
            vals[k] = _clamp_value(k, d[k] + (v - d[k]) * scale)
    return vals


def _round_start(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Start-Runde: Analyse + Methoden."""
    d = _defaults()
    candidates: list[Candidate] = []

    # 1. analysis.compare (100 %)
    v1 = _compare_values(ctx.image_analysis, ctx.reference_analysis, 1.0)
    candidates.append(Candidate(v1, "Analyse 100 %"))

    # 2. Reinhard (Lab) — LUT-Basis, values = Default
    candidates.append(Candidate(dict(d), "Reinhard"))

    # 3. MKL — LUT-Basis
    candidates.append(Candidate(dict(d), "MKL"))

    # 4. Zonal-Match — LUT-Basis
    candidates.append(Candidate(dict(d), "Zonal"))

    # 5. slider_fit (Placeholder: compare 80 %)
    v5 = _compare_values(ctx.image_analysis, ctx.reference_analysis, 0.8)
    candidates.append(Candidate(v5, "Regler-Fit"))

    # 6. analysis.compare (50 %)
    v6 = _compare_values(ctx.image_analysis, ctx.reference_analysis, 0.5)
    candidates.append(Candidate(v6, "Analyse 50 %"))

    # 7. compare mit auto_ignore (approximiert: compare 70 %)
    v7 = _compare_values(ctx.image_analysis, ctx.reference_analysis, 0.7)
    candidates.append(Candidate(v7, "Analyse + Maske"))

    # 8. Reinhard 50 % (LUT-Basis, values = Default)
    candidates.append(Candidate(dict(d), "Reinhard 50 %"))

    # Anker in die Mitte (Index 4), Kandidaten in die übrigen 8 Slots
    result = [None] * _N_CANDIDATES
    result[_ANCHOR_IDX] = Candidate(dict(anchor), "Anker")
    for i, c in enumerate(candidates):
        slot = i if i < _ANCHOR_IDX else i + 1
        result[slot] = c
    return result


def _round_light(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Light-Runde: exposure, contrast, lift_m, highlight_rolloff."""
    d = _defaults()
    candidates: list[Candidate] = []

    for name, base in _LIGHT_STEPS.items():
        w = _step_width(base, step_level)
        # + Richtung
        v_plus = dict(anchor)
        v_plus[name] = _clamp_value(name, anchor.get(name, d[name]) + w)
        label_plus = _LIGHT_LABELS[name][0]
        # − Richtung
        v_minus = dict(anchor)
        v_minus[name] = _clamp_value(name, anchor.get(name, d[name]) - w)
        label_minus = _LIGHT_LABELS[name][1]
        candidates.append(Candidate(v_plus, label_plus))
        candidates.append(Candidate(v_minus, label_minus))

    result = [None] * _N_CANDIDATES
    result[_ANCHOR_IDX] = Candidate(dict(anchor), "Anker")
    for i, c in enumerate(candidates):
        if i < _ANCHOR_IDX:
            result[i] = c
        else:
            result[i + 1] = c
    return result


def _round_color(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Color-Runde: temperature, tint, saturation, Schattentönung."""
    d = _defaults()
    candidates: list[Candidate] = []

    for name, base in _COLOR_STEPS.items():
        w = _step_width(base, step_level)
        v_plus = dict(anchor)
        v_plus[name] = _clamp_value(name, anchor.get(name, d[name]) + w)
        label_plus = _COLOR_LABELS[name][0]
        v_minus = dict(anchor)
        v_minus[name] = _clamp_value(name, anchor.get(name, d[name]) - w)
        label_minus = _COLOR_LABELS[name][1]
        candidates.append(Candidate(v_plus, label_plus))
        candidates.append(Candidate(v_minus, label_minus))

    # Schattentönung Richtung wärmer/kühler: shadow_amount anheben, Hue drehen
    amt = _step_width(_SHADOW_TONING_STEP, step_level)
    for hue, label in ((_SHADOW_HUE_WARM, "Schatten wärmer"),
                       (_SHADOW_HUE_COOL, "Schatten kühler")):
        v = dict(anchor)
        v["shadow_amount"] = _clamp_value(
            "shadow_amount", anchor.get("shadow_amount", d["shadow_amount"]) + amt)
        v["shadow_hue"] = _clamp_value("shadow_hue", hue)
        candidates.append(Candidate(v, label))

    result = [None] * _N_CANDIDATES
    result[_ANCHOR_IDX] = Candidate(dict(anchor), "Anker")
    for i, c in enumerate(candidates):
        if i < _ANCHOR_IDX:
            result[i] = c
        else:
            result[i + 1] = c
    return result


def _round_matrix(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Matrix-Runde: 3 Methoden × 3 Stärken, kein Anker."""
    candidates: list[Candidate] = []
    for method in _MATRIX_METHODS:
        for strength in _MATRIX_STRENGTHS:
            d = _defaults()
            v = dict(d)
            v["_method"] = method
            v["_strength"] = strength
            label = f"{method} {int(strength*100)} %"
            candidates.append(Candidate(v, label))
    return candidates


def _round_strength(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Strength-Runde: verschiedene Stärken."""
    # 9 Stufen um den Anker
    d = _defaults()
    current = anchor.get("_strength", 0.65)
    step = 0.05 / (2 ** step_level)

    candidates: list[Candidate] = []
    for i in range(_N_CANDIDATES):
        offset = (i - _ANCHOR_IDX) * step
        s = max(0.0, min(1.5, current + offset))
        v = dict(anchor)
        v["_strength"] = s
        label = f"{int(s*100)} %"
        candidates.append(Candidate(v, label))
    return candidates


def _round_reference(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Reference-Runde: verschiedene Vorlagen."""
    d = _defaults()
    candidates: list[Candidate] = []

    paths = ctx.reference_paths
    if not paths:
        # Fallback: nur Anker
        for i in range(_N_CANDIDATES):
            candidates.append(Candidate(dict(anchor), "Anker" if i == _ANCHOR_IDX else f"Vorlage {i}"))
        return candidates

    # Anker in der Mitte
    for i in range(_N_CANDIDATES):
        if i == _ANCHOR_IDX:
            candidates.append(Candidate(dict(anchor), "Aktuelle Vorlage"))
        else:
            # Vorlage aus der Bibliothik
            idx = (i - 1) % len(paths) if i > _ANCHOR_IDX else (i + len(paths)) % len(paths)
            v = dict(anchor)
            v["_reference_idx"] = idx
            candidates.append(Candidate(v, f"Vorlage {idx + 1}"))
    return candidates


def _round_mix(ctx: GalleryContext, anchor: dict, step_level: int) -> list[Candidate]:
    """Mix-Runde: Anker ↔ zweiter Kandidat."""
    d = _defaults()
    candidates: list[Candidate] = []

    second = ctx.second_candidate
    if second is None:
        for i in range(_N_CANDIDATES):
            candidates.append(Candidate(dict(anchor), "Anker" if i == _ANCHOR_IDX else f"Mix {i}"))
        return candidates

    # 9 Stufen: t = 0 (Anker) bis t = 1 (zweiter Kandidat)
    for i in range(_N_CANDIDATES):
        t = i / (_N_CANDIDATES - 1)
        v = {}
        for k in anchor:
            if k.startswith("_"):
                v[k] = anchor[k]
                continue
            a_val = anchor.get(k, d.get(k, 0.0))
            b_val = second.get("values", {}).get(k, d.get(k, 0.0))
            v[k] = a_val + t * (b_val - a_val)
        v["_mix_t"] = t
        label = f"Mix {int(t*100)} %"
        candidates.append(Candidate(v, label))
    return candidates


def round(
    kind: str,
    anchor_values: dict,
    context: GalleryContext,
    step_level: int = 0,
) -> list[Candidate]:
    """Eine Galerie-Runde berechnen.

    Args:
        kind: "start", "light", "color", "matrix", "strength", "reference", "mix"
        anchor_values: aktuelle Reglerwerte (Anker).
        context: GalerieContext mit Bild, Vorlage, Analyse.
        step_level: wie viele Runden dieser Art schon (0 = erste).

    Returns:
        Liste von 9 Candidates. Index 4 = Anker (außer matrix).
    """
    if kind == "start":
        return _round_start(context, anchor_values, step_level)
    elif kind == "light":
        return _round_light(context, anchor_values, step_level)
    elif kind == "color":
        return _round_color(context, anchor_values, step_level)
    elif kind == "matrix":
        return _round_matrix(context, anchor_values, step_level)
    elif kind == "strength":
        return _round_strength(context, anchor_values, step_level)
    elif kind == "reference":
        return _round_reference(context, anchor_values, step_level)
    elif kind == "mix":
        return _round_mix(context, anchor_values, step_level)
    else:
        raise ValueError(f"Unbekannte Runde-Art: {kind!r}")
