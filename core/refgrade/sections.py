"""Bereiche: SECTIONS-Tabelle, VISIBLE, section_values, Kette (chain_fn)."""

from __future__ import annotations

import torch

from .sliders import SLIDERS

# SECTIONS: key → Reglernamen des Bereichs.
# reference: Übertragung aus der Vorlage (protect + method)
# tone: Belichtung, Kontrast, Lightroom-Zonen, Clarity
# color: Temperatur, Tint, Sättigung, Vibrance
# wheels: Lift/Gamma/Gain (je m, r, g, b = 12)
# hsl: Sechs Farbbänder × (Hue, Sat, Light) = 18
# curves: keine Regler; Daten in mix["curves"]
# lut: keine Regler; Pfad in mix["lut"]
SECTIONS: dict[str, tuple[str, ...]] = {
    "reference": ("light_strength", "color_strength", "protect_shadows", "protect_highlights"),
    "tone": ("exposure", "contrast", "highlights", "shadows", "whites", "blacks",
             "clarity", "texture", "dehaze"),
    "color": ("temperature", "tint", "saturation", "vibrance"),
    "wheels": tuple(f"{k}_{c}" for k in ("lift", "gamma", "gain") for c in ("m", "r", "g", "b"))
             + ("wheel_blending", "wheel_balance", "wheel_intensity"),
    "rgb": tuple(f"rgb_{p}_{k}" for p in ("m", "r", "g", "b") for k in ("gamma", "pedestal", "gain"))
           + tuple(f"rgb_use_{z}" for z in ("shadows", "midtones", "highlights")),
    "hsl": tuple(f"sel_{b}_{p}" for b in ("red", "yellow", "green", "cyan", "blue", "magenta")
                 for p in ("hue", "sat", "light")),
    "curves": (),
    "lut": (),
}

# Alle sichtbaren Regler (47).
VISIBLE: tuple[str, ...] = tuple(n for names in SECTIONS.values() for n in names)

# Regler die gefittet werden (tone + color + wheels; wheel_blending/balance/clarity/texture/dehaze ausgeschlossen).
FITTABLE: tuple[str, ...] = tuple(
    n for key, names in SECTIONS.items()
    if key in ("tone", "color", "wheels") for n in names
    if n not in ("wheel_blending", "wheel_balance", "wheel_intensity", "clarity", "texture",
                 "dehaze")
)


def section_values(values: dict, key: str) -> dict:
    """Alle Regler auf Standard, außer denen des Bereichs `key`.

    Args:
        values: Vollständiger Reglerwert-Dict (unvollständig erlaubt).
        key: Bereichsschlüssel aus SECTIONS.

    Returns:
        Dict mit allen SLIDERS-Namen; Bereichsregler aus `values`, Rest auf Standard.

    Raises:
        KeyError: bei unbekanntem Bereichsschlüssel.
    """
    if key not in SECTIONS:
        raise KeyError(f"Unbekannter Bereich: {key!r}")
    defaults = {s.name: s.default for s in SLIDERS}
    result = dict(defaults)
    for name in SECTIONS[key]:
        result[name] = values.get(name, defaults[name])
    return result


# Reihenfolge der Kette (Nutzer 2026-10-12): jeder Bereich arbeitet auf dem Ergebnis des
# vorigen — erst Vorlage und Neutralisieren, dann Helligkeit, dann die Farbräder auf der neuen
# Helligkeit, HSL als Feinarbeit, die kreative LUT zuletzt. mix["order"] überschreibt sie (später
# per Drag & Drop der Kapitel); fehlende Bereiche werden in dieser Reihenfolge angehängt.
DEFAULT_ORDER: tuple[str, ...] = ("reference", "color", "tone", "rgb", "curves", "wheels", "hsl", "lut")


def chain_order(mix: dict) -> tuple[str, ...]:
    """Reihenfolge der Kette aus mix["order"] (optional), geprüft und vervollständigt.

    Raises:
        KeyError: unbekannter Bereich oder doppelter Eintrag in mix["order"].
    """
    order = list(mix.get("order") or ())
    for key in order:
        if key not in DEFAULT_ORDER:
            raise KeyError(f"Unbekannter Bereich in order: {key!r}")
    if len(set(order)) != len(order):
        raise KeyError(f"Doppelter Bereich in order: {order!r}")
    return tuple(order) + tuple(k for k in DEFAULT_ORDER if k not in order)


def chain_step(y: torch.Tensor, f: callable, w: float) -> torch.Tensor:
    """Ein Glied der Kette: y + w · (f(y) − y), auf 0..1 geklemmt (nächster Bereich erwartet 0..1)."""
    return (y + w * (f(y) - y)).clamp(0.0, 1.0)


def chain_fn(source_fns: dict[str, callable], mix: dict) -> callable:
    """Kette: R(x) = clamp(x + master · (y_n − x)), y_i = y_{i−1} + w_i · (f_i(y_{i−1}) − y_{i−1}).

    Vorher additiv parallel (jeder Bereich auf dem Original) — die Farbräder sahen so nie die
    Helligkeit nach TONE (Nutzer 2026-10-12).

    Args:
        source_fns: {key: callable} — sRGB → sRGB pro Bereich.
        mix: {"weights": {key: 0..1}, "master": 0..1, "solo": key | None, "order": [key, …]}.
             SOLO = nur dieser Bereich auf dem Original.

    Returns:
        Callable [B,H,W,3] → [B,H,W,3] (sRGB 0..1).
    """
    weights = mix.get("weights", {})
    master = mix.get("master", 1.0)
    solo = mix.get("solo", None)
    order = chain_order(mix)

    def fn(x: torch.Tensor) -> torch.Tensor:
        if solo is not None:
            return source_fns[solo](x).clamp(0.0, 1.0)
        if master == 0.0:
            return x.clamp(0.0, 1.0)
        y = x.clamp(0.0, 1.0)
        for key in order:
            w = weights.get(key, 0.0)
            if w > 0.0 and key in source_fns:
                y = chain_step(y, source_fns[key], w)
        return (x + master * (y - x)).clamp(0.0, 1.0)

    return fn
