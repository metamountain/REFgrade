"""SLIDERS: die eine Tabelle aller Regler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slider:
    name: str
    group: str
    min: float
    max: float
    default: float
    step: float
    core: bool = False


def _lgg() -> list[Slider]:
    chans = ("m", "r", "g", "b")
    out: list[Slider] = []
    for c in chans:
        out.append(Slider(f"lift_{c}", "lgg", -0.5, 0.5, 0.0, 0.01,
                          core=(c == "m")))
    for c in chans:
        out.append(Slider(f"gamma_{c}", "lgg", 0.2, 5.0, 1.0, 0.01,
                          core=(c == "m")))
    for c in chans:
        out.append(Slider(f"gain_{c}", "lgg", 0.0, 4.0, 1.0, 0.01,
                          core=(c == "m")))
    out.append(Slider("wheel_blending", "lgg", 0.0, 1.0, 0.5, 0.01))   # Lightroom-Mitte (2026-10-11)
    out.append(Slider("wheel_balance", "lgg", -1.0, 1.0, -0.3, 0.01))   # Nutzer 2026-10-12: Lichter sonst zu dominant
    out.append(Slider("wheel_intensity", "lgg", 0.0, 2.0, 1.0, 0.01))  # Reichweite der Räder
    return out


def _rgb() -> list[Slider]:
    out: list[Slider] = []
    for prefix in ("m", "r", "g", "b"):
        out.append(Slider(f"rgb_{prefix}_gamma", "rgb", 0.2, 5.0, 1.0, 0.01))
        out.append(Slider(f"rgb_{prefix}_pedestal", "rgb", -0.5, 0.5, 0.0, 0.01))
        out.append(Slider(f"rgb_{prefix}_gain", "rgb", 0.0, 4.0, 1.0, 0.01))
    for zone in ("shadows", "midtones", "highlights"):
        out.append(Slider(f"rgb_use_{zone}", "rgb", 0.0, 1.0, 1.0, 1.0))
    return out


def _sel() -> list[Slider]:
    bands = ("red", "yellow", "green", "cyan", "blue", "magenta")
    out: list[Slider] = []
    for b in bands:
        out.append(Slider(f"sel_{b}_hue", "color", -30.0, 30.0, 0.0, 1.0))
        out.append(Slider(f"sel_{b}_sat", "color", -1.0, 1.0, 0.0, 0.01))
        out.append(Slider(f"sel_{b}_light", "color", -1.0, 1.0, 0.0, 0.01))
    return out


def _color_replace() -> list[Slider]:
    out: list[Slider] = []
    for i in range(8):
        out.append(Slider(f"cr_{i}_src_hue", "color_replace", 0.0, 360.0, 0.0, 1.0))
        out.append(Slider(f"cr_{i}_src_sat", "color_replace", 0.0, 1.0, 1.0, 0.01))
        out.append(Slider(f"cr_{i}_src_bright", "color_replace", 0.0, 1.0, 0.5, 0.01))
        out.append(Slider(f"cr_{i}_tgt_hue", "color_replace", 0.0, 360.0, 0.0, 1.0))
        out.append(Slider(f"cr_{i}_tgt_sat", "color_replace", 0.0, 1.0, 1.0, 0.01))
        out.append(Slider(f"cr_{i}_tgt_bright", "color_replace", 0.0, 1.0, 0.5, 0.01))
        out.append(Slider(f"cr_{i}_intensity", "color_replace", 0.0, 2.0, 0.0, 0.01))
        out.append(Slider(f"cr_{i}_range", "color_replace", 1.0, 180.0, 30.0, 1.0))
    return out


SLIDERS: list[Slider] = [
    # match
    Slider("light_strength", "match", 0.0, 1.0, 1.0, 0.01, core=True),
    Slider("color_strength", "match", 0.0, 1.0, 1.0, 0.01, core=True),
    Slider("protect_shadows", "match", 0.0, 1.0, 0.0, 0.01, core=True),
    Slider("protect_midtones", "match", 0.0, 1.0, 0.0, 0.01, core=True),
    Slider("protect_highlights", "match", 0.0, 1.0, 0.0, 0.01, core=True),
    # exposure
    Slider("exposure", "exposure", -4.0, 4.0, 0.0, 0.01),
    Slider("temperature", "exposure", -1.0, 1.0, 0.0, 0.01),
    Slider("tint", "exposure", -1.0, 1.0, 0.0, 0.01),
    # lgg
    *_lgg(),
    # rgb
    *_rgb(),
    # contrast
    Slider("contrast", "contrast", 0.0, 3.0, 1.0, 0.01),
    Slider("pivot", "contrast", 0.0, 1.0, 0.5, 0.01),
    Slider("highlight_rolloff", "contrast", 0.0, 1.0, 0.0, 0.01),
    Slider("highlight_reconstruct", "contrast", 0.0, 1.0, 0.0, 0.01),
    # tone zones (Lightroom-like)
    Slider("highlights", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("shadows", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("whites", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("blacks", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("clarity", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("texture", "contrast", -1.0, 1.0, 0.0, 0.01),
    Slider("dehaze", "contrast", -1.0, 1.0, 0.0, 0.01),
    # color
    Slider("saturation", "color", 0.0, 3.0, 1.0, 0.01),
    Slider("vibrance", "color", -1.0, 1.0, 0.0, 0.01),
    *_sel(),
    # split
    Slider("shadow_hue", "split", 0.0, 360.0, 30.0, 1.0),
    Slider("shadow_amount", "split", 0.0, 1.0, 0.0, 0.01),
    Slider("highlight_hue", "split", 0.0, 360.0, 40.0, 1.0),
    Slider("highlight_amount", "split", 0.0, 1.0, 0.0, 0.01),
    # color replace (5 Instanzen × 6 Werte)
    *_color_replace(),
]
