"""Color Replace: HSL-Distanz-Maske, mehrere Instanzen.

Jede Instanz ersetzt Farben in einem Hue-Bereich (Quelle) durch eine Ziel-Farbe.
Werte: cr_{i}_src_hue, cr_{i}_src_sat, cr_{i}_tgt_hue, cr_{i}_tgt_sat,
        cr_{i}_intensity, cr_{i}_range
"""

from __future__ import annotations

import torch

from .color import rgb_to_hsv as _rgb_to_hsv, hsv_to_rgb as _hsv_to_rgb

MAX_INSTANCES = 8


def _cr_names(i: int) -> list[str]:
    return [f"cr_{i}_{k}" for k in ("src_hue", "src_sat", "tgt_hue", "tgt_sat", "intensity", "range")]


def all_cr_names() -> list[str]:
    """Alle cr_*-Reglernamen (für _is_default-Check)."""
    return [n for i in range(MAX_INSTANCES) for n in _cr_names(i)]


def apply(x: torch.Tensor, values: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Alle aktiven Color-Replace-Instanzen anwenden.

    Args:
        x: sRGB [B,H,W,3] 0..1
        values: flaches Dict mit cr_{i}_* keys

    Returns:
        sRGB [B,H,W,3] 0..1
    """
    y = x
    for i in range(MAX_INSTANCES):
        intensity = values.get(f"cr_{i}_intensity", 0.0)
        # Quick-skip nur bei Python-float (Gradient-Test braucht Tensoren)
        if isinstance(intensity, float) and intensity <= 0.0:
            continue
        src_hue = values.get(f"cr_{i}_src_hue", 0.0)
        src_sat = values.get(f"cr_{i}_src_sat", 1.0)
        tgt_hue = values.get(f"cr_{i}_tgt_hue", 0.0)
        tgt_sat = values.get(f"cr_{i}_tgt_sat", 1.0)
        range_deg = values.get(f"cr_{i}_range", 30.0)
        y = _replace_one(y, src_hue, src_sat, tgt_hue, tgt_sat, intensity, range_deg)
    return y


def _replace_one(
    x: torch.Tensor,
    src_hue: float | torch.Tensor,
    src_sat: float | torch.Tensor,
    tgt_hue: float | torch.Tensor,
    tgt_sat: float | torch.Tensor,
    intensity: float | torch.Tensor,
    range_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Eine Instanz: Pixel im Hue-Bereich um src_hue werden zu tgt_Farbe gemischt.

    src_sat skaliert die Maske: gesättigte Quelle → stärkerer Effekt.
    """
    h, S, V = _rgb_to_hsv(x)

    # Hue-Distanz (kreisförmig, 0..180)
    dh = torch.remainder(h - src_hue + 180.0, 360.0) - 180.0
    hue_dist = dh.abs()

    # Weiche Maske: 1 im Zentrum → 0 am Rand des Bereichs
    mask = (1.0 - hue_dist / range_deg).clamp(0.0, 1.0)

    # Sättigungsguard: graue Pixel (S≈0) werden nicht berührt
    sat_guard = ((S - 0.02) / 0.13).clamp(0.0, 1.0)
    sat_guard = sat_guard * sat_guard * (3.0 - 2.0 * sat_guard)  # smoothstep

    # Endgültige Maske: hue-distanz × sat_guard × src_sat × intensity
    mask = mask * sat_guard * src_sat * intensity

    # Ziel-Farbe: tgt_hue/tgt_sat, Helligkeit des Pixels beibehalten
    # zeros_like + value: Gradient-kompatible "fill" (full_like akzeptiert keine Tensoren)
    tgt_rgb = _hsv_to_rgb(
        torch.zeros_like(h) + tgt_hue,
        torch.zeros_like(S) + tgt_sat,
        V,
    )

    # Mischen: original → ziel, gewichtet durch Maske ([B,H,W,1] broadcastet auf 3 Kanäle)
    return x + mask * (tgt_rgb - x)
