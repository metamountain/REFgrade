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
    return [f"cr_{i}_{k}" for k in ("src_hue", "src_sat", "src_bright", "tgt_hue", "tgt_sat", "tgt_bright", "intensity", "range")]


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
        src_bright = values.get(f"cr_{i}_src_bright", 1.0)
        tgt_hue = values.get(f"cr_{i}_tgt_hue", 0.0)
        tgt_sat = values.get(f"cr_{i}_tgt_sat", 1.0)
        tgt_bright = values.get(f"cr_{i}_tgt_bright", 1.0)
        range_deg = values.get(f"cr_{i}_range", 30.0)
        y = _replace_one(y, src_hue, src_sat, src_bright, tgt_hue, tgt_sat, tgt_bright, intensity, range_deg)
    return y


def _replace_one(
    x: torch.Tensor,
    src_hue: float | torch.Tensor,
    src_sat: float | torch.Tensor,
    src_bright: float | torch.Tensor,
    tgt_hue: float | torch.Tensor,
    tgt_sat: float | torch.Tensor,
    tgt_bright: float | torch.Tensor,
    intensity: float | torch.Tensor,
    range_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Eine Instanz: Pixel im Hue-Bereich um src_hue werden zu tgt_Farbe gemischt.

    src_sat skaliert die Maske: gesättigte Quelle → stärkerer Effekt.
    src_bright: Proximitätsfilter für die SELECTION — nur Pixel mit V nahe src_bright
        werden ersetzt (1.0=volle Helligkeit, 0.5=nur Midtones, 0.0=nur Schatten).
    tgt_bright: skaliert die OUTPUT-Helligkeit der ersetzten Pixel
        (1.0=normal, <1.0=dunkler, >1.0= heller, geclampt 0..1).
    """
    h, S, V = _rgb_to_hsv(x)

    # Hue-Distanz (kreisförmig, 0..180)
    dh = torch.remainder(h - src_hue + 180.0, 360.0) - 180.0
    hue_dist = dh.abs()

    # Maske mit Plateau: Zentrum voll (mask=1), dann weicher Abfall am Rand.
    # Ohne Plateau bleibt im Rand Blau stehen → Pink statt sauberem Rot.
    # range_deg ist Float oder Tensor (Gradient-Test); UI-Minimum ist 1.0.
    t = (hue_dist / range_deg).clamp(0.0, 1.0)
    plateau = 0.6
    mask = torch.where(t < plateau, torch.ones_like(t), (1.0 - t) / (1.0 - plateau))

    # Sättigungsguard: graue Pixel (S≈0) werden nicht berührt
    sat_guard = ((S - 0.02) / 0.13).clamp(0.0, 1.0)
    sat_guard = sat_guard * sat_guard * (3.0 - 2.0 * sat_guard)  # smoothstep

    # Helligkeits-Proximitätsfilter (SELECTION): nur Pixel mit V nahe src_bright matchen.
    # Halbfenster 0.25: 1.0 → V∈[0.75,1] (volle Helligkeit), 0.5 → V∈[0.25,0.75] (nur Midtones),
    # 0.0 → V∈[0,0.25] (nur Schatten).
    bright_gate = (1.0 - (V - src_bright).abs() * 4.0).clamp(0.0, 1.0)

    # Endgültige Maske: hue-plateau × sat_guard × bright_gate × src_sat × intensity
    mask = mask * sat_guard * bright_gate * src_sat * intensity

    # Ziel-Farbe: tgt_hue/tgt_sat, OUTPUT-Helligkeit = Pixel-V × tgt_bright (geclampt 0..1)
    V_tgt = (V * tgt_bright).clamp(0.0, 1.0)
    tgt_rgb = _hsv_to_rgb(
        torch.zeros_like(h) + tgt_hue,
        torch.zeros_like(S) + tgt_sat,
        V_tgt,
    )

    # Mischen: original → ziel, gewichtet durch Maske ([B,H,W,1] broadcastet auf 3 Kanäle).
    # Final-Clamp: intensity > 1 (Overshoot) kann über 0..1 hinausreichen.
    return (x + mask * (tgt_rgb - x)).clamp(0.0, 1.0)
