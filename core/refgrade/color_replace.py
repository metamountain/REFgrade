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
    return [f"cr_{i}_{k}" for k in ("src_hue", "src_sat", "src_bright", "tgt_hue", "tgt_sat", "tgt_bright", "intensity", "range", "sat_floor")]


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
        src_bright = values.get(f"cr_{i}_src_bright", 0.5)
        tgt_hue = values.get(f"cr_{i}_tgt_hue", 0.0)
        tgt_sat = values.get(f"cr_{i}_tgt_sat", 1.0)
        tgt_bright = values.get(f"cr_{i}_tgt_bright", 0.5)
        range_deg = values.get(f"cr_{i}_range", 30.0)
        sat_floor = values.get(f"cr_{i}_sat_floor", 0.15)
        y = _replace_one(y, src_hue, src_sat, src_bright, tgt_hue, tgt_sat, tgt_bright, intensity, range_deg, sat_floor)
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
    sat_floor: float | torch.Tensor = 0.15,
) -> torch.Tensor:
    """Eine Instanz: Pixel im Hue-Bereich um src_hue werden zu tgt_Farbe gemischt.

    src_sat skaliert die Maske: gesättigte Quelle → stärkerer Effekt.
    src_bright: schwach modulierender Helligkeits-Faktor (0.7..1.0) — V nahe src_bright
        → volle Selektion (1.0), V weit entfernt → nur 0.7. Farbe bleibt primär,
        Helligkeit ist nur feine Abstimmung (dunkle + helle Varianten werden getroffen).
    tgt_bright: Helligkeit der Zielfarbe. Die OUTPUT-Helligkeit der ersetzten Pixel skaliert mit
        tgt_bright / src_bright (Quellfarbe → Zielfarbe): gleich = unverändert, Ziel halb so hell
        = halb so hell (2026-10-17, Standard 0.5/0.5; alte 1.0/1.0 bleiben neutral).
        Faktor begrenzt auf 0..4, Ergebnis geclampt 0..1.
    sat_floor: Sättigungs-Guard (2026-10-18): die Sättigung, bei der der Effekt voll greift.
        Pixel mit S ≥ sat_floor sind voll betroffen, darunter (bis sat_floor−0.13) weicher
        Abfall, darunter gar nicht. Standard 0.15 = altes Verhalten (guard bei S<0.02 aus).
        0.0 = kein Guard (auch Weiß/Grau, S≈0, wird getroffen).
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

    # Sättigungsguard: graue Pixel (S≈0) werden nicht berührt. sat_floor verschiebt die
    # Schwelle: Standard 0.15 → Rampenstart 0.02 (altes Verhalten); 0.0 → kein Guard.
    sat_start = sat_floor - 0.13
    sat_guard = ((S - sat_start) / 0.13).clamp(0.0, 1.0)
    sat_guard = sat_guard * sat_guard * (3.0 - 2.0 * sat_guard)  # smoothstep

    # Helligkeits-Proximität (0..1): 1 wenn V ≈ src_bright, 0 wenn |V-src_bright| ≥ 0.25.
    # Wird unten zu einem schwachen Modulator (0.7..1.0) — kein hartes Selektions-Gate.
    bright_gate = (1.0 - (V - src_bright).abs() * 4.0).clamp(0.0, 1.0)

    # Endgültige Maske: hue-plateau × sat_guard × bright_mod × src_sat × intensity
    # Helligkeit nur schwach modulierend (0.7..1.0): Farbe ist primär,
    # Helligkeit nur feine Abstimmung. Dunkle + helle Varianten derselben
    # Farbe werden beide getroffen.
    bright_mod = 0.7 + 0.3 * bright_gate
    mask = mask * sat_guard * bright_mod * src_sat * intensity

    # Ziel-Farbe: tgt_hue/tgt_sat, OUTPUT-Helligkeit = Pixel-V × tgt_bright / src_bright
    gain = torch.as_tensor(tgt_bright, dtype=V.dtype, device=V.device) / torch.as_tensor(
        src_bright, dtype=V.dtype, device=V.device).clamp(min=0.05)
    V_tgt = (V * gain.clamp(0.0, 4.0)).clamp(0.0, 1.0)
    tgt_rgb = _hsv_to_rgb(
        torch.zeros_like(h) + tgt_hue,
        torch.zeros_like(S) + tgt_sat,
        V_tgt,
    )

    # Mischen: original → ziel, gewichtet durch Maske ([B,H,W,1] broadcastet auf 3 Kanäle).
    # Final-Clamp: intensity > 1 (Overshoot) kann über 0..1 hinausreichen.
    return (x + mask * (tgt_rgb - x)).clamp(0.0, 1.0)
