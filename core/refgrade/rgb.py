"""RGB-Farbkontrolle: Master + pro Kanal (Gamma/Pedestal/Gain) mit Zonenauswahl.

Reine Funktion: Tensor rein, Tensor raus. Kein Side-Effekt, kein In-Place.
"""

from __future__ import annotations

import torch

_EPS = 1e-6

# Zonengrenzen (Luminanz 0..1)
_SHADOW_EDGE = 0.35
_MID_LO = 0.35
_MID_HI = 0.65
_HIGHLIGHT_EDGE = 0.65


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Kubischer Smoothstep zwischen edge0 und edge1."""
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _zone_masks(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drei Zonenmasken aus Luminanz [B,H,W] → (shadows, midtones, highlights), je [B,H,W]."""
    s = 1.0 - _smoothstep(0.0, _SHADOW_EDGE, y)
    h = _smoothstep(_HIGHLIGHT_EDGE, 1.0, y)
    m = 1.0 - s - h
    return s.clamp(0.0, 1.0), m.clamp(0.0, 1.0), h.clamp(0.0, 1.0)


def _lgg_channel(ch: torch.Tensor, gamma: float, pedestal: float, gain: float) -> torch.Tensor:
    """Einzelner Kanal: Gamma → Gain → Pedestal. Eingabe/Ausgabe [B,H,W], 0..1."""
    out = ch.clamp_min(_EPS) ** (1.0 / gamma)
    out = out * gain
    out = out + pedestal
    return out


def _apply_zones(
    orig: torch.Tensor,
    graded: torch.Tensor,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    use_shadows: float,
    use_midtones: float,
    use_highlights: float,
) -> torch.Tensor:
    """Zonales Blending: graded gewichtet nach Zonenanteil, Rest bleibt Original.

    Die drei Gewichte (0..1) skalieren die weichen Zonenmasken. 0 = Zone wirkt
    nicht, 1 = volle Wirkung. Alle drei = 1 ergibt s+m+h=1 (volle Wirkung);
    alle = 0 ergibt keine Wirkung.
    """
    s, m, h = masks
    total = use_shadows * s + use_midtones * m + use_highlights * h
    return orig * (1.0 - total) + graded * total


def rgb_grade(
    x: torch.Tensor,
    master_gamma: float = 1.0,
    master_pedestal: float = 0.0,
    master_gain: float = 1.0,
    r_gamma: float = 1.0,
    r_pedestal: float = 0.0,
    r_gain: float = 1.0,
    g_gamma: float = 1.0,
    g_pedestal: float = 0.0,
    g_gain: float = 1.0,
    b_gamma: float = 1.0,
    b_pedestal: float = 0.0,
    b_gain: float = 1.0,
    use_shadows: float = 1.0,
    use_midtones: float = 1.0,
    use_highlights: float = 1.0,
) -> torch.Tensor:
    """RGB-Farbkontrolle anwenden.

    Reihenfolge pro Pixel:
      1. Master: Gamma → Gain → Pedestal (alle Kanäle gleich)
      2. Kanal:  Gamma → Gain → Pedestal (pro R/G/B, zonales Blending)

    Args:
        x: [B,H,W,3] sRGB 0..1.
        master_gamma/pedestal/gain: Master-Regler.
        r/g/b_gamma/pedestal/gain: Kanal-Regler.
        use_shadows/midtones/highlights: Zonenanteil (0..1) für Kanal-Regler.

    Returns:
        [B,H,W,3] sRGB 0..1 (geclampt).
    """
    out = x.clone()

    # 1. Master (global, ohne Zonen)
    if master_gamma != 1.0 or master_pedestal != 0.0 or master_gain != 1.0:
        out = _lgg_channel(out, master_gamma, master_pedestal, master_gain)

    # 2. Pro Kanal (mit Zonenmasken)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    s, m, h = _zone_masks(y)

    channels = (
        (0, r_gamma, r_pedestal, r_gain),
        (1, g_gamma, g_pedestal, g_gain),
        (2, b_gamma, b_pedestal, b_gain),
    )
    for i, (idx, gamma, pedestal, gain) in enumerate(channels):
        if gamma != 1.0 or pedestal != 0.0 or gain != 1.0:
            base = out[..., idx]
            graded = _lgg_channel(base, gamma, pedestal, gain)
            out[..., idx] = _apply_zones(
                base, graded, (s, m, h),
                use_shadows, use_midtones, use_highlights,
            )

    return out.clamp(0.0, 1.0)


def rgb_channel_curves(
    master_gamma: float = 1.0,
    master_pedestal: float = 0.0,
    master_gain: float = 1.0,
    r_gamma: float = 1.0,
    r_pedestal: float = 0.0,
    r_gain: float = 1.0,
    g_gamma: float = 1.0,
    g_pedestal: float = 0.0,
    g_gain: float = 1.0,
    b_gamma: float = 1.0,
    b_pedestal: float = 0.0,
    b_gain: float = 1.0,
    n_points: int = 256,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Transferkurven pro Kanal für die Anzeige (Schaubild).

    Berechnet die Gesamtkurve (Master + Kanal) für reine R/G/B-Eingaben
    auf einer 1D-Achse, ohne Zonenmasken (Master + Kanal kombiniert).

    Args:
        Reglerwerte wie in rgb_grade.
        n_points: Auflösung der Kurve.
        device: torch-Device (Default: CPU).

    Returns:
        {"x": [...], "r": [...], "g": [...], "b": [...]} — je n_points Werte, 0..1.
    """
    if device is None:
        device = torch.device("cpu")
    xs = torch.linspace(0.0, 1.0, n_points, device=device)

    def curve(gamma: float, pedestal: float, gain: float) -> list[float]:
        # Master dann Kanal (Kanal hier = identisch zur Master-Formel auf 1D)
        v = xs.clamp_min(_EPS) ** (1.0 / master_gamma)
        v = v * master_gain + master_pedestal
        v = v.clamp_min(_EPS) ** (1.0 / gamma)
        v = v * gain + pedestal
        return v.clamp(0.0, 1.0).tolist()

    return {
        "x": xs.tolist(),
        "r": curve(r_gamma, r_pedestal, r_gain),
        "g": curve(g_gamma, g_pedestal, g_gain),
        "b": curve(b_gamma, b_pedestal, b_gain),
    }


def rgb_histogram(x: torch.Tensor, bins: int = 64) -> dict[str, list[float]]:
    """Pro-Kanal-Histogramm für das Schaubild (R/G/B getrennt, normalisiert).

    Args:
        x: [B,H,W,3] sRGB 0..1.
        bins: Anzahl Bins pro Kanal.

    Returns:
        {"r": [...], "g": [...], "b": [...]} — je bins Werte, max = 1.0.
    """
    result: dict[str, list[float]] = {}
    for i, name in enumerate(("r", "g", "b")):
        ch = x[..., i].flatten()
        idx = (ch * bins).long().clamp(0, bins - 1)
        hist = torch.bincount(idx, minlength=bins).float()
        mx = hist.max()
        if mx > 0:
            hist = hist / mx
        result[name] = hist.tolist()
    return result
