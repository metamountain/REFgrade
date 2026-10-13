"""Hue-Kurven in LCh(ab): fünf PCHIP-Kurven, zyklisch, Chroma-Gewichtung."""

from __future__ import annotations

import torch

from .color import rgb_to_lab, lab_to_rgb
from .curves import HUE_KEYS, cyclic_points, pchip_eval

_EPS = 1e-6
_CYCLIC_KEYS = ("hue_hue", "hue_sat", "hue_lum")
_TWO_PI = 2.0 * 3.141592653589793


def validate(curves: dict) -> None:
    """Prüfe die Hue-Kurven-Format; wirft ValueError bei Verletzung.

    Args:
        curves: Dict mit optionalen Schlüsseln aus HUE_KEYS.

    Raises:
        ValueError: wenn Format, Punktezahl, Monotonie oder Wertebereich verletzt.
    """
    for key in curves:
        if key not in HUE_KEYS:
            raise ValueError(f"Unknown hue curve key: {key!r}")
    for key, pts in curves.items():
        if key not in HUE_KEYS:
            continue
        if not isinstance(pts, (list, tuple)):
            raise ValueError(f"Curve {key!r} must be a list of [x, y] pairs")
        n = len(pts)
        if n < 2:
            raise ValueError(f"Curve {key!r} needs at least 2 points, got {n}")
        if n > 16:
            raise ValueError(f"Curve {key!r} allows at most 16 points, got {n}")
        for i, p in enumerate(pts):
            if len(p) != 2:
                raise ValueError(f"Curve {key!r} point {i} must have exactly 2 values")
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"Curve {key!r} point {i}: x={x} outside [0, 1]")
            if not (0.0 <= y <= 1.0):
                raise ValueError(f"Curve {key!r} point {i}: y={y} outside [0, 1]")
        for i in range(1, n):
            if pts[i][0] <= pts[i - 1][0]:
                raise ValueError(f"Curve {key!r}: x values must be strictly increasing")


def is_identity(curves: dict | None) -> bool:
    """True wenn alle fünf fehlen oder nur Punkte mit y = 0,5 haben."""
    if not curves:
        return True
    for key in HUE_KEYS:
        if key not in curves:
            continue
        for p in curves[key]:
            if abs(float(p[1]) - 0.5) > _EPS:
                return False
    return True


def _cyclic_pchip(x: torch.Tensor, pts: list) -> torch.Tensor:
    """Periodic PCHIP on the hue axis (0..1 wraps): seamless value and slope at 0 = 1."""
    xs, ys = cyclic_points(pts)
    return pchip_eval(x.remainder(1.0), xs, ys)


def _non_cyclic_pchip(x: torch.Tensor, pts: list) -> torch.Tensor:
    """PCHIP, constant outside the first/last point."""
    return pchip_eval(x, [float(p[0]) for p in pts], [float(p[1]) for p in pts])


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Glatte Übergangsfunktion: 0 unter edge0, 1 über edge1."""
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply(x: torch.Tensor, curves: dict) -> torch.Tensor:
    """Hue-Kurven anwenden (nach RGB-Kurven).

    Args:
        x: [B,H,W,3] sRGB 0..1 (nach RGB-Kurven).
        curves: Dict mit optionalen Schlüsseln aus HUE_KEYS.

    Returns:
        [B,H,W,3] sRGB 0..1 (geclampt).
    """
    active = {k: v for k, v in curves.items() if k in HUE_KEYS}
    if not active:
        return x

    lab = rgb_to_lab(x)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    h = torch.atan2(b, a) / _TWO_PI
    h = (h + 1.0) % 1.0
    C = torch.sqrt(a * a + b * b + _EPS)

    # Chroma-Gewichtung für Hue-Kurven
    w = _smoothstep(0.0, 8.0, C)

    # hue_hue: Hue-Verschiebung (y 0..1 → −180°..+180°, d.h. shift = y−0,5 in h-Einheiten)
    if "hue_hue" in active:
        y_val = _cyclic_pchip(h.unsqueeze(-1), active["hue_hue"]).squeeze(-1)
        shift = y_val - 0.5
        h = (h + w * shift) % 1.0

    # hue_sat: Chroma-Faktor
    if "hue_sat" in active:
        y_val = _cyclic_pchip(h.unsqueeze(-1), active["hue_sat"]).squeeze(-1)
        factor = y_val * 2.0
        C = C * (1.0 + w * (factor - 1.0))

    # hue_lum: L-Verschiebung
    if "hue_lum" in active:
        y_val = _cyclic_pchip(h.unsqueeze(-1), active["hue_lum"]).squeeze(-1)
        shift = (y_val - 0.5) * 100.0
        L = L + w * shift

    # lum_sat: Chroma-Faktor nach L
    if "lum_sat" in active:
        x_val = (L / 100.0).clamp(0.0, 1.0)
        y_val = _non_cyclic_pchip(x_val.unsqueeze(-1), active["lum_sat"]).squeeze(-1)
        factor = y_val * 2.0
        C = C * factor

    # sat_sat: Chroma-Faktor nach C
    if "sat_sat" in active:
        x_val = (C / 150.0).clamp(0.0, 1.0)
        y_val = _non_cyclic_pchip(x_val.unsqueeze(-1), active["sat_sat"]).squeeze(-1)
        factor = y_val * 2.0
        C = C * factor

    # LCh → Lab
    a_new = C * torch.cos(h * _TWO_PI)
    b_new = C * torch.sin(h * _TWO_PI)
    lab_out = torch.stack([L, a_new, b_new], dim=-1)
    return lab_to_rgb(lab_out)
