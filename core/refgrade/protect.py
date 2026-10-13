"""Tonal protection: reduce the weight of the colour match in shadows, midtones, highlights."""

from __future__ import annotations

import torch

from .color import rgb_to_lab
from .zonal import zone_weights


def protect(original: torch.Tensor, matched: torch.Tensor, values: dict) -> torch.Tensor:
    """Blend `matched` back toward `original` by zone (weights from the original's L).

    Per pixel: out = orig + (matched - orig) * (1 - ps*w_sh - pm*w_mid - ph*w_hi), so 1.0 in a
    zone means "no correction there". Depends only on the pixel itself -> stays LUT-capable.
    """
    ps = float(values.get("protect_shadows", 0.0))
    pm = float(values.get("protect_midtones", 0.0))
    ph = float(values.get("protect_highlights", 0.0))
    if ps == 0.0 and pm == 0.0 and ph == 0.0:
        return matched
    w_sh, w_hi, w_mid = zone_weights(rgb_to_lab(original)[..., 0])
    keep = (1.0 - ps * w_sh - pm * w_mid - ph * w_hi).clamp(0.0, 1.0).unsqueeze(-1)
    return original + (matched - original) * keep
