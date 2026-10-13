"""White balance: Temperature/Tint values that neutralise a colour cast (user 2026-10-08).

Matches grade._temp_tint exactly (von Kries since 2026-10-13): channel gains in linear light
R·2^(0.8·t), G·2^(−0.5·n), B·2^(−0.8·t). A patch with linear mean (r, g, b) becomes neutral for
t = log2(b/r)/1.6 and n = −2·(0.8·t + log2(r/g)) — `_to_values` solves exactly that.

- `estimate`: automatic — median a*/b* of the near-neutral pixels (mid lightness, low
  chroma: plaster, concrete, sky, paper); if there are too few, the whole image (grey world).
  The median keeps single colourful objects from pulling the result.
- `sample`: eyedropper — mean a*/b* of a small patch the user says should be neutral.
"""

from __future__ import annotations

import torch

import math

from .color import rgb_to_lab, srgb_to_linear

L_RANGE = (20.0, 95.0)       # ignore near-black and clipped highlights
NEUTRAL_CHROMA = 25.0        # "probably neutral" if the cast is below this
MIN_NEUTRAL_SHARE = 0.02     # below this share of pixels: fall back to grey world


def _to_values(rgb_lin: torch.Tensor) -> dict[str, float]:
    """Temperature/Tint that make the linear colour rgb_lin = (r, g, b) neutral."""
    r, g, b = (max(float(v), 1e-6) for v in rgb_lin)
    t = math.log2(b / r) / 1.6
    n = -2.0 * (0.8 * t + math.log2(r / g))
    clamp = lambda v: max(-1.0, min(1.0, v))  # noqa: E731
    return {"temperature": round(clamp(t), 3), "tint": round(clamp(n), 3)}


def estimate(rgb: torch.Tensor) -> dict[str, float]:
    """Auto white balance of an sRGB image [..., 3] in 0..1."""
    lab = rgb_to_lab(rgb.reshape(1, 1, -1, 3)).reshape(-1, 3)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    keep = (L > L_RANGE[0]) & (L < L_RANGE[1])
    neutral = keep & (torch.sqrt(a * a + b * b) < NEUTRAL_CHROMA)
    if neutral.sum() < MIN_NEUTRAL_SHARE * L.numel():
        neutral = keep if keep.any() else torch.ones_like(keep)
    lin = srgb_to_linear(rgb.reshape(-1, 3).clamp(0.0, 1.0))[neutral]
    return _to_values(lin.median(dim=0).values)


def sample(rgb: torch.Tensor, x: float, y: float, radius: int = 4) -> dict[str, float]:
    """Eyedropper: neutralise the patch of (2·radius+1)² pixels around (x, y) of [1,H,W,3]."""
    _n, h, w, _c = rgb.shape
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"point ({x}, {y}) outside the image {w}×{h}")
    patch = rgb[:, y0:y1, x0:x1, :].float().reshape(-1, 3).clamp(0.0, 1.0)
    return _to_values(srgb_to_linear(patch).mean(dim=0))
