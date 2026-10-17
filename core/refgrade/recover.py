"""Reconstruct: partial rebuild of clipped colour channels (K39b). Not LUT-capable.

Spatial effect (needs neighbours) -> runs like clarity/texture/dehaze, not through the LUT.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .grade import _rolloff


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel, radius = ceil(3*sigma), normalized."""
    radius = int(math.ceil(3 * sigma))
    k = 2 * radius + 1
    x = torch.arange(k, dtype=dtype, device=device) - radius
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable 2D Gaussian blur, reflect border. x: [B,C,H,W] -> [B,C,H,W]."""
    pad = kernel.shape[0] // 2
    C = x.shape[1]
    kh = kernel.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()  # [C,1,1,k]
    kw = kernel.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()  # [C,1,k,1]
    out = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kh, groups=C)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), kw, groups=C)
    return out


def estimate(x: torch.Tensor, thr: float = 0.98, sigma: float = 8.0) -> torch.Tensor:
    """[B,H,W,3] -> Schätzung ohne Clamp (Werte > 1 möglich).

    Rebuilds channels that are clipped (>= thr) while at least one other channel is not,
    from the colour ratios of the surrounding good pixels. Fully clipped pixels
    (all channels >= thr) are left unchanged.
    """
    x_ref = x.mean(dim=-1)  # [B,H,W]
    maxc = x.max(dim=-1).values  # [B,H,W]
    minc = x.min(dim=-1).values  # [B,H,W]

    good = maxc < thr  # all channels < thr
    part = (maxc >= thr) & (minc < thr)  # at least one >= thr and at least one < thr
    good_f = good.to(x.dtype)  # [B,H,W]

    # Ratios on good pixels: q_c = x_c / (x_ref + 1e-4)
    q = x / (x_ref.unsqueeze(-1) + 1e-4)  # [B,H,W,3]

    # Gaussian blur (separable, reflect) of q_c*good and good
    kernel = _gaussian_kernel(sigma, x.device, x.dtype)
    qg = (q * good_f.unsqueeze(-1)).permute(0, 3, 1, 2)  # [B,3,H,W]
    good_ch = good_f.unsqueeze(1)  # [B,1,H,W]
    qg_blur = _blur(qg, kernel)  # [B,3,H,W]
    good_blur = _blur(good_ch, kernel)  # [B,1,H,W]
    Q = (qg_blur / good_blur.clamp_min(1e-6)).permute(0, 2, 3, 1)  # [B,H,W,3]

    # Reference channel u = largest among the non-clipped channels
    clipped = x >= thr  # [B,H,W,3]
    x_for_max = x.masked_fill(clipped, float("-inf"))  # [B,H,W,3]
    u_idx = x_for_max.argmax(dim=-1)  # [B,H,W]
    x_u = x_for_max.max(dim=-1, keepdim=True).values.clamp_min(0.0)  # [B,H,W,1]
    Q_u = Q.gather(-1, u_idx.unsqueeze(-1)).squeeze(-1)  # [B,H,W]

    # For clipped channels of part pixels: x_c' = max(x_c, x_u * Q_c / max(Q_u, 1e-6))
    candidate = x_u * Q / Q_u.clamp_min(1e-6).unsqueeze(-1)  # [B,H,W,3]
    replace = clipped & part.unsqueeze(-1)  # [B,H,W,3]
    return torch.where(replace, torch.maximum(x, candidate), x)


def apply(x: torch.Tensor, amount: float, thr: float = 0.98, sigma: float = 8.0) -> torch.Tensor:
    """amount 0..1; 0 -> Eingabe unverändert (exakt)."""
    if amount == 0.0:
        return x
    y = estimate(x, thr, sigma)
    y_rolled = _rolloff(y, amount)
    out = x + amount * (y_rolled - x)
    return out.clamp(0.0, 1.0)
