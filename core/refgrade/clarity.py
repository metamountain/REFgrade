"""Clarity: Gaussian unsharp on L* only, masked to midtones. Not LUT-capable."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .color import rgb_to_lab, lab_to_rgb

CLARITY_AMOUNT = 0.6
CLARITY_RADIUS = 0.008


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel, size = 6*ceil(sigma)+1, normalized."""
    k = int(6 * math.ceil(sigma) + 1)
    x = torch.arange(k, dtype=dtype, device=device) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def apply(img: torch.Tensor, clarity: float, full_size: int,
          scale: float | None = None) -> torch.Tensor:
    """Apply clarity (local contrast) to an sRGB image.

    Args:
        img: [B,H,W,3] float32, 0..1, sRGB-encoded.
        clarity: Slider value, -1..1. 0 returns input unchanged.
        full_size: max(W,H) of the *original* image (for halfSize in original pixels).

    Returns:
        [B,H,W,3] float32, 0..1, sRGB-encoded.
    """
    if clarity == 0.0:
        return img

    amount = CLARITY_AMOUNT * clarity
    h, w = img.shape[1], img.shape[2]
    # scale: pixels of `img` per original pixel; a full-resolution crop passes 1.0 (its size
    # says nothing about the original's) — debug round 2026-10-17
    preview_scale = max(h, w) / full_size if scale is None else scale
    half_size = CLARITY_RADIUS * full_size * preview_scale
    sigma = 0.3 * half_size + 0.3
    kernel = _gaussian_kernel(sigma, img.device, img.dtype)

    # Convert to Lab, operate on L* only
    lab = rgb_to_lab(img)
    L = lab[..., 0:1]  # [B,H,W,1]
    L_ch = L.permute(0, 3, 1, 2)  # [B,1,H,W]

    # 2D Gaussian blur (separable)
    pad = kernel.shape[0] // 2
    L_blur = F.conv2d(F.pad(L_ch, (pad, pad, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    L_blur = F.conv2d(F.pad(L_blur, (0, 0, pad, pad), mode="reflect"), kernel.view(1, 1, -1, 1))

    # Unsharp: L_out = L*(1+amount) - amount*blur(L)
    L_sharp = L_ch * (1.0 + amount) - amount * L_blur

    # Midtone mask from original L (before clarity), rounded to 8-bit
    L01 = L.squeeze(-1) / 100.0  # [B,H,W] in 0..1
    m = torch.round(255.0 * 4.0 * L01 * (1.0 - L01)).clamp(0, 255) / 255.0
    m = m.unsqueeze(-1)  # [B,H,W,1]

    # Blend: out = x + m*(unsharp(x) - x)
    L_out = L_ch + m.permute(0, 3, 1, 2) * (L_sharp - L_ch)

    lab = lab.clone()
    lab[..., 0:1] = L_out.permute(0, 2, 3, 1)
    return lab_to_rgb(lab).clamp(0.0, 1.0)
