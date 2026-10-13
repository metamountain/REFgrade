"""Texture: high-pass contrast of fine details on L* only. Not LUT-capable."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .color import rgb_to_lab, lab_to_rgb

TEXTURE_AMOUNT = 1.5
TEXTURE_SIGMA_REF = 2.0  # sigma in px at 1024 reference


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel, size = 6*ceil(sigma)+1, normalized."""
    k = max(1, int(6 * math.ceil(sigma) + 1))
    x = torch.arange(k, dtype=dtype, device=device) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def apply(img: torch.Tensor, texture: float, full_size: int) -> torch.Tensor:
    """Apply texture (fine-detail contrast) to an sRGB image.

    Args:
        img: [B,H,W,3] float32, 0..1, sRGB-encoded.
        texture: Slider value, -1..1. 0 returns input unchanged.
        full_size: max(W,H) of the original image (for sigma scaling).

    Returns:
        [B,H,W,3] float32, 0..1, sRGB-encoded.
    """
    if texture == 0.0:
        return img

    amount = TEXTURE_AMOUNT * texture
    h, w = img.shape[1], img.shape[2]
    scale = max(h, w) / full_size
    sigma = TEXTURE_SIGMA_REF * scale
    if sigma < 0.5:
        return img

    kernel = _gaussian_kernel(sigma, img.device, img.dtype)

    # Convert to Lab, operate on L* only
    lab = rgb_to_lab(img)
    L = lab[..., 0:1]  # [B,H,W,1]
    L_ch = L.permute(0, 3, 1, 2)  # [B,1,H,W]

    # 2D Gaussian blur (separable)
    pad = kernel.shape[0] // 2
    L_blur = F.conv2d(F.pad(L_ch, (pad, pad, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
    L_blur = F.conv2d(F.pad(L_blur, (0, 0, pad, pad), mode="reflect"), kernel.view(1, 1, -1, 1))

    # High-pass: detail = L - blur(L)
    detail = L_ch - L_blur

    # Add detail with factor
    L_out = L_ch + amount * detail

    lab = lab.clone()
    lab[..., 0:1] = L_out.permute(0, 2, 3, 1)
    return lab_to_rgb(lab).clamp(0.0, 1.0)
