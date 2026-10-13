"""Dehaze: Dark Channel Prior (He et al. 2009). Not LUT-capable."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEHAZE_OMEGA = 0.95
DEHAZE_T_MIN = 0.1
DEHAZE_WINDOW_REF = 15  # dark channel window in px at 1024 reference
DEHAZE_GUIDED_R_REF = 40  # guided filter radius in px at 1024 reference
DEHAZE_GUIDED_EPS = 1e-3


def _box_kernel(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2D box kernel of given size, normalized."""
    size = max(1, size)
    return torch.ones(1, 1, size, size, dtype=dtype, device=device) / (size * size)


def _dark_channel(img: torch.Tensor, window: int) -> torch.Tensor:
    """Dark channel: min over 3 channels in a square window.

    Args:
        img: [B,H,W,3] float32, 0..1.
        window: odd window size in pixels.

    Returns:
        [B,H,W] float32.
    """
    if window <= 1:
        return img.min(dim=-1).values  # [B,H,W]

    # Min over channels first: [B,H,W]
    min_ch = img.min(dim=-1).values  # [B,H,W]
    min_ch = min_ch.unsqueeze(1)  # [B,1,H,W]

    # Min filter via -max of negated (box min = -box_max of -x)
    pad = window // 2
    # Use a sliding window minimum via conv with a box kernel on negated values
    # For exact min, we use a large negative value approach with conv
    # Simpler: use F.avg_pool on negated then negate back — but that gives mean, not min.
    # Correct approach: use a min-pooling (available as F.pool2d or manual)
    # torch doesn't have min_pool2d, so we use the trick: min = -max(-x)
    # F.max_pool2d gives max, so: min = -F.max_pool2d(-x)
    neg = -min_ch
    # Use max_pool2d with the window size, stride 1, padding
    neg_max = F.max_pool2d(neg, kernel_size=window, stride=1, padding=pad)
    return -neg_max  # [B,1,H,W]


def _atmospheric_light(dark: torch.Tensor) -> torch.Tensor:
    """Atmospheric light: mean of the brightest 0.1% of the dark channel.

    Args:
        dark: [B,1,H,W] float32.

    Returns:
        [B,1,1,3] float32 (per-channel atmospheric light from the original image
        at the brightest dark-channel pixel).
    """
    # For simplicity and following the spec literally: mean of brightest 0.1% of dark values
    # We compute the atmospheric light as the mean of the top 0.1% pixels of the dark channel
    B, _, H, W = dark.shape
    n_pixels = H * W
    n_top = max(1, int(0.001 * n_pixels))
    flat = dark.squeeze(1).reshape(B, -1)  # [B, H*W]
    # Get the top 0.1% values (brightest)
    top_vals, top_idx = torch.topk(flat, n_top, dim=1)  # [B, n_top]
    # Mean of top values per batch
    A_scalar = top_vals.mean(dim=1, keepdim=True)  # [B,1]
    # For per-channel atmospheric light, we use the scalar to look up the original image
    # at the brightest locations. Simplification: use the mean of the original image
    # at those locations.
    # For the dehaze formula, A is per-channel. We compute it as the mean of the
    # original image at the top-0.1% dark-channel locations.
    return A_scalar  # [B,1] — we'll use this as a scalar per channel


def _guided_filter(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """Box guided filter (He et al. 2010).

    Args:
        guide: [B,1,H,W] float32 (guide image, e.g. luminance).
        src: [B,1,H,W] float32 (input to smooth, e.g. transmission).
        radius: filter radius in pixels.
        eps: regularization.

    Returns:
        [B,1,H,W] float32.
    """
    if radius <= 0:
        return src

    size = 2 * radius + 1
    kernel = _box_kernel(size, guide.device, guide.dtype)
    pad = radius

    # mean_I, mean_p
    mean_I = F.conv2d(F.pad(guide, (pad, pad, pad, pad), mode="reflect"), kernel)
    mean_p = F.conv2d(F.pad(src, (pad, pad, pad, pad), mode="reflect"), kernel)

    # var_Ip = mean(I*p) - mean_I * mean_p
    mean_Ip = F.conv2d(F.pad(guide, (pad, pad, pad, pad), mode="reflect") *
                       F.pad(src, (pad, pad, pad, pad), mode="reflect"), kernel)
    var_Ip = mean_Ip - mean_I * mean_p

    # var_I = mean(I*I) - mean_I * mean_I
    guide_sq = guide * guide
    mean_I2 = F.conv2d(F.pad(guide_sq, (pad, pad, pad, pad), mode="reflect"), kernel)
    var_I = mean_I2 - mean_I * mean_I

    # a = var_Ip / (var_I + eps)
    a = var_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # q = a * I + b
    return a * guide + b


def apply(img: torch.Tensor, dehaze: float, full_size: int) -> torch.Tensor:
    """Apply dehaze (or add haze) to an sRGB image.

    Args:
        img: [B,H,W,3] float32, 0..1, sRGB-encoded.
        dehaze: Slider value, -1..1. 0 returns input unchanged.
            Positive = remove haze, negative = add haze.
        full_size: max(W,H) of the original image (for scaling).

    Returns:
        [B,H,W,3] float32, 0..1, sRGB-encoded.
    """
    if dehaze == 0.0:
        return img

    h, w = img.shape[1], img.shape[2]
    scale = max(h, w) / full_size

    # Dark channel window (odd, >= 3)
    window = max(3, int(DEHAZE_WINDOW_REF * scale) | 1)  # ensure odd
    # Guided filter radius
    gf_radius = max(1, int(DEHAZE_GUIDED_R_REF * scale))

    # Compute dark channel
    dark = _dark_channel(img, window)  # [B,1,H,W]

    # Atmospheric light: mean of brightest 0.1% of dark channel
    # Per-channel: we use the scalar A for all channels (common simplification)
    B = img.shape[0]
    n_pixels = h * w
    n_top = max(1, int(0.001 * n_pixels))
    flat_dark = dark.squeeze(1).reshape(B, -1)
    top_vals, top_idx = torch.topk(flat_dark, n_top, dim=1)  # [B, n_top]
    # Get the original image values at those locations for per-channel A
    flat_img = img.reshape(B, n_pixels, 3)
    # top_idx is [B, n_top], we need to gather from flat_img [B, n_pixels, 3]
    # Expand top_idx to [B, n_top, 1] and gather
    A = flat_img.gather(1, top_idx.unsqueeze(-1).expand(B, n_top, 3)).mean(dim=1)  # [B,3]
    A_img = A.view(B, 1, 1, 3)  # [B,1,1,3]

    # Transmission: t = 1 - omega * dark / A
    omega = DEHAZE_OMEGA * abs(dehaze)
    # A per-channel for the dark channel (which is already min over channels)
    # Use the minimum of A across channels for the dark channel division
    A_dark = A_img.min(dim=-1, keepdim=True).values  # [B,1,1,1]
    t = 1.0 - omega * dark / A_dark.clamp(min=1e-6)
    t = t.clamp(min=DEHAZE_T_MIN, max=1.0)  # [B,1,H,W]

    # Smooth transmission with guided filter (guide = luminance of input)
    lum = (0.299 * img[..., 0:1] + 0.587 * img[..., 1:2] + 0.114 * img[..., 2:3])  # [B,H,W,1]
    lum_ch = lum.permute(0, 3, 1, 2)  # [B,1,H,W]
    t_smooth = _guided_filter(lum_ch, t, gf_radius, DEHAZE_GUIDED_EPS)  # [B,1,H,W]

    # Reshape t_smooth from [B,1,H,W] to [B,H,W,1] to broadcast over channels
    t_out = t_smooth.permute(0, 2, 3, 1)  # [B,H,W,1]

    if dehaze > 0:
        # Dehaze: J = (I - A) / t + A
        result = (img - A_img) / t_out.clamp(min=DEHAZE_T_MIN) + A_img
    else:
        # Add haze: I_out = I * t + A * (1 - t)
        result = img * t_out + A_img * (1.0 - t_out)

    return result.clamp(0.0, 1.0)
