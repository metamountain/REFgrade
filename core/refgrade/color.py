"""Farbraum-Wandlungen: sRGB, linear RGB, CIELAB (D65), Rec.709-Luma."""

from __future__ import annotations

import torch
from kornia.color import rgb_to_lab as _kornia_rgb_to_lab
from kornia.color import lab_to_rgb as _kornia_lab_to_rgb

_EPS = 1e-6


def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    """IEC 61966-2-1: sRGB-kodiert [0,1] → linear."""
    return torch.where(
        t <= 0.04045,
        t / 12.92,
        ((t + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(t: torch.Tensor) -> torch.Tensor:
    """IEC 61966-2-1: linear → sRGB-kodiert [0,1]."""
    return torch.where(
        t <= 0.0031308,
        12.92 * t,
        # clamp only matters for the branch torch.where discards: at t = 0 the power's
        # derivative is inf and inf·0 = NaN in the gradient (FIT on any image with black).
        1.055 * t.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(t: torch.Tensor) -> torch.Tensor:
    """sRGB [B,H,W,3] → CIELAB D65 [B,H,W,3]. kornia linearisiert selbst."""
    # clamp min=0: kornia's srgb_to_linear pow((t+0.055)/1.055, 2.4) ist für t<−0.055
    # NaN → 0·NaN=NaN im torch.where-Gradienten. Werte für t≥0 bleiben unverändert.
    bchw = t.clamp_min(0.0).permute(0, 3, 1, 2)
    lab = _kornia_rgb_to_lab(bchw)
    return lab.permute(0, 2, 3, 1)


def lab_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """CIELAB D65 [B,H,W,3] → sRGB [B,H,W,3] (geclampt). kornia delinearisiert selbst."""
    bchw = t.permute(0, 3, 1, 2)
    srgb = _kornia_lab_to_rgb(bchw)
    return srgb.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def luma(t: torch.Tensor) -> torch.Tensor:
    """Rec.709-Luma aus sRGB [B,H,W,3] → [B,H,W,1]."""
    return 0.2126 * t[..., 0:1] + 0.7152 * t[..., 1:2] + 0.0722 * t[..., 2:3]


def rgb_to_hsv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB [B,H,W,3] → (H [0,360), S [0,1], V [0,1]), jeweils [B,H,W,1]."""
    r, g, b = x[..., 0:1], x[..., 1:2], x[..., 2:3]
    mx = x.amax(dim=-1, keepdim=True)
    mn = x.amin(dim=-1, keepdim=True)
    delta = mx - mn
    V = mx
    S = delta / mx.clamp_min(_EPS)
    h = torch.zeros_like(V)
    h = torch.where(
        (mx == r) & (delta > _EPS),
        (((g - b) / delta.clamp_min(_EPS)) % 6.0) * 60.0,
        torch.where(
            (mx == g) & (delta > _EPS),
            ((b - r) / delta.clamp_min(_EPS) + 2.0) * 60.0,
            ((r - g) / delta.clamp_min(_EPS) + 4.0) * 60.0,
        ),
    )
    h = h % 360.0
    return h, S, V


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """H [0,360), S [0,1], V [0,1] → RGB [B,H,W,3]."""
    C = v * s
    hp = h / 60.0
    X = C * (1.0 - torch.abs(hp % 2.0 - 1.0))
    m = v - C
    sector = hp.floor().long() % 6
    # (R1,G1,B1) pro Sektor: 0:(C,X,0) 1:(X,C,0) 2:(0,C,X) 3:(0,X,C) 4:(X,0,C) 5:(C,0,X)
    r = torch.where(sector == 0, C,
        torch.where(sector == 1, X,
        torch.where(sector == 4, X,
        torch.where(sector == 5, C, 0.0))))
    g = torch.where(sector == 0, X,
        torch.where(sector == 1, C,
        torch.where(sector == 2, C,
        torch.where(sector == 3, X, 0.0))))
    b = torch.where(sector == 2, X,
        torch.where(sector == 3, C,
        torch.where(sector == 4, C,
        torch.where(sector == 5, X, 0.0))))
    return torch.cat([r + m, g + m, b + m], dim=-1)
