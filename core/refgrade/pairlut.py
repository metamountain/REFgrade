"""PAIR: LUT aus einem pixelgleichen Vorher/Nachher-Paar (Splat + Jacobi)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PREBLUR_SIGMA = 1.0
_MAX_PIXELS = 1_000_000
_SEED = 0
_EPS = 1e-6


def _gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 2D Gaussian blur on [B,H,W,3]."""
    if sigma < 0.1:
        return img
    ksize = int(2 * 3 * sigma) | 1
    if ksize < 3:
        ksize = 3
    x = torch.arange(ksize, dtype=torch.float64, device=img.device) - ksize // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = (kernel / kernel.sum()).float()
    img_4d = img.permute(0, 3, 1, 2)  # [B,3,H,W]
    # Blur along W (dim=3): weight [3,1,1,ksize] for depthwise
    padded = F.pad(img_4d, (ksize // 2, ksize // 2, 0, 0), mode="replicate")
    kw = kernel.view(1, 1, 1, ksize).expand(3, 1, 1, ksize).contiguous()
    img_4d = F.conv2d(padded, kw, groups=3)
    # Blur along H (dim=2): weight [3,1,ksize,1] for depthwise
    padded = F.pad(img_4d, (0, 0, ksize // 2, ksize // 2), mode="replicate")
    kh = kernel.view(1, 1, ksize, 1).expand(3, 1, ksize, 1).contiguous()
    img_4d = F.conv2d(padded, kh, groups=3)
    return img_4d.permute(0, 2, 3, 1)


def fit(
    before: torch.Tensor,
    after: torch.Tensor,
    size: int = 33,
    lam: float = 5.0,
    iters: int = 300,
) -> torch.Tensor:
    """Fit a 3D LUT from an aligned before/after pair.

    Args:
        before: [1,H,W,3] sRGB 0..1.
        after: [1,H,W,3] sRGB 0..1, same spatial size as before.
        size: LUT cube edge length.
        lam: Jacobi confidence parameter (higher = more smoothing in empty cells).
        iters: Number of Jacobi diffusion steps.

    Returns:
        Tensor [size,size,size,3] float32, index [b][g][r], values 0..1.

    Raises:
        ValueError: if before and after have different spatial sizes.
    """
    if before.shape != after.shape:
        raise ValueError(
            f"PAIR needs same-size images: {tuple(before.shape)} vs {tuple(after.shape)}"
        )

    # Pre-blur against JPEG noise
    before_b = _gaussian_blur(before, PREBLUR_SIGMA)
    after_b = _gaussian_blur(after, PREBLUR_SIGMA)

    # Flatten and subsample
    B, H, W, C = before_b.shape
    total = B * H * W
    x = before_b.reshape(total, C).double()
    y = after_b.reshape(total, C).double()

    if total > _MAX_PIXELS:
        gen = torch.Generator().manual_seed(_SEED)
        idx = torch.randperm(total, generator=gen, device=x.device)[:_MAX_PIXELS]
        x, y = x[idx], y[idx]

    # Trilinear splat
    N = size
    g = x * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    f = g - i0

    num = torch.zeros(N ** 3, 3, dtype=torch.float64, device=x.device)
    den = torch.zeros(N ** 3, dtype=torch.float64, device=x.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * \
                    (f[:, 1] if dy else 1 - f[:, 1]) * \
                    (f[:, 2] if dz else 1 - f[:, 2])
                idx3 = ((i0[:, 0] + dx) * N + (i0[:, 1] + dy)) * N + (i0[:, 2] + dz)
                num.index_add_(0, idx3, w[:, None] * y)
                den.index_add_(0, idx3, w)

    # Grid in [r][g][b] order (matching the splat axis convention)
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64, device=x.device)] * 3,
                       indexing="ij"),
        -1,
    ).reshape(-1, 3)

    # Residual and confidence
    r = torch.where(
        den[:, None] > 0,
        num / den.clamp_min(_EPS)[:, None] - grid,
        torch.zeros_like(grid),
    )
    conf = den / (den + lam)

    # Jacobi diffusion: r ← c·r₀ + (1−c)·mean(6 neighbors), replicate boundary
    r = r.reshape(N, N, N, 3)
    c = conf.reshape(N, N, N, 1)
    target = r.clone()

    for _ in range(iters):
        p = F.pad(r.permute(3, 0, 1, 2)[None], (1,) * 6, mode="replicate")[0].permute(1, 2, 3, 0)
        nb = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
              p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
              p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / 6
        r = c * target + (1 - c) * nb

    table = (grid.reshape(N, N, N, 3) + r).clamp(0, 1)

    # Permute from [r][g][b] to [b][g][r] (LUT convention)
    table = table.permute(2, 1, 0, 3).contiguous()
    return table.float()


def split_pair(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a side-by-side image into left (before) and right (after) halves.

    Args:
        img: [1,H,W,3] sRGB 0..1.

    Returns:
        (before, after), each [1,H,W//2,3]. For odd W the middle column is dropped.
    """
    W = img.shape[2]
    half = W // 2
    before = img[:, :, :half, :]
    after = img[:, :, W - half:, :]
    return before, after


def as_lut(table: torch.Tensor, title: str = "PAIR") -> dict:
    """Wrap a [N,N,N,3] table tensor into the LUT dict format used by lut.apply/read_cube.

    Args:
        table: [N,N,N,3] float32, index [b][g][r], values 0..1.
        title: Title for cube export.

    Returns:
        dict(size, table, title).
    """
    N = table.shape[0]
    return {"size": N, "table": table, "title": title}
