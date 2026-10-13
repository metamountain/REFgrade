"""Gewichtete Pixel-Statistik: Mittel, Std, Kovarianz, Quantile, Subsampling."""

from __future__ import annotations

import torch

_EPS = 1e-6


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Gewichtetes Mittel über [N,3] mit Gewichten [N] → [3]."""
    w = weights.unsqueeze(1)  # [N,1]
    return (values * w).sum(dim=0) / w.sum().clamp_min(_EPS)


def weighted_std(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Gewichtete Standardabweichung über [N,3] mit Gewichten [N] → [3]."""
    mu = weighted_mean(values, weights)
    w = weights.unsqueeze(1)
    w_sum = w.sum().clamp_min(_EPS)
    diff = values - mu
    var = (diff * diff * w).sum(dim=0) / w_sum
    return var.clamp_min(_EPS).sqrt()


def weighted_covariance(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Gewichtete Kovarianzmatrix [3,3] (float64)."""
    values = values.to(torch.float64)
    weights = weights.to(torch.float64)
    mu = weighted_mean(values, weights)
    w = weights.unsqueeze(1)
    w_sum = w.sum().clamp_min(_EPS)
    diff = values - mu
    cov = (diff.T * w.squeeze(1)).mm(diff) / w_sum
    return cov


def weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Gewichtete Quantile.

    Args:
        values: [N] 1D.
        weights: [N] 1D.
        q: [K] Quantil-Positionen in (0,1].

    Returns:
        [K] Quantil-Werte.
    """
    weights = weights.to(values.device)
    q = q.to(values.device)
    # Sortieren
    sorted_vals, idx = values.sort()
    sorted_w = weights[idx]
    cum_w = torch.cumsum(sorted_w, dim=0)
    total_w = cum_w[-1].clamp_min(_EPS)
    # Normalisierte kumulative Gewichte
    cdf = cum_w / total_w
    # Für jedes q die Position finden und linear interpolieren
    result = torch.empty_like(q)
    for i, qi in enumerate(q):
        pos = torch.searchsorted(cdf, qi)
        pos = pos.clamp(1, len(sorted_vals) - 1)
        w0 = cdf[pos - 1]
        w1 = cdf[pos]
        frac = (qi - w0) / (w1 - w0).clamp_min(_EPS)
        result[i] = sorted_vals[pos - 1] + frac * (sorted_vals[pos] - sorted_vals[pos - 1])
    return result


def sample(
    img: torch.Tensor,
    weights: torch.Tensor | None,
    n: int = 1_000_000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel aus Bild sub-sampeln.

    Args:
        img: [B,H,W,3] float.
        weights: [B,H,W] float oder None (→ alle 1).
        n: maximale Anzahl Pixel.
        seed: fester Seed für Determinismus.

    Returns:
        (pixels [N,3], weights [N])
    """
    b, h, w, c = img.shape
    total = b * h * w

    flat = img.reshape(total, c)
    if weights is None:
        w_flat = torch.ones(total, device=img.device, dtype=torch.float32)
    else:
        w_flat = weights.reshape(total)

    if total <= n:
        return flat, w_flat

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    indices = torch.randperm(total, generator=gen)[:n]
    return flat[indices], w_flat[indices]
