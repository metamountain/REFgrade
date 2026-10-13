"""Metriken: coverage, swd, delta_e00, structure, banding, lut_fidelity."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import colour
import cv2

from .color import rgb_to_lab, luma
from .stats import sample
from .lut import build, apply as lut_apply

_EPS = 1e-6
_NUM_HIST_BINS = 64
_NUM_SW_PROJECTIONS = 128
_SW_SEED = 42
_SW_QUANTILES = 256
_STRUCTURE_SHIFT_RANGE = 3


def _lab_scaled(lab: torch.Tensor) -> torch.Tensor:
    """Lab → skaliert: L/100, a/128, b/128 (je 0..1 Bereich)."""
    out = lab.clone()
    out[..., 0] = out[..., 0] / 100.0
    out[..., 1] = (out[..., 1] + 128.0) / 256.0
    out[..., 2] = (out[..., 2] + 128.0) / 256.0
    return out


def _weighted_hist(vals: torch.Tensor, weights: torch.Tensor,
                   lo: float, hi: float) -> np.ndarray:
    """Gewichtetes Histogramm mit numpy."""
    v = vals.cpu().numpy()
    w = weights.cpu().numpy()
    # Normalisieren in [0, 1)
    v_norm = (v - lo) / (hi - lo)
    v_norm = np.clip(v_norm, 0.0, 1.0 - 1e-9)
    bins = np.minimum((v_norm * _NUM_HIST_BINS).astype(int), _NUM_HIST_BINS - 1)
    hist = np.bincount(bins, weights=w, minlength=_NUM_HIST_BINS)
    hist = hist[:_NUM_HIST_BINS]
    total = hist.sum()
    if total < _EPS:
        return np.ones(_NUM_HIST_BINS) / _NUM_HIST_BINS
    return hist / total


def coverage(
    img: torch.Tensor,
    ref: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> float:
    """Mittlere Histogramm-Intersection (64 Bins) über R,G,B,L,a,b in %."""
    img_c = img.clamp(0.0, 1.0)
    ref_c = ref.clamp(0.0, 1.0)

    lab_img = rgb_to_lab(img_c)
    lab_ref = rgb_to_lab(ref_c)

    pix_s, w_s = sample(lab_img, image_weights)
    pix_r, w_r = sample(lab_ref, reference_weights)

    rgb_s = img_c.reshape(-1, 3)
    rgb_r = ref_c.reshape(-1, 3)
    w_rgb_s = torch.ones(rgb_s.shape[0], device=img.device)
    w_rgb_r = torch.ones(rgb_r.shape[0], device=ref.device)

    dims = [
        (rgb_s[:, 0], rgb_r[:, 0], w_rgb_s, w_rgb_r, 0.0, 1.0),
        (rgb_s[:, 1], rgb_r[:, 1], w_rgb_s, w_rgb_r, 0.0, 1.0),
        (rgb_s[:, 2], rgb_r[:, 2], w_rgb_s, w_rgb_r, 0.0, 1.0),
        (pix_s[:, 0], pix_r[:, 0], w_s, w_r, 0.0, 100.0),
        (pix_s[:, 1], pix_r[:, 1], w_s, w_r, -128.0, 128.0),
        (pix_s[:, 2], pix_r[:, 2], w_s, w_r, -128.0, 128.0),
    ]

    intersections = []
    for ds, dr, ws, wr, lo, hi in dims:
        hs = _weighted_hist(ds, ws, lo, hi)
        hr = _weighted_hist(dr, wr, lo, hi)
        intersections.append(float(np.sum(np.minimum(hs, hr))))
    return float(np.mean(intersections) * 100.0)


def _fast_quantiles(sorted_vals: torch.Tensor, cum_w: torch.Tensor,
                    q_positions: torch.Tensor) -> torch.Tensor:
    """Vektorisierte Quantile aus sortierten Werten und kumulativen Gewichten."""
    total_w = cum_w[-1].clamp_min(_EPS)
    cdf = cum_w / total_w
    # searchsorted für alle q-Positionen gleichzeitig
    idx = torch.searchsorted(cdf, q_positions)
    idx = idx.clamp(1, len(sorted_vals) - 1)
    w0 = cdf[idx - 1]
    w1 = cdf[idx]
    frac = (q_positions - w0) / (w1 - w0).clamp_min(_EPS)
    return sorted_vals[idx - 1] + frac * (sorted_vals[idx] - sorted_vals[idx - 1])


def swd(
    img: torch.Tensor,
    ref: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> float:
    """Sliced Wasserstein in skaliertem Lab, 128 feste Projektionen."""
    lab_img = _lab_scaled(rgb_to_lab(img))
    lab_ref = _lab_scaled(rgb_to_lab(ref))

    pix_s, w_s = sample(lab_img, image_weights)
    pix_r, w_r = sample(lab_ref, reference_weights)

    # Feste Projektionsvektoren (deterministisch)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SW_SEED)
    proj = torch.randn(_NUM_SW_PROJECTIONS, 3, generator=gen)
    proj = proj / proj.norm(dim=1, keepdim=True)
    proj = proj.to(pix_s.device, dtype=torch.float64)

    # Projektieren: [N, 3] @ [3, 128] → [N, 128]
    proj_s = pix_s.double() @ proj.T
    proj_r = pix_r.double() @ proj.T

    # Quantil-Positionen
    n_q = _SW_QUANTILES
    q_positions = (torch.arange(n_q, dtype=torch.float64, device=proj_s.device) + 0.5) / n_q

    w_s_d = w_s.double()
    w_r_d = w_r.double()

    total = 0.0
    for j in range(_NUM_SW_PROJECTIONS):
        # Sortieren
        sorted_s, idx_s = proj_s[:, j].sort()
        sorted_r, idx_r = proj_r[:, j].sort()
        cum_s = torch.cumsum(w_s_d[idx_s], dim=0)
        cum_r = torch.cumsum(w_r_d[idx_r], dim=0)

        qs = _fast_quantiles(sorted_s, cum_s, q_positions)
        qr = _fast_quantiles(sorted_r, cum_r, q_positions)
        total += (qs - qr).abs().mean().item()

    return total / _NUM_SW_PROJECTIONS


def delta_e00(
    result: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[float, float]:
    """CIE 2000: Mittel und 95. Perzentil.

    Returns:
        (mean, p95)
    """
    lab_res = rgb_to_lab(result).cpu().numpy().reshape(-1, 3)
    lab_tru = rgb_to_lab(truth).cpu().numpy().reshape(-1, 3)

    if mask is not None:
        m = mask.cpu().numpy().reshape(-1)
        m = m > 0.5
        lab_res = lab_res[m]
        lab_tru = lab_tru[m]

    de = colour.delta_E(lab_res, lab_tru, method="CIE 2000")
    return float(np.mean(de)), float(np.percentile(de, 95))


def _sobel_magnitude(luma_map: np.ndarray) -> np.ndarray:
    """Sobel-Gradientenbetrag einer 2D-Luma-Karte."""
    gx = cv2.Sobel(luma_map, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(luma_map, cv2.CV_64F, 0, 1)
    return np.sqrt(gx**2 + gy**2)


def structure(
    img: torch.Tensor,
    result: torch.Tensor,
) -> float:
    """Korrelation der Sobel-Gradientenbeträge von L (Rec.709-Luma),
    robust gegen kleine Verschiebungen: Maximum über ±_STRUCTURE_SHIFT_RANGE px."""
    l_img = luma(img).cpu().numpy()  # [B,H,W,1]
    l_res = luma(result).cpu().numpy()

    b = img.shape[0]
    correlations = []
    for i in range(b):
        mag1 = _sobel_magnitude(l_img[i, :, :, 0])
        mag2 = _sobel_magnitude(l_res[i, :, :, 0])

        c1 = mag1 - mag1.mean()
        c2 = mag2 - mag2.mean()
        n1 = np.sqrt((c1**2).sum())
        n2 = np.sqrt((c2**2).sum())
        if n1 < _EPS or n2 < _EPS:
            correlations.append(1.0)
            continue
        c1 = c1 / n1
        c2 = c2 / n2

        best = -2.0
        for dy in range(-_STRUCTURE_SHIFT_RANGE, _STRUCTURE_SHIFT_RANGE + 1):
            for dx in range(-_STRUCTURE_SHIFT_RANGE, _STRUCTURE_SHIFT_RANGE + 1):
                best = max(best, float((c1 * np.roll(c2, (dy, dx))).sum()))
        correlations.append(best)

    return float(np.mean(correlations))


def banding(
    img: torch.Tensor,
    result: torch.Tensor,
    num_bins: int = 64,
) -> float:
    """Anteil im Ergebnis leerer L-Bins, die im Eingang belegt waren."""
    l_img = luma(img).cpu().numpy().reshape(-1)
    l_res = luma(result).cpu().numpy().reshape(-1)

    hist_in, _ = np.histogram(l_img, bins=num_bins, range=(0, 1))
    hist_out, _ = np.histogram(l_res, bins=num_bins, range=(0, 1))

    occupied_in = hist_in > 0
    empty_out = hist_out == 0

    if occupied_in.sum() == 0:
        return 0.0

    return float((occupied_in & empty_out).sum() / occupied_in.sum())


def lut_fidelity(
    img: torch.Tensor,
    fn: callable,
    lut_size: int = 33,
) -> tuple[float, float]:
    """Mittlere/max. Differenz: direkt vs. über LUT.

    Returns:
        (mean_diff, max_diff)
    """
    direct = fn(img)
    lut = build(fn, size=lut_size, device=img.device)
    via_lut = lut_apply(img, lut)

    diff = (direct - via_lut).abs()
    return float(diff.mean().item()), float(diff.max().item())
