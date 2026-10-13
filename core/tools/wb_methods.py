"""White-balance estimators: each `estimate(rgb_srgb [1,H,W,3]) -> illuminant RGB (linear, sum=1)."""

from __future__ import annotations

import logging
import numpy as np
import torch

from refgrade.color import srgb_to_linear, lab_to_rgb
from refgrade.whitebalance import estimate as wb_estimate

log = logging.getLogger("RefGrade")

# ---------------------------------------------------------------------------
# Illuminant helpers
# ---------------------------------------------------------------------------


def _srgb_to_linear_np(t: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear (numpy)."""
    return np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


def _normalize(rgb: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector to sum=1; fallback to equal if degenerate."""
    s = rgb.sum()
    if s < 1e-9 or not np.all(np.isfinite(rgb)):
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return rgb / s


def _lab_to_illuminant(L: float, a: float, b: float) -> np.ndarray:
    """Convert a CIELAB point to a linear-RGB illuminant (sum=1)."""
    lab = torch.tensor([[[[L, a, b]]]], dtype=torch.float32)
    srgb = lab_to_rgb(lab).reshape(-1).numpy()
    linear = _srgb_to_linear_np(srgb)
    return _normalize(linear)


def _d65_illuminant() -> np.ndarray:
    """D65 illuminant in linear sRGB, normalized to sum=1."""
    import colour
    srgb = colour.RGB_COLOURSPACES["sRGB"]
    M = colour.normalised_primary_matrix(srgb.primaries, srgb.whitepoint)
    xyz = colour.sd_to_XYZ(colour.SDS_ILLUMINANTS["D65"])
    xyz_n = xyz / xyz[1]
    return _normalize(np.linalg.inv(M) @ xyz_n)


def _blackbody_illuminant(T: float) -> np.ndarray:
    """Blackbody illuminant at T Kelvin in linear sRGB, sum=1."""
    import colour
    srgb = colour.RGB_COLOURSPACES["sRGB"]
    M = colour.normalised_primary_matrix(srgb.primaries, srgb.whitepoint)
    xyz = colour.sd_to_XYZ(colour.sd_blackbody(T))
    xyz_n = xyz / xyz[1]
    return _normalize(np.linalg.inv(M) @ xyz_n)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def est_ours(rgb: np.ndarray) -> np.ndarray:
    """Current whitebalance.estimate → illuminant = 1 / von-Kries-Faktoren (grade._temp_tint, 2026-10-13)."""
    t = torch.from_numpy(rgb).float()
    vals = wb_estimate(t)
    tp, tn = vals["temperature"], vals["tint"]
    ill = np.array([2.0 ** (-0.8 * tp), 2.0 ** (0.5 * tn), 2.0 ** (0.8 * tp)])
    return _normalize(ill)


def est_grayworld(rgb: np.ndarray, p: float = 0.0) -> np.ndarray:
    """Mean in linear RGB, excluding darkest and lightest p% by luminance."""
    linear = _srgb_to_linear_np(rgb[0])  # [H,W,3]
    Y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    flat = Y.ravel()
    n = flat.size
    k = int(n * p / 100.0)
    if k == 0:
        return _normalize(linear.reshape(-1, 3).mean(axis=0))
    lo, hi = np.partition(flat, [k, n - 1 - k])[[k, n - 1 - k]]
    mask = (flat > lo) & (flat < hi)
    return _normalize(linear.reshape(-1, 3)[mask].mean(axis=0))


def est_whitepatch(rgb: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Mean of brightest p% per channel in linear RGB; clipped (>=0.99) removed first."""
    linear = _srgb_to_linear_np(rgb[0])  # [H,W,3]
    clipped = linear >= 0.99
    out = np.zeros(3)
    for c in range(3):
        ch = linear[..., c].ravel()
        ch = ch[~clipped[..., c].ravel()]
        if ch.size == 0:
            out[c] = 1.0 / 3
            continue
        k = max(1, int(ch.size * p / 100.0))
        top = np.partition(ch, -k)[-k:]
        out[c] = top.mean()
    return _normalize(out)


def est_cheng(rgb: np.ndarray, p: float = 1.0) -> np.ndarray:
    """Cheng et al. 2014: PCA on extreme pixels projected onto normalized mean."""
    linear = _srgb_to_linear_np(rgb[0]).reshape(-1, 3)  # [N,3]
    mean = linear.mean(axis=0)
    mean_n = mean / (np.linalg.norm(mean) + 1e-12)
    proj = linear @ mean_n
    n = proj.size
    k = max(1, int(n * p / 100.0))
    idx = np.argsort(proj)
    sel = np.concatenate([idx[:k], idx[-k:]])
    pixels = linear[sel]
    cov = np.cov(pixels, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, -1]  # largest eigenvalue
    if np.sum(v) < 0:
        v = -v
    return _normalize(np.abs(v))


def est_greyedge(rgb: np.ndarray) -> np.ndarray:
    """Grey-Edge 1st order: Minkowski p=6 of per-channel gradient magnitudes, Gaussian σ=1."""
    linear = _srgb_to_linear_np(rgb[0])  # [H,W,3]
    H, W, _ = linear.shape
    weights = np.zeros(3)
    for c in range(3):
        ch = linear[..., c]
        gx = np.abs(np.diff(ch, axis=1, prepend=ch[:, :1]))
        gy = np.abs(np.diff(ch, axis=0, prepend=ch[:1, :]))
        mag = (gx ** 6 + gy ** 6) ** (1.0 / 6.0)
        # Gaussian smoothing σ=1
        kernel = _gaussian_kernel_1d(1.0, 5)
        mag = _convolve2d(mag, kernel)
        weights[c] = mag.sum()
    return _normalize(weights)


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """1-D Gaussian kernel of given sigma and radius."""
    x = np.arange(-radius, radius + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 2-D convolution (scipy-free, numpy-only)."""
    k = kernel
    pad = len(k) // 2
    H, W = img.shape
    # Horizontal
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    h = np.zeros_like(img)
    for i in range(len(k)):
        h += k[i] * padded[:, i:i + W]
    # Vertical
    padded = np.pad(h, ((pad, pad), (0, 0)), mode="edge")
    v = np.zeros_like(img)
    for i in range(len(k)):
        v += k[i] * padded[i:i + H, :]
    return v


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ESTIMATORS: dict[str, callable] = {
    "ours": est_ours,
    "grayworld_0": lambda rgb: est_grayworld(rgb, 0.0),
    "grayworld_1": lambda rgb: est_grayworld(rgb, 1.0),
    "grayworld_2": lambda rgb: est_grayworld(rgb, 2.0),
    "whitepatch_0.25": lambda rgb: est_whitepatch(rgb, 0.25),
    "whitepatch_0.5": lambda rgb: est_whitepatch(rgb, 0.5),
    "whitepatch_1": lambda rgb: est_whitepatch(rgb, 1.0),
    "whitepatch_2": lambda rgb: est_whitepatch(rgb, 2.0),
    "cheng_0.5": lambda rgb: est_cheng(rgb, 0.5),
    "cheng_1": lambda rgb: est_cheng(rgb, 1.0),
    "cheng_3.5": lambda rgb: est_cheng(rgb, 3.5),
    "cheng_5": lambda rgb: est_cheng(rgb, 5.0),
    "greyedge": est_greyedge,
}
