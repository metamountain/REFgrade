"""IDT (Iterative Dichromatic Transformation, Pitié et al. 2005/2007).

Auf dem Ergebnis des Regler-Fits, in skaliertem Lab: pro Iteration zufällige
orthonormale Rotation, pro Achse 1D-Quantil-Mapping, zurückrotieren.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import rgb_to_lab, lab_to_rgb
from .stats import sample

_EPS = 1e-6
_NUM_QUANTILES = 512
_SMOOTH = 0.2
_MAX_ITERS = 50
_SW_THRESHOLD = 1e-3
_EARLY_STOP_PATIENCE = 3
_EARLY_STOP_TOL = 0.005  # 0.5 %
_SEED_BASE = 0


def _lab_scaled(lab: torch.Tensor) -> torch.Tensor:
    """Lab → skaliert: L/100, a/128, b/128 (je 0..1)."""
    out = lab.clone()
    out[..., 0] = out[..., 0] / 100.0
    out[..., 1] = (out[..., 1] + 128.0) / 256.0
    out[..., 2] = (out[..., 2] + 128.0) / 256.0
    return out


def _lab_unscaled(scaled: torch.Tensor) -> torch.Tensor:
    """Skaliert → Lab."""
    out = scaled.clone()
    out[..., 0] = out[..., 0] * 100.0
    out[..., 1] = out[..., 1] * 256.0 - 128.0
    out[..., 2] = out[..., 2] * 256.0 - 128.0
    return out


def _random_rotation(seed: int, device: torch.device) -> torch.Tensor:
    """Zufällige orthonormale 3×3-Matrix (QR einer Gauß-Matrix)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    m = torch.randn(3, 3, generator=gen)
    q, r = torch.linalg.qr(m)
    d = torch.diagonal(r, dim1=0, dim2=1)
    ph = d.sign()
    q = q * ph.unsqueeze(0)
    return q.to(device)


def _smooth_monotone(curve: torch.Tensor, alpha: float = _SMOOTH) -> torch.Tensor:
    """Moving-Average-Glättung + Monotonie."""
    if alpha <= 0:
        return curve
    k = 5
    pad = k // 2
    c = curve.unsqueeze(0).unsqueeze(0)
    cp = F.pad(c, (pad, pad), mode="replicate")
    kern = torch.ones(1, 1, k, device=curve.device, dtype=curve.dtype) / k
    smoothed = F.conv1d(cp, kern).squeeze()
    out = alpha * smoothed + (1.0 - alpha) * curve
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _quantiles_1d(values: torch.Tensor, n_q: int = _NUM_QUANTILES) -> torch.Tensor:
    """n_q Quantile aus 1D-Werten."""
    q_pos = (torch.arange(n_q, device=values.device, dtype=values.dtype) + 0.5) / n_q
    return torch.quantile(values, q_pos)


def _apply_curve(x: torch.Tensor, src_q: torch.Tensor, dst_q: torch.Tensor) -> torch.Tensor:
    """Stückweise lineare Interpolation: x → mapped via (src_q, dst_q)."""
    idx = torch.searchsorted(src_q, x)
    idx = idx.clamp(1, len(src_q) - 1)
    x0 = src_q[idx - 1]
    x1 = src_q[idx]
    frac = (x - x0) / (x1 - x0).clamp_min(_EPS)
    return dst_q[idx - 1] + frac * (dst_q[idx] - dst_q[idx - 1])


def _swd_check(lab1: torch.Tensor, lab2: torch.Tensor, seed: int = 0) -> float:
    """Schnelle SWD-Approximation für Abbruchprüfung."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    proj = torch.randn(16, 3, generator=gen)
    proj = proj / proj.norm(dim=1, keepdim=True)
    proj = proj.to(lab1.device, dtype=torch.float64)

    p1 = lab1.double() @ proj.T
    p2 = lab2.double() @ proj.T
    s1 = p1.sort(dim=0).values
    s2 = p2.sort(dim=0).values
    n = min(len(s1), len(s2))
    return float((s1[:n] - s2[:n]).abs().mean().item())


def idt(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    max_iters: int = _MAX_ITERS,
) -> list[dict]:
    """IDT: Iterative Dichromatic Transformation.

    Args:
        image: [B,H,W,3] sRGB 0..1 (Ergebnis des Regler-Fits).
        reference: [B,H,W,3] sRGB 0..1.
        image_weights: [B,H,W] 0..1 oder None.
        reference_weights: [B,H,W] 0..1 oder None.
        max_iters: maximale Iterationen.

    Returns:
        Liste von dict: {"rotation": 3×3 list, "curves": [{"src":[512], "dst":[512]}]×3}
    """
    lab_img = rgb_to_lab(image)
    lab_ref = rgb_to_lab(reference)

    pix_img, _ = sample(lab_img, image_weights)
    pix_ref, _ = sample(lab_ref, reference_weights)

    scaled_img = _lab_scaled(pix_img)
    scaled_ref = _lab_scaled(pix_ref)

    steps: list[dict] = []
    current = scaled_img.clone()
    prev_swd = float("inf")
    patience = 0

    for i in range(max_iters):
        R = _random_rotation(_SEED_BASE + i, current.device)

        # Bild und Vorlage gemeinsam rotieren (IDT: Mapping im rotierten System)
        rotated = current @ R.T  # [N,3]
        rotated_ref = scaled_ref @ R.T  # [N,3]

        # Pro Achse: Quantil-Mapping
        curves: list[dict] = []
        mapped = torch.empty_like(rotated)
        for ch in range(3):
            src_q = _quantiles_1d(rotated[:, ch])
            dst_q = _quantiles_1d(rotated_ref[:, ch])
            dst_q = _smooth_monotone(dst_q)
            mapped[:, ch] = _apply_curve(rotated[:, ch], src_q, dst_q)
            curves.append({
                "src": src_q.tolist(),
                "dst": dst_q.tolist(),
            })

        # Zurückrotieren
        current = mapped @ R

        steps.append({
            "rotation": R.tolist(),
            "curves": curves,
        })

        # Abbruch
        swd = _swd_check(current, scaled_ref, seed=i)
        if swd < _SW_THRESHOLD:
            break
        if swd < prev_swd * (1.0 - _EARLY_STOP_TOL):
            prev_swd = swd
            patience = 0
        else:
            patience += 1
            if patience >= _EARLY_STOP_PATIENCE:
                break

    return steps


def apply_idt(
    image: torch.Tensor,
    steps: list[dict],
    residual_strength: float = 0.0,
) -> torch.Tensor:
    """IDT-Schritte auf ein Bild anwenden.

    Args:
        image: [B,H,W,3] sRGB 0..1.
        steps: Ausgabe von idt().
        residual_strength: 0..1. 0 = nur IDT, 1 = nur Original.

    Returns:
        [B,H,W,3] sRGB 0..1.
    """
    if not steps:
        return image

    lab = rgb_to_lab(image)
    scaled = _lab_scaled(lab)
    orig_shape = scaled.shape
    flat = scaled.reshape(-1, 3)

    for step in steps:
        R = torch.tensor(step["rotation"], device=flat.device, dtype=flat.dtype)
        rotated = flat @ R.T

        for ch in range(3):
            src_q = torch.tensor(step["curves"][ch]["src"], device=flat.device, dtype=flat.dtype)
            dst_q = torch.tensor(step["curves"][ch]["dst"], device=flat.device, dtype=flat.dtype)
            rotated[:, ch] = _apply_curve(rotated[:, ch], src_q, dst_q)

        flat = rotated @ R

    result = _lab_unscaled(flat.reshape(orig_shape))
    out = lab_to_rgb(result)

    if residual_strength > 0:
        out = image * residual_strength + out * (1.0 - residual_strength)

    return out.clamp(0.0, 1.0)
