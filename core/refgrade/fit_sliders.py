"""Regler-Fit: optimiert SLIDERS (ohne match) per autograd + Adam gegen die Vorlage.

Verlust: Σ über R,G,B,L,a,b der 1D-Wasserstein-2-Distanz über 256 sortierte Quantile
+ Sliced Wasserstein in Lab mit 32 Projektionen (pro Schritt neu, Seed fest)
+ 1e-3·Σ((wert−standard)/(max−min))²
"""

from __future__ import annotations

import logging
import math
import torch
import torch.nn.functional as F

from .color import rgb_to_lab
from .grade import grade
from .sliders import SLIDERS

logger = logging.getLogger("RefGrade")

_EPS = 1e-6
_NUM_PIXELS = 65_536
_SEED = 0
_NUM_SW_PROJECTIONS = 32
_NUM_QUANTILES = 256
_LR = 0.02
_MAX_STEPS = 300
_REG_WEIGHT = 1e-3
_EARLY_STOP_PATIENCE = 20
_EARLY_STOP_TOL = 0.001  # 0.1 %
_AT_TARGET = 1e-8        # Verlust darunter = schon am Ziel (Selbst-Abgleich liegt bei ~3e-13)


def _slider_names(names: list[str] | None = None) -> list[str]:
    """Reglernamen zum Fitten. Standard: alle außer match-Gruppe und wheel_blending/wheel_balance."""
    if names is not None:
        return list(names)
    exclude = {"wheel_blending", "wheel_balance", "wheel_intensity"}
    return [s.name for s in SLIDERS if s.group != "match" and s.name not in exclude]


def _slider_bounds(names: list[str] | None = None) -> dict[str, tuple[float, float, float]]:
    """name → (min, max, default) für die gefitteten Regler."""
    if names is not None:
        return {s.name: (s.min, s.max, s.default) for s in SLIDERS if s.name in names}
    return {s.name: (s.min, s.max, s.default) for s in SLIDERS if s.group != "match"}


def _subsample(img: torch.Tensor, weights: torch.Tensor | None,
               n: int = _NUM_PIXELS, seed: int = _SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Festes Subsample: [B,H,W,3] → [n,3] mit Gewichten [n]."""
    B, H, W, C = img.shape
    total = B * H * W
    flat = img.reshape(total, C)
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


def _w2_1d(v1: torch.Tensor, v2: torch.Tensor, n_q: int = _NUM_QUANTILES) -> torch.Tensor:
    """1D W2² via sortierter Quantil-Interpolation (differenzierbar)."""
    s1 = v1.sort(dim=0).values
    s2 = v2.sort(dim=0).values
    n1, n2 = s1.shape[0], s2.shape[0]
    q = (torch.arange(n_q, device=v1.device, dtype=v1.dtype) + 0.5) / n_q
    idx1 = q * (n1 - 1)
    i0 = idx1.floor().long().clamp(0, n1 - 2)
    f1 = (idx1 - i0.to(v1.dtype)).reshape(-1, *([1] * (s1.dim() - 1)))
    q1 = s1[i0] * (1.0 - f1) + s1[i0 + 1] * f1
    idx2 = q * (n2 - 1)
    i0_2 = idx2.floor().long().clamp(0, n2 - 2)
    f2 = (idx2 - i0_2.to(v2.dtype)).reshape(-1, *([1] * (s2.dim() - 1)))
    q2 = s2[i0_2] * (1.0 - f2) + s2[i0_2 + 1] * f2
    return ((q1 - q2) ** 2).mean(dim=0)


def _sliced_w2(lab1: torch.Tensor, lab2: torch.Tensor,
               n_proj: int = _NUM_SW_PROJECTIONS, seed: int = 0) -> torch.Tensor:
    """Sliced W2 in Lab: n_proj Projektionen, 1D W2 pro Projektion."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    proj = torch.randn(n_proj, 3, generator=gen)
    proj = proj / proj.norm(dim=1, keepdim=True)
    proj = proj.to(lab1.device, dtype=lab1.dtype)
    return _w2_1d(lab1 @ proj.T, lab2 @ proj.T).mean()


def _loss(
    graded: torch.Tensor,
    ref_pixels: torch.Tensor,
    params: torch.nn.Parameter,
    names: list[str],
    bounds: dict[str, tuple[float, float, float]],
    step: int,
) -> torch.Tensor:
    """Gesamtverlust: 6× W2 + Sliced W2 + Regularisierung."""
    graded_c = graded.clamp(0.0, 1.0)

    w2_sum = torch.zeros((), device=graded.device, dtype=graded.dtype)
    for ch in range(3):
        w2_sum = w2_sum + _w2_1d(graded_c[:, ch], ref_pixels[:, ch])

    n_g = graded_c.shape[0]
    n_r = ref_pixels.shape[0]
    lab_graded = rgb_to_lab(graded_c.reshape(1, 1, n_g, 3)).reshape(-1, 3)
    lab_ref = rgb_to_lab(ref_pixels.reshape(1, 1, n_r, 3)).reshape(-1, 3)
    lab_g_s = lab_graded.clone()
    lab_g_s[:, 0] = lab_g_s[:, 0] / 100.0
    lab_g_s[:, 1] = (lab_g_s[:, 1] + 128.0) / 256.0
    lab_g_s[:, 2] = (lab_g_s[:, 2] + 128.0) / 256.0
    lab_r_s = lab_ref.clone()
    lab_r_s[:, 0] = lab_r_s[:, 0] / 100.0
    lab_r_s[:, 1] = (lab_r_s[:, 1] + 128.0) / 256.0
    lab_r_s[:, 2] = (lab_r_s[:, 2] + 128.0) / 256.0
    for ch in range(3):
        w2_sum = w2_sum + _w2_1d(lab_g_s[:, ch], lab_r_s[:, ch])
    sw2 = _sliced_w2(lab_g_s, lab_r_s, seed=step)

    reg = torch.zeros((), device=graded.device, dtype=graded.dtype)
    for i, name in enumerate(names):
        lo, hi, default = bounds[name]
        norm = (params[i] - default) / (hi - lo)
        reg = reg + norm ** 2
    reg = _REG_WEIGHT * reg

    return w2_sum + sw2 + reg


def fit_sliders(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    max_steps: int = _MAX_STEPS,
    lr: float = _LR,
    progress_callback: callable | None = None,
    names: list[str] | None = None,
) -> tuple[dict[str, float], list[float]]:
    """Regler fit: optimiert die genannten Regler gegen die Vorlage.

    Args:
        image: [B,H,W,3] sRGB 0..1.
        reference: [B,H,W,3] sRGB 0..1.
        image_weights: [B,H,W] 0..1 (1 = voll gewichtet) oder None.
        reference_weights: [B,H,W] 0..1 oder None.
        max_steps: maximale Adam-Schritte.
        lr: Lernrate.
        progress_callback: optional callable(iter, coverage, swd) pro Iteration.
        names: Regler zum Fitten (Standard: alle außer match).

    Returns:
        (values: dict name→float, loss_log: list[float])
    """
    fit_names = _slider_names(names)
    bounds = _slider_bounds(names)

    img_pixels, _ = _subsample(image, image_weights)
    ref_pixels, _ = _subsample(reference, reference_weights)
    ref_pixels = ref_pixels.detach()

    defaults = torch.tensor(
        [bounds[n][2] for n in fit_names],
        device=image.device, dtype=torch.float32,
    )
    params = torch.nn.Parameter(defaults.clone())
    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=lr * 0.05
    )

    loss_log: list[float] = []
    n_pix = img_pixels.shape[0]
    initial_loss: float | None = None

    for step in range(max_steps):
        optimizer.zero_grad()

        values = {n: params[i] for i, n in enumerate(fit_names)}
        graded_4d = img_pixels.reshape(1, 1, n_pix, 3)
        graded = grade(graded_4d, values).reshape(-1, 3)

        loss = _loss(graded, ref_pixels, params, fit_names, bounds, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

        loss_val = loss.item()
        # A single NaN/inf in loss or gradient would poison every parameter (clamp_ keeps
        # NaN) and reach the UI as NaN slider values: stop on the last finite parameters.
        if not math.isfinite(loss_val) or not torch.isfinite(params.grad).all():
            logger.warning("FIT stopped at step %d: non-finite loss/gradient (loss=%s)", step, loss_val)
            break
        loss_log.append(loss_val)
        if initial_loss is None:
            initial_loss = loss_val
            if loss_val < _AT_TARGET:
                # Bild trifft die Vorlage schon: nicht bewegen. Adam normiert die Schrittweite und
                # machte aus Rundungsrauschen (~1e-7) volle Schritte weg vom Ziel (2026-10-13)
                break

        if progress_callback is not None:
            coverage = max(0.0, 1.0 - loss_val / max(initial_loss, _EPS))
            progress_callback(step, coverage, loss_val)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            for i, n in enumerate(fit_names):
                lo, hi, _ = bounds[n]
                params[i].clamp_(lo, hi)

        # Early stopping: < 0.1 % Verbesserung des Minimums über 20 Schritte.
        if step + 1 > _EARLY_STOP_PATIENCE:
            best_now = min(loss_log[step + 1 - _EARLY_STOP_PATIENCE:])
            best_prev = min(loss_log[:step + 1 - _EARLY_STOP_PATIENCE])
            if (best_prev - best_now) / max(best_prev, _EPS) < _EARLY_STOP_TOL:
                break

    result = {n: float(params[i].item()) for i, n in enumerate(fit_names)}
    return result, loss_log
