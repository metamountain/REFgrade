"""Farbtransfer: Reinhard, MKL, Histogramm — jeweils als Pro-Pixel-Callable."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import rgb_to_lab, lab_to_rgb
from .stats import sample, weighted_mean, weighted_std, weighted_covariance, weighted_quantile

_EPS = 1e-6
_NUM_QUANTILES = 1024
_SMOOTH_BASE = 128.0


def _smooth_quantiles(q: torch.Tensor, n_pixels: int) -> torch.Tensor:
    """Sortierte Quantil-Kurve glätten (Moving Average + Monotonie).

    Reduziert das Rauschen der Quantil-Schätzung bei kleinen Stichproben, damit
    die stückweise lineare Kurve LUT-fähig bleibt. Fenster wächst mit sqrt(n_q/N).
    """
    n = q.shape[0]
    if n < 3 or n_pixels < 1:
        return q
    k = int(max(3, min(n, round(_SMOOTH_BASE * (n / n_pixels) ** 0.5))))
    if k <= 1:
        return q
    # Asymmetrisch polstern (links (k-1)//2, rechts k//2): Gesamtpolsterung k-1,
    # damit F.conv1d immer exakt n Elemente liefert (bei geradem k wäre
    # symmetrisches k//2 auf beiden Seiten um 1 zu lang).
    pad_left = (k - 1) // 2
    pad_right = k // 2
    q2 = q.unsqueeze(0).unsqueeze(0)
    qp2 = F.pad(q2, (pad_left, pad_right), mode="replicate")
    kern = torch.ones(1, 1, k, device=q.device, dtype=q.dtype) / k
    out = F.conv1d(qp2, kern).squeeze()
    return torch.cummax(out, dim=0).values


def _prepare(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None,
    reference_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Konvertiere nach Lab und sampelt Pixel.

    Returns:
        (lab_img, lab_ref, img_weights_flat, ref_weights_flat)
    """
    lab_img = rgb_to_lab(image)
    lab_ref = rgb_to_lab(reference)

    pixels_s, w_s = sample(lab_img, image_weights)
    pixels_r, w_r = sample(lab_ref, reference_weights)
    return lab_img, lab_ref, pixels_s, w_s, pixels_r, w_r


def _make_pipeline(
    lab_transform: torch.Tensor | callable,
    strength: float,
    device: torch.device,
):
    """Erstelle das Full-Pipeline-Callable (sRGB → sRGB)."""

    def pipeline(srgb: torch.Tensor) -> torch.Tensor:
        if strength == 0:
            return srgb
        lab = rgb_to_lab(srgb)
        lab_out = lab_transform(lab)
        return lab_to_rgb(lab_out)

    pipeline.lab_transform = lab_transform
    return pipeline


def reinhard(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    strength: float = 1.0,
) -> callable:
    """Reinhard-Transfer pro Lab-Kanal: Mittel und Std angleichen."""
    _, _, pixels_s, w_s, pixels_r, w_r = _prepare(
        image, reference, image_weights, reference_weights
    )

    mu_s = weighted_mean(pixels_s, w_s)
    std_s = weighted_std(pixels_s, w_s)
    mu_r = weighted_mean(pixels_r, w_r)
    std_r = weighted_std(pixels_r, w_r)

    device = image.device

    def lab_transform(lab: torch.Tensor) -> torch.Tensor:
        # Pro Kanal: y = (x - mu_s) / max(std_s, eps) * std_r + mu_r
        scaled = (lab - mu_s) / std_s.clamp_min(_EPS) * std_r + mu_r
        if strength < 1.0:
            scaled = lab + strength * (scaled - lab)
        return scaled

    return _make_pipeline(lab_transform, strength, device)


def mkl(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    strength: float = 1.0,
) -> callable:
    """MKL-Transfer: 3×3-Matrix in Lab, Kovarianz angleichen."""
    _, _, pixels_s, w_s, pixels_r, w_r = _prepare(
        image, reference, image_weights, reference_weights
    )

    mu_s = weighted_mean(pixels_s, w_s)
    mu_r = weighted_mean(pixels_r, w_r)
    cov_s = weighted_covariance(pixels_s, w_s)
    cov_r = weighted_covariance(pixels_r, w_r)

    device = image.device

    # Matrixwurzeln per eigh (float64)
    eigval_s, eigvec_s = torch.linalg.eigh(cov_s)
    eigval_s = eigval_s.clamp_min(_EPS)
    cov_s_sqrt = eigvec_s @ torch.diag(eigval_s.sqrt()) @ eigvec_s.T
    cov_s_inv_sqrt = eigvec_s @ torch.diag(1.0 / eigval_s.sqrt()) @ eigvec_s.T

    inner = cov_s_sqrt @ cov_r @ cov_s_sqrt
    eigval_i, eigvec_i = torch.linalg.eigh(inner)
    eigval_i = eigval_i.clamp_min(_EPS)
    inner_sqrt = eigvec_i @ torch.diag(eigval_i.sqrt()) @ eigvec_i.T

    A = cov_s_inv_sqrt @ inner_sqrt @ cov_s_inv_sqrt
    A = A.to(torch.float32)

    def lab_transform(lab: torch.Tensor) -> torch.Tensor:
        # y = A @ (x - mu_s) + mu_r
        centered = lab - mu_s.to(torch.float32)
        transformed = centered @ A.T + mu_r.to(torch.float32)
        if strength < 1.0:
            transformed = lab + strength * (transformed - lab)
        return transformed

    return _make_pipeline(lab_transform, strength, device)


def histogram(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    strength: float = 1.0,
) -> callable:
    """Histogramm-Transfer pro Lab-Kanal: Quantil-Mapping, 1024 Stufen."""
    _, _, pixels_s, w_s, pixels_r, w_r = _prepare(
        image, reference, image_weights, reference_weights
    )

    device = image.device
    n_q = _NUM_QUANTILES
    # Quantil-Positionen: (i + 0.5) / n_q für i = 0..n_q-1
    q_positions = (torch.arange(n_q, device=device, dtype=torch.float64) + 0.5) / n_q

    # Pro Kanal (3 Lab-Kanäle)
    n_s = pixels_s.shape[0]
    n_r = pixels_r.shape[0]
    src_quantiles = []
    ref_quantiles = []
    for ch in range(3):
        sq = weighted_quantile(pixels_s[:, ch].double(), w_s.double(), q_positions)
        rq = weighted_quantile(pixels_r[:, ch].double(), w_r.double(), q_positions)
        src_quantiles.append(_smooth_quantiles(sq, n_s))
        ref_quantiles.append(_smooth_quantiles(rq, n_r))

    src_q = torch.stack(src_quantiles, dim=1)  # [1024, 3]
    ref_q = torch.stack(ref_quantiles, dim=1)  # [1024, 3]
    src_q = src_q.to(torch.float32)
    ref_q = ref_q.to(torch.float32)

    def lab_transform(lab: torch.Tensor) -> torch.Tensor:
        # Pro Pixel, pro Kanal: in src_q lokalisieren, in ref_q interpolieren
        orig_shape = lab.shape
        flat = lab.reshape(-1, 3)

        result = torch.empty_like(flat)
        for ch in range(3):
            s = src_q[:, ch]  # [1024] sortiert aufsteigend
            r = ref_q[:, ch]  # [1024]
            x = flat[:, ch]

            # Position in src_q finden
            idx = torch.searchsorted(s.contiguous(), x.contiguous())
            idx = idx.clamp(1, len(s) - 1)

            # Stückweise linear: innerhalb interpolieren, außerhalb mit der
            # Steigung des Randsegments extrapolieren (Identität bei Bild==Vorlage).
            x0 = s[idx - 1]
            x1 = s[idx]
            frac = (x - x0) / (x1 - x0).clamp_min(_EPS)
            result[:, ch] = r[idx - 1] + frac * (r[idx] - r[idx - 1])

        transformed = result.reshape(orig_shape)
        if strength < 1.0:
            transformed = lab + strength * (transformed - lab)
        return transformed

    return _make_pipeline(lab_transform, strength, device)
