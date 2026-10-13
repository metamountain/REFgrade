"""Klassische Methoden registrieren: reinhard, mkl, histogram, zonal."""

from __future__ import annotations

import torch

from ..methods import register
from ..transfer import reinhard as _reinhard, mkl as _mkl, histogram as _histogram
from ..zonal import analyze as _zonal_analyze, apply_params as _zonal_apply


@register("reinhard", stable=True)
def fit_reinhard(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> callable:
    """Reinhard-Transfer (stabil)."""
    return _reinhard(image, reference, image_weights, reference_weights)


@register("mkl", stable=True)
def fit_mkl(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> callable:
    """MKL-Transfer (stabil)."""
    return _mkl(image, reference, image_weights, reference_weights)


@register("histogram", stable=True)
def fit_histogram(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> callable:
    """Histogramm-Transfer (stabil)."""
    return _histogram(image, reference, image_weights, reference_weights)


@register("zonal", stable=True)
def fit_zonal(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> callable:
    """Zonal-Abgleich (stabil)."""
    params = _zonal_analyze(image, reference, image_weights, reference_weights)
    lab_fn = _zonal_apply(params, light_strength=1.0, color_strength=1.0)

    from ..color import rgb_to_lab, lab_to_rgb

    def pipeline(srgb: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(srgb)
        lab_out = lab_fn(lab)
        return lab_to_rgb(lab_out)

    pipeline.lab_transform = lab_fn
    return pipeline
