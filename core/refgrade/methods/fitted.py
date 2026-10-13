"""Registrierung: slider_fit und slider_fit_idt (stable=False)."""

from __future__ import annotations

import torch

from ..methods import register
from ..fit_sliders import fit_sliders
from ..grade import grade
from ..idt import idt, apply_idt
from ..color import rgb_to_lab, lab_to_rgb


@register("slider_fit", stable=False)
def fit_slider_fit(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> callable:
    """Regler-Fit: optimiert SLIDERS gegen die Vorlage."""
    values, _ = fit_sliders(image, reference, image_weights, reference_weights)

    def pipeline(srgb: torch.Tensor) -> torch.Tensor:
        return grade(srgb, values)

    pipeline.slider_values = values
    return pipeline


@register("slider_fit_idt", stable=False)
def fit_slider_fit_idt(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
    residual_strength: float = 0.0,
) -> callable:
    """Regler-Fit + IDT: erst Regler, dann IDT auf dem Ergebnis."""
    values, _ = fit_sliders(image, reference, image_weights, reference_weights)

    # IDT auf dem gegradeten Bild
    graded = grade(image, values)
    steps = idt(graded, reference, image_weights, reference_weights)

    def pipeline(srgb: torch.Tensor) -> torch.Tensor:
        g = grade(srgb, values)
        if steps:
            g = apply_idt(g, steps, residual_strength)
        return g

    pipeline.slider_values = values
    pipeline.idt_steps = steps
    return pipeline
