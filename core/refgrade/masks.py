"""Masken-Handling: Skalierung, Gewichte, Plausibilitätsprüfung."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger("RefGrade")


def to_weights(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Maske auf Bildgröße skalieren und in Gewichte umwandeln.

    Args:
        mask: [B,H,W] oder [1,H,W] oder [H,W], Werte 0..1 (1 = ignorieren).
        like: Referenz-Tensor [B,H,W,3] für Zielgröße und Batch.

    Returns:
        Gewicht [B,H,W], Werte 0..1 (1 = voll gewichtet).
    """
    target_h, target_w = like.shape[1], like.shape[2]
    batch = like.shape[0]

    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.shape[0] == 1 and batch > 1:
        m = m.expand(batch, -1, -1)

    if m.shape[1] != target_h or m.shape[2] != target_w:
        m = F.interpolate(
            m.unsqueeze(1), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

    # Plausibilität: Maske ignoriert ≥ 99 % der Pixel → verwerfen
    frac_ignored = m.mean().item()
    if frac_ignored >= 0.99:
        logger.warning(
            "Mask ignoriert %.1f%% der Pixel — Maske verworfen, Gewicht 1 überall.",
            frac_ignored * 100,
        )
        return torch.ones_like(m, dtype=torch.float32)

    return (1.0 - m).clamp(0.0, 1.0)
