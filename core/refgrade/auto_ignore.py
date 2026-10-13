"""Auto-Ignore: Vorschlag für image_ignore_mask aus Lab-Chroma und Farbton-Bins."""

from __future__ import annotations

import cv2
import torch
import torch.nn.functional as F

from .color import rgb_to_lab

_EPS = 1e-6
_NUM_HUE_BINS = 24


def suggest(
    image: torch.Tensor,
    reference: torch.Tensor,
    chroma_percentile: float = 99.0,
    min_area: float = 0.02,
    feather_px: int = 8,
) -> torch.Tensor:
    """Vorschlag für image_ignore_mask.

    Markiert Bildpixel, deren Lab-Chroma über dem Perzentil der Vorlage liegt
    UND deren Farbton-Bin (24 Bins) in der Vorlage < 1 % Anteil hat.

    Args:
        image: [B,H,W,3] sRGB 0..1.
        reference: [B,H,W,3] sRGB 0..1.
        chroma_percentile: Schwellwert für Chroma (Perzentil der Vorlage).
        min_area: minimale Insel-Fläche (Anteil des Bildes).
        feather_px: Weichzeichnen in Pixeln.

    Returns:
        MASK [B,H,W] 0..1 (1 = ignorieren).
    """
    lab_img = rgb_to_lab(image)
    lab_ref = rgb_to_lab(reference)

    # Chroma
    chroma_img = torch.sqrt(lab_img[..., 1]**2 + lab_img[..., 2]**2)
    chroma_ref = torch.sqrt(lab_ref[..., 1]**2 + lab_ref[..., 2]**2)

    # Chroma-Schwelle: Perzentil der Vorlage
    chroma_ref_flat = chroma_ref.reshape(-1)
    threshold = float(torch.quantile(chroma_ref_flat, chroma_percentile / 100.0).item())

    # Farbton-Bins der Vorlage (nur Pixel mit Chroma > 5, da Hue von Graustufen zufällig ist)
    chroma_threshold_ref = 5.0
    hue_ref = torch.atan2(lab_ref[..., 2], lab_ref[..., 1])
    hue_ref_norm = (hue_ref / (2 * 3.14159265) + 0.5) % 1.0
    bin_ref = (hue_ref_norm * _NUM_HUE_BINS).long().clamp(0, _NUM_HUE_BINS - 1)

    # Nur hoch-Chroma-Pixel zählen
    high_chroma_ref = chroma_ref > chroma_threshold_ref
    bin_counts = torch.bincount(
        bin_ref[high_chroma_ref].flatten(), minlength=_NUM_HUE_BINS
    ).float()
    bin_fractions = bin_counts / bin_counts.sum().clamp_min(_EPS)

    # Bild: Chroma über Schwelle UND Bin in Vorlage < 1 %
    hue_img = torch.atan2(lab_img[..., 2], lab_img[..., 1])
    hue_img_norm = (hue_img / (2 * 3.14159265) + 0.5) % 1.0
    bin_img = (hue_img_norm * _NUM_HUE_BINS).long().clamp(0, _NUM_HUE_BINS - 1)

    chroma_exceeds = chroma_img > threshold
    bin_rare = bin_fractions[bin_img] < 0.01

    mask = (chroma_exceeds & bin_rare).float()

    # Inseln < min_area verwerfen
    b, h, w = mask.shape
    for bi in range(b):
        m = mask[bi].cpu().numpy()
        m_uint8 = (m > 0.5).astype("uint8")
        n_labels, labels = cv2.connectedComponents(m_uint8)
        total_area = h * w
        for label_id in range(1, n_labels):
            area = (labels == label_id).sum()
            if area / total_area < min_area:
                m[labels == label_id] = 0.0
        mask[bi] = torch.from_numpy(m).to(image.device)

    # Rand weichzeichnen
    if feather_px > 0:
        for bi in range(b):
            m_np = mask[bi].cpu().numpy()
            ksize = 2 * feather_px + 1
            m_np = cv2.GaussianBlur(m_np, (ksize, ksize), sigmaX=feather_px / 2.0)
            mask[bi] = torch.from_numpy(m_np).to(image.device)
        mask = mask.clamp(0.0, 1.0)

    return mask
