"""Zonal-Abgleich: zone_weights, analyze, apply_params."""

from __future__ import annotations

import math
import torch

from .color import rgb_to_lab
from .stats import sample, weighted_mean, weighted_std, weighted_quantile

_EPS = 1e-6
_NUM_L_QUANTILES = 17


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Glatte Schrittfunktion: 0 unter edge0, 1 über edge1."""
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def zone_weights(L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zonen-Gewichte aus Lab-L* [0,100].

    Returns:
        (w_shadows, w_highlights, w_mids), jeweils gleiche Form wie L.
    """
    Ln = L / 100.0
    w_sh = 1.0 - _smoothstep(0.15, 0.5, Ln)
    w_hi = _smoothstep(0.5, 0.85, Ln)
    w_mid = (1.0 - w_sh - w_hi).clamp_min(0.0)
    return w_sh, w_hi, w_mid


def analyze(
    image: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None = None,
    reference_weights: torch.Tensor | None = None,
) -> dict:
    """Zonal-Analyse: L-Kurve, Zonen-Δab, Chroma-Skala.

    Returns:
        JSON-fähiges dict mit version, L_curve, zones, chroma_scale.
    """
    lab_img = rgb_to_lab(image)
    lab_ref = rgb_to_lab(reference)

    # Subsampling
    pix_s, w_s = sample(lab_img, image_weights)
    pix_r, w_r = sample(lab_ref, reference_weights)

    # L-Kurve: 17 Quantile
    dev = pix_s.device
    q_positions = (torch.arange(_NUM_L_QUANTILES, device=dev, dtype=torch.float64) + 0.5) / _NUM_L_QUANTILES
    L_s = weighted_quantile(pix_s[:, 0].double(), w_s.double(), q_positions)
    L_r = weighted_quantile(pix_r[:, 0].double(), w_r.double(), q_positions)

    # Zonen nach L **vor** der Kurve
    w_sh_s, w_hi_s, w_mid_s = zone_weights(pix_s[:, 0])
    w_sh_r, w_hi_r, w_mid_r = zone_weights(pix_r[:, 0])

    # Δab pro Zone: gewichteter Mittelwert (Vorlage − Bild)
    zones: dict[str, list[float]] = {}
    for name, w_s_z, w_r_z in (
        ("shadows", w_sh_s, w_sh_r),
        ("mids", w_mid_s, w_mid_r),
        ("highlights", w_hi_s, w_hi_r),
    ):
        zw_s = w_s_z * w_s
        zw_r = w_r_z * w_r
        w_s_sum = zw_s.sum().clamp_min(_EPS)
        w_r_sum = zw_r.sum().clamp_min(_EPS)
        da = ((pix_r[:, 1] * zw_r).sum() / w_r_sum).item() - ((pix_s[:, 1] * zw_s).sum() / w_s_sum).item()
        db = ((pix_r[:, 2] * zw_r).sum() / w_r_sum).item() - ((pix_s[:, 2] * zw_s).sum() / w_s_sum).item()
        zones[name] = [da, db]

    # Chroma-Skala: σC_r / σC_s
    C_s = (pix_s[:, 1] ** 2 + pix_s[:, 2] ** 2).sqrt()
    C_r = (pix_r[:, 1] ** 2 + pix_r[:, 2] ** 2).sqrt()
    sigma_c_s = weighted_std(C_s.unsqueeze(-1), w_s).item()
    sigma_c_r = weighted_std(C_r.unsqueeze(-1), w_r).item()
    chroma_scale = min(max(sigma_c_r / max(sigma_c_s, _EPS), 0.25), 4.0)

    return {
        "version": 1,
        "L_curve": {
            "src": [float(v) for v in L_s.tolist()],
            "dst": [float(v) for v in L_r.tolist()],
        },
        "zones": zones,
        "chroma_scale": float(chroma_scale),
    }


def apply_params(
    params: dict,
    light_strength: float = 1.0,
    color_strength: float = 1.0,
) -> callable:
    """Erstelle Pro-Pixel-Callable aus Zonal-Parametern.

    Args:
        params: Ausgabe von analyze().
        light_strength: skaliert die L-Änderung.
        color_strength: skaliert die a/b-Änderung.

    Returns:
        Callable: Lab [B,H,W,3] → Lab [B,H,W,3].
    """
    src = torch.tensor(params["L_curve"]["src"], dtype=torch.float64)
    dst = torch.tensor(params["L_curve"]["dst"], dtype=torch.float64)
    # Monotonie sicherstellen
    for i in range(1, len(dst)):
        if dst[i] < dst[i - 1]:
            dst[i] = dst[i - 1]

    zones = params["zones"]
    chroma_scale = params["chroma_scale"]

    def lab_fn(lab: torch.Tensor) -> torch.Tensor:
        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]
        dev = lab.device

        # L-Kurve: stückweise lineare Interpolation
        src_d = src.to(dev)
        dst_d = dst.to(dev)
        L_flat = L.double().flatten()
        idx = torch.searchsorted(src_d, L_flat)
        idx = idx.clamp(1, len(src_d) - 1)
        x0 = src_d[idx - 1]
        x1 = src_d[idx]
        frac = (L_flat - x0) / (x1 - x0).clamp_min(_EPS)
        L_new_flat = dst_d[idx - 1] + frac * (dst_d[idx] - dst_d[idx - 1])
        L_new = L_new_flat.reshape(L.shape).float()

        # L mit light_strength mischen
        L_out = L + light_strength * (L_new - L)

        # Zonen-Δab
        w_sh, w_hi, w_mid = zone_weights(L)
        da_sh, db_sh = zones["shadows"]
        da_mid, db_mid = zones["mids"]
        da_hi, db_hi = zones["highlights"]
        da = w_sh * da_sh + w_mid * da_mid + w_hi * da_hi
        db = w_sh * db_sh + w_mid * db_mid + w_hi * db_hi

        # Chroma skalieren + Zonen-Offset; color_strength mischt zwischen Original und Ziel
        C = (a ** 2 + b ** 2).sqrt().clamp_min(_EPS)
        C_new = C * chroma_scale
        angle = torch.atan2(b, a)
        a_full = C_new * torch.cos(angle) + da
        b_full = C_new * torch.sin(angle) + db
        a_out = a + color_strength * (a_full - a)
        b_out = b + color_strength * (b_full - b)

        return torch.stack([L_out, a_out, b_out], dim=-1)

    return lab_fn
