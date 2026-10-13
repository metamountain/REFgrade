"""Bildanalyse: analyze_image, compare — Grundlage für Galerie und manuelle Arbeit."""

from __future__ import annotations

import math
import numpy as np
import torch

from .color import rgb_to_lab, luma, srgb_to_linear
from .stats import sample, weighted_mean
from .zonal import zone_weights

_EPS = 1e-6
_NUM_PALETTE = 6
_PALETTE_SEED = 0


def _quantile(values: torch.Tensor, q: float) -> float:
    """Einfacher Quantil (linear interpolation)."""
    sorted_v = values.sort().values
    n = sorted_v.shape[0]
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac)


def _hue_strength(a: float, b: float) -> tuple[float, float]:
    """Farbton in Grad (0..360) und Stärke (Chroma)."""
    hue = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0
    strength = math.sqrt(a * a + b * b)
    return hue, strength


def analyze_image(
    img: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> dict:
    """Vollständige Bildanalyse.

    Args:
        img: [B,H,W,3] sRGB 0..1.
        weights: [B,H,W] 0..1 (1 = voll gewichtet).

    Returns:
        dict mit JSON-Typen: tone, color, palette.
    """
    lab = rgb_to_lab(img)
    pix, w = sample(lab, weights)

    L = pix[:, 0]
    a = pix[:, 1]
    b = pix[:, 2]

    # --- Ton ---
    l_p01 = _quantile(L, 0.01)
    l_p99 = _quantile(L, 0.99)
    l_median = _quantile(L, 0.50)
    l_p10 = _quantile(L, 0.10)
    l_p90 = _quantile(L, 0.90)
    contrast = l_p90 - l_p10

    clipped_shadows = float((L < 1.0).float().mean().item())
    clipped_highlights = float((L > 99.0).float().mean().item())

    # Median lineare Luma (für Exposure-Schätzung)
    luma_srgb = luma(img)  # [B,H,W,1]
    luma_linear = srgb_to_linear(luma_srgb)
    luma_linear_flat = luma_linear.reshape(-1)
    if weights is not None:
        w_flat = weights.reshape(-1)
        median_linear = float(
            torch.quantile(luma_linear_flat, 0.5,
                           weights=w_flat).item()
        ) if luma_linear_flat.numel() < 100000 else float(
            luma_linear_flat.sort().values[len(luma_linear_flat) // 2].item()
        )
    else:
        median_linear = float(luma_linear_flat.median().item())

    # L-Standardabweichung (zur Trennung Exposure vs. Temperature)
    l_std = float(L.std().item()) if L.shape[0] > 1 else 0.0

    # --- Farbe ---
    # Gesamtstich: gewichtetes a/b-Mittel der fast neutralen Pixel (Chroma < 10)
    chroma = torch.sqrt(a**2 + b**2)
    neutral_mask = chroma < 10.0
    if neutral_mask.sum() > 10:
        w_neutral = w * neutral_mask.float()
        w_sum = w_neutral.sum().clamp_min(_EPS)
        cast_a = float((a * w_neutral).sum() / w_sum)
        cast_b = float((b * w_neutral).sum() / w_sum)
    else:
        cast_a, cast_b = 0.0, 0.0
    cast_hue, cast_strength = _hue_strength(cast_a, cast_b)

    # Mittel in linearem RGB (Weißabgleich ist von Kries in linearem Licht, 2026-10-13)
    lin_pix, w_lin = sample(srgb_to_linear(img.clamp(0.0, 1.0)), weights)
    mean_linear_rgb = [float((lin_pix[:, c] * w_lin).sum() / w_lin.sum().clamp_min(_EPS)) for c in range(3)]

    # a/b-Mittel über alle Pixel
    mean_a = float((a * w).sum() / w.sum().clamp_min(_EPS))
    mean_b = float((b * w).sum() / w.sum().clamp_min(_EPS))

    # Zonen-Tönung
    w_sh, w_hi, w_mid = zone_weights(L)
    zone_tones = {}
    for name, wz in (("shadows", w_sh), ("mids", w_mid), ("highlights", w_hi)):
        zw = w * wz
        zw_sum = zw.sum().clamp_min(_EPS)
        za = float((a * zw).sum() / zw_sum)
        zb = float((b * zw).sum() / zw_sum)
        zh, zs = _hue_strength(za, zb)
        zone_tones[name] = {"hue": round(zh, 1), "strength": round(zs, 2)}

    # Sättigung
    chroma_median = _quantile(chroma, 0.50)
    chroma_p90 = _quantile(chroma, 0.90)

    # --- Palette: k-means, 6 Farben, Seed 0 ---
    n_km = min(4096, pix.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_PALETTE_SEED)
    if pix.shape[0] > n_km:
        idx = torch.randperm(pix.shape[0], generator=gen)[:n_km]
        pix_km = pix[idx].cpu().numpy()
    else:
        pix_km = pix.cpu().numpy()

    rng = np.random.default_rng(_PALETTE_SEED)
    centers = pix_km[rng.choice(n_km, _NUM_PALETTE, replace=False)]
    for _ in range(20):
        dists = np.linalg.norm(pix_km[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)
        new_centers = centers.copy()
        for k in range(_NUM_PALETTE):
            mask_k = labels == k
            if mask_k.sum() > 0:
                new_centers[k] = pix_km[mask_k].mean(axis=0)
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    dists = np.linalg.norm(pix_km[:, None, :] - centers[None, :, :], axis=2)
    labels = dists.argmin(axis=1)
    palette = []
    for k in range(_NUM_PALETTE):
        count = (labels == k).sum()
        fraction = float(count / n_km)
        palette.append({
            "lab": [round(float(centers[k][0]), 2),
                     round(float(centers[k][1]), 2),
                     round(float(centers[k][2]), 2)],
            "fraction": round(fraction, 4),
        })
    palette.sort(key=lambda p: -p["fraction"])

    return {
        "tone": {
            "black_point": round(l_p01, 2),
            "white_point": round(l_p99, 2),
            "median_l": round(l_median, 2),
            "std_l": round(l_std, 3),
            "contrast": round(contrast, 2),
            "clipped_shadows": round(clipped_shadows, 5),
            "clipped_highlights": round(clipped_highlights, 5),
            "median_linear_luma": round(median_linear, 6),
        },
        "color": {
            "cast": {"hue": round(cast_hue, 1), "strength": round(cast_strength, 2)},
            "mean_linear_rgb": [round(v, 6) for v in mean_linear_rgb],
            "mean_a": round(mean_a, 3),
            "mean_b": round(mean_b, 3),
            "zones": zone_tones,
            "saturation": {
                "chroma_median": round(chroma_median, 2),
                "chroma_p90": round(chroma_p90, 2),
            },
        },
        "palette": palette,
    }


def compare(
    src: dict,
    ref: dict,
) -> dict:
    """Vergleich zweier Analysen → vorgeschlagene Reglerwerte + deutsche Sätze.

    Args:
        src: analyze_image(Bild)
        ref: analyze_image(Vorlage)

    Returns:
        dict mit sliders (dict name→value) und sentences (list[str]).
    """
    sliders: dict[str, float] = {}
    sentences: list[str] = []

    # --- Exposure from the linear-luma ratio; Temperature/Tint from the linear channel ratio ---
    # grade._temp_tint is von Kries since 2026-10-13: gains R·2^(0.8t), G·2^(−0.5n), B·2^(−0.8t).
    # The gains that turn the source mean into the reference mean give t and n exactly.
    ml_src = max(src["tone"]["median_linear_luma"], _EPS)
    ml_ref = ref["tone"]["median_linear_luma"]
    delta_l = ref["tone"]["median_l"] - src["tone"]["median_l"]
    exposure = math.log2(ml_ref / ml_src) if ml_ref > _EPS else 0.0
    if abs(exposure) < 0.05:
        exposure = 0.0
    g = [max(r, _EPS) / max(q, _EPS) for r, q in zip(ref["color"]["mean_linear_rgb"],
                                                     src["color"]["mean_linear_rgb"])]
    temperature = math.log2(g[0] / g[2]) / 1.6

    sliders["exposure"] = round(max(-4.0, min(4.0, exposure)), 3)
    sliders["temperature"] = round(max(-1.0, min(1.0, temperature)), 3)
    if abs(exposure) > 0.1:
        direction = "heller" if exposure > 0 else "dunkler"
        sentences.append(f"Vorlage ist {direction} (ΔL {delta_l:+.1f}).")

    # --- Tint: aus dem Grün-Verhältnis (gleiche von-Kries-Umkehrung) ---
    tint = -2.0 * (math.log2(g[1] / g[0]) + 0.8 * temperature)
    sliders["tint"] = round(max(-1.0, min(1.0, tint)), 3)

    # --- Contrast ---
    c_src = max(src["tone"]["contrast"], _EPS)
    c_ref = ref["tone"]["contrast"]
    contrast_val = max(0.0, min(3.0, c_ref / c_src))
    sliders["contrast"] = round(contrast_val, 2)

    # --- Pivot ---
    wp_src = max(src["tone"]["white_point"], _EPS)
    wp_ref = ref["tone"]["white_point"]
    pivot = max(0.0, min(1.0, 1.0 - (wp_ref - wp_src) / 100.0))
    sliders["pivot"] = round(pivot, 2)

    # --- Saturation: Chroma-Median-Verhältnis ---
    sat_src = max(src["color"]["saturation"]["chroma_median"], _EPS)
    sat_ref = ref["color"]["saturation"]["chroma_median"]
    saturation = max(0.0, min(3.0, sat_ref / sat_src))
    sliders["saturation"] = round(saturation, 2)

    # --- Lift_m: aus Schwarzpunkt ---
    bp_src = src["tone"]["black_point"]
    bp_ref = ref["tone"]["black_point"]
    lift_m = (bp_src - bp_ref) / 100.0
    sliders["lift_m"] = round(max(-0.5, min(0.5, lift_m)), 3)

    # --- Zonen-Tönung ---
    for zone_name, slider_prefix in (("shadows", "shadow"), ("highlights", "highlight")):
        z_src = src["color"]["zones"][zone_name]
        z_ref = ref["color"]["zones"][zone_name]
        hue_diff = (z_ref["hue"] - z_src["hue"] + 180) % 360 - 180
        amount_diff = (z_ref["strength"] - z_src["strength"]) / 30.0

        sliders[f"{slider_prefix}_hue"] = round((z_src["hue"] + hue_diff) % 360, 1)
        sliders[f"{slider_prefix}_amount"] = round(max(0.0, min(1.0, amount_diff)), 3)

        if abs(amount_diff) > 0.05:
            zone_de = "Schatten" if zone_name == "shadows" else "Lichter"
            h = z_ref["hue"]
            s = z_ref["strength"]
            if 0 <= h < 45 or h >= 315:
                color_name = "warm-braun"
            elif 45 <= h < 90:
                color_name = "gelblich"
            elif 90 <= h < 150:
                color_name = "grünlich"
            elif 150 <= h < 210:
                color_name = "kühl-blau"
            elif 210 <= h < 270:
                color_name = "blau-violett"
            else:
                color_name = "magenta"
            sentences.append(f"{zone_de} der Vorlage: {color_name}, Farbton {h:.0f}°, Stärke {s:.1f}")

    return {"sliders": sliders, "sentences": sentences}
