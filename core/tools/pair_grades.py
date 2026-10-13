"""Ground-Truth-Gradings für den PAIR-Bench.

Jede Funktion: f(rgb: Tensor[...,3]) -> Tensor[...,3], sRGB 0..1.
Akzeptiert [H,W,3] oder [P,3] oder [B,H,W,3]. Keine Zufallszahlen.
"""

from __future__ import annotations

import math
import torch

from refgrade.color import (
    srgb_to_linear,
    linear_to_srgb,
    rgb_to_lab,
    lab_to_rgb,
    luma,
)

_EPS = 1e-6


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ensure_4d(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Stellt sicher, dass x 4D [B,H,W,3] ist. Gibt (x_4d, orig_dim) zurück."""
    if x.dim() == 2:  # [P,3] → [1,1,P,3]
        return x.unsqueeze(0).unsqueeze(1), 2
    elif x.dim() == 3:  # [H,W,3] → [1,H,W,3]
        return x.unsqueeze(0), 3
    return x, x.dim()


def _restore(x: torch.Tensor, orig_dim: int) -> torch.Tensor:
    if orig_dim == 2:
        return x.reshape(-1, 3)
    elif orig_dim == 3:
        return x.squeeze(0)
    return x


def gamma_up(x: torch.Tensor) -> torch.Tensor:
    """x^(1/1.4), alle Kanäle."""
    return x.clamp_min(_EPS) ** (1.0 / 1.4)


def gamma_down(x: torch.Tensor) -> torch.Tensor:
    """x^1.4, alle Kanäle."""
    return x.clamp_min(_EPS) ** 1.4


def s_curve(x: torch.Tensor) -> torch.Tensor:
    """Starker S-Kontrast: Sigmoid um 0.5, Steigung 2.5, Schwarz/Weiß bleiben."""
    k = 2.5 * math.log(99.0)
    raw = 1.0 / (1.0 + torch.exp(-k * (x - 0.5)))
    f0 = 1.0 / (1.0 + math.exp(k * 0.5))
    f1 = 1.0 / (1.0 + math.exp(-k * 0.5))
    out = (raw - f0) / (f1 - f0)
    return out.clamp(0.0, 1.0)


def flat_log(x: torch.Tensor) -> torch.Tensor:
    """Kontrast stark runter, Schwarz auf 0.08 angehoben (Log-Look)."""
    e = math.e
    log_x = torch.log1p(x * (e - 1.0)) / math.log(e)
    lin = x
    out = 0.6 * log_x + 0.4 * lin
    out = out * (1.0 - 0.08) + 0.08
    return out.clamp(0.0, 1.0)


def lift_gamma_gain(x: torch.Tensor) -> torch.Tensor:
    """Lift/Gamma/Gain je Kanal verschieden (Formel wie grade._lgg, ohne Zonen)."""
    out = x.clone()
    for i, (lift, gamma, gain) in enumerate([(0.02, 1.1, 0.95), (-0.01, 0.95, 1.05), (0.03, 1.2, 0.9)]):
        ch = out[..., i]
        ch = gain * (ch + lift * (1.0 - ch))
        ch = ch.clamp_min(_EPS) ** (1.0 / gamma)
        out[..., i] = ch
    return out.clamp(0.0, 1.0)


def split_tone(x: torch.Tensor) -> torch.Tensor:
    """Schatten nach Teal, Lichter nach Orange, Mitten neutral (weiche Zonen nach Luminanz)."""
    x4, od = _ensure_4d(x)
    lab = rgb_to_lab(x4)
    L = lab[..., 0]
    Ln = L / 100.0
    w_sh = 1.0 - _smoothstep(0.15, 0.5, Ln)
    w_hi = _smoothstep(0.5, 0.85, Ln)
    lab = lab.clone()
    lab[..., 1] = lab[..., 1] + w_sh * (-10.0) + w_hi * 15.0
    lab[..., 2] = lab[..., 2] + w_sh * (-15.0) + w_hi * 20.0
    return _restore(lab_to_rgb(lab), od)


def three_zone(x: torch.Tensor) -> torch.Tensor:
    """Schatten, Mitten und Lichter je mit eigener Tönung und eigenem Kontrast."""
    x4, od = _ensure_4d(x)
    lab = rgb_to_lab(x4)
    L = lab[..., 0]
    Ln = L / 100.0
    w_sh = 1.0 - _smoothstep(0.15, 0.5, Ln)
    w_hi = _smoothstep(0.5, 0.85, Ln)
    w_mid = (1.0 - w_sh - w_hi).clamp_min(0.0)

    contrast = w_sh * 0.8 + w_mid * 1.0 + w_hi * 1.3
    L_new = (L - 50.0) * contrast + 50.0
    a_off = w_sh * (-8.0) + w_hi * 10.0
    b_off = w_sh * (-12.0) + w_hi * 15.0

    lab = lab.clone()
    lab[..., 0] = L_new
    lab[..., 1] = lab[..., 1] + a_off
    lab[..., 2] = lab[..., 2] + b_off
    return _restore(lab_to_rgb(lab), od)


def hue_band(x: torch.Tensor) -> torch.Tensor:
    """Nur Grün → Richtung Gelb um 25°, Sättigung der Blautöne −40 %."""
    r, g, b = x[..., 0:1], x[..., 1:2], x[..., 2:3]
    mx = x.amax(dim=-1, keepdim=True)
    mn = x.amin(dim=-1, keepdim=True)
    delta = mx - mn
    V = mx
    S = delta / mx.clamp_min(_EPS)
    h = torch.zeros_like(V)
    h = torch.where(
        (mx == r) & (delta > _EPS),
        (((g - b) / delta.clamp_min(_EPS)) % 6.0) * 60.0,
        torch.where(
            (mx == g) & (delta > _EPS),
            ((b - r) / delta.clamp_min(_EPS) + 2.0) * 60.0,
            ((r - g) / delta.clamp_min(_EPS) + 4.0) * 60.0,
        ),
    )
    h = h % 360.0

    dh_green = torch.remainder(h - 120.0 + 180.0, 360.0) - 180.0
    w_green = torch.where(dh_green.abs() < 60.0,
                          0.5 * (1.0 + torch.cos(math.pi * dh_green / 60.0)),
                          torch.zeros_like(V))
    dh_blue = torch.remainder(h - 240.0 + 180.0, 360.0) - 180.0
    w_blue = torch.where(dh_blue.abs() < 60.0,
                         0.5 * (1.0 + torch.cos(math.pi * dh_blue / 60.0)),
                         torch.zeros_like(V))

    h_new = (h + w_green * 25.0) % 360.0
    s_new = S * (1.0 - w_blue * 0.4)

    C = V * s_new
    hp = h_new / 60.0
    X = C * (1.0 - torch.abs(hp % 2.0 - 1.0))
    m = V - C
    sector = hp.floor().long() % 6
    r2 = torch.where(sector == 0, C,
        torch.where(sector == 1, X,
        torch.where(sector == 4, X,
        torch.where(sector == 5, C, 0.0))))
    g2 = torch.where(sector == 0, X,
        torch.where(sector == 1, C,
        torch.where(sector == 2, C,
        torch.where(sector == 3, X, 0.0))))
    b2 = torch.where(sector == 2, C,
        torch.where(sector == 3, X,
        torch.where(sector == 4, C,
        torch.where(sector == 5, X, 0.0))))
    return torch.cat([r2 + m, g2 + m, b2 + m], dim=-1).clamp(0.0, 1.0)


def sat_by_lum(x: torch.Tensor) -> torch.Tensor:
    """Sättigung in Schatten −50 %, in Lichtern +30 %."""
    Y = luma(x)
    Ln = Y.squeeze(-1)
    w_sh = 1.0 - _smoothstep(0.15, 0.5, Ln)
    w_hi = _smoothstep(0.5, 0.85, Ln)
    sat_factor = 1.0 + w_sh.unsqueeze(-1) * (-0.5) + w_hi.unsqueeze(-1) * 0.3
    return (Y + sat_factor * (x - Y)).clamp(0.0, 1.0)


def cross_matrix(x: torch.Tensor) -> torch.Tensor:
    """3×3-Kanalmischung (Film-Übersprechen) + Offset."""
    M = torch.tensor([
        [1.05, 0.02, 0.01],
        [0.01, 1.03, 0.02],
        [0.02, 0.01, 1.06],
    ], dtype=x.dtype, device=x.device)
    offset = torch.tensor([0.01, 0.005, -0.01], dtype=x.dtype, device=x.device)
    return (x @ M.T + offset).clamp(0.0, 1.0)


def highlight_clip(x: torch.Tensor) -> torch.Tensor:
    """Belichtung +1 Blende mit weichem Schulter-Roll-off."""
    lin = srgb_to_linear(x) * 2.0
    k = 4.0
    out = (1.0 - torch.exp(-lin * k)) / (1.0 - math.exp(-k))
    return linear_to_srgb(out).clamp(0.0, 1.0)


def bw_tint(x: torch.Tensor) -> torch.Tensor:
    """Schwarzweiß mit Sepia-Tönung."""
    Y = luma(x)
    sepia = torch.cat([Y, Y * 0.85, Y * 0.7], dim=-1)
    return sepia.clamp(0.0, 1.0)


def combo(x: torch.Tensor) -> torch.Tensor:
    """s_curve ∘ split_tone ∘ hue_band ∘ gamma_down."""
    return s_curve(split_tone(hue_band(gamma_down(x))))


# --- Registry (ohne real_lut_k, die werden im Bench dynamisch erzeugt) ---

GRADES: dict[str, callable] = {
    "gamma_up": gamma_up,
    "gamma_down": gamma_down,
    "s_curve": s_curve,
    "flat_log": flat_log,
    "lift_gamma_gain": lift_gamma_gain,
    "split_tone": split_tone,
    "three_zone": three_zone,
    "hue_band": hue_band,
    "sat_by_lum": sat_by_lum,
    "cross_matrix": cross_matrix,
    "highlight_clip": highlight_clip,
    "bw_tint": bw_tint,
    "combo": combo,
}
