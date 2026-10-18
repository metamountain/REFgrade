"""Manuelle Grading-Regler: grade(x, values) — differenzierbar, ohne finales clamp."""

from __future__ import annotations

import math
import torch

from .color import srgb_to_linear, linear_to_srgb, rgb_to_lab, lab_to_rgb, luma
from .color import rgb_to_hsv as _rgb_to_hsv, hsv_to_rgb as _hsv_to_rgb
from .color_replace import apply as _color_replace_apply, all_cr_names
from .sliders import SLIDERS
from .zonal import zone_weights

_EPS = 1e-6
_SEL_CENTERS = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
_SEL_BANDS = ("red", "yellow", "green", "cyan", "blue", "magenta")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _is_default(vals: dict[str, float | torch.Tensor], names: list[str]) -> bool:
    """True wenn alle genannten Werte Python-float und auf Standard."""
    d = _defaults()
    for n in names:
        v = vals.get(n, d[n])
        if not isinstance(v, float):
            return False
        if v != d[n]:
            return False
    return True


def _exposure(x: torch.Tensor, val: float | torch.Tensor) -> torch.Tensor:
    """Belichtung in linearem RGB: x · 2^exposure."""
    linear = srgb_to_linear(x)
    linear = linear * (2.0 ** val)
    return linear_to_srgb(linear)


_WB_STOPS_TEMP = 0.8     # ±1 → R/B-Verhältnis × 2^(±1,6) ≈ 3 (deckt Glühlampe ↔ Schatten ab)
_WB_STOPS_TINT = 0.5     # ±1 → G × 2^(∓0,5)


def _temp_tint(x: torch.Tensor, temp: float | torch.Tensor, tint: float | torch.Tensor) -> torch.Tensor:
    """Weißabgleich (von Kries): Kanalfaktoren in linearem Licht, Grau behält seine Helligkeit.

    Vorher additiv in Lab (b* += 20·temp): verschob Schwarz, Grau und Weiß um denselben Betrag —
    ein Farbstich durch das Licht ist aber multiplikativ (wächst mit der Helligkeit) und ließ sich so
    nicht entfernen (K23: 15 statt 3,5 ΔE; AUTO blieb bei Stich 4,5 stehen; 2026-10-13).
    temp > 0 = wärmer (R hoch, B runter), tint > 0 = magenta (G runter) — Richtungen wie vorher.
    """
    t = torch.as_tensor(temp, dtype=x.dtype, device=x.device)
    n = torch.as_tensor(tint, dtype=x.dtype, device=x.device)
    gains = torch.stack([2.0 ** (_WB_STOPS_TEMP * t), 2.0 ** (-_WB_STOPS_TINT * n),
                         2.0 ** (-_WB_STOPS_TEMP * t)])
    gains = gains / (0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2])
    # nur die Änderung addieren: bei temp = tint = 0 exakt die Eingabe (Identitätsregel) — der
    # Hin-/Rückweg sRGB↔linear allein weicht ~3e-6 ab, und Adam (FIT) machte daraus volle Schritte
    lin = srgb_to_linear(x.clamp(0.0, 1.0))
    return x + (linear_to_srgb(lin * gains) - linear_to_srgb(lin))


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Smoothstep: 0 unter edge0, 1 über edge1, kubisch dazwischen."""
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _zone_masks(Y: torch.Tensor, gain: float, falloff: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drei Zonenmasken als weiche Bumps (Glow) auf der Luminanzachse 0…1.

    2026-10-18 (Nutzer): ersetzt das alte Split-Modell (p/w/o). Jede Zone ist ein kleiner
    WEISSER BEREICH (Kern) an ihrem Zentrum — Schatten bei Y=0, Mitten bei Y=0,5, Lichter
    bei Y=1 — mit weichem Falloff (Glow/Blur) darum. Gain macht den Kern größer, Falloff
    den Rand weicher (Falloff 1,0 = volle Reichweite 0→1, schwarz→weiß). Im Standard
    (0/0,5) überlappen die Zonen weich über die ganze Skala; Gain und/oder Falloff
    verkleinert (bzw. erhöht) die Überdeckung — niedrige Werte öffnen Lücken, in denen
    kein Rad wirkt.
    """
    g = torch.as_tensor(gain, dtype=Y.dtype, device=Y.device)
    f = torch.as_tensor(falloff, dtype=Y.dtype, device=Y.device)
    f_val = f.detach().item()   # nur für die Zweig-Entscheidung; die Formel läuft mit Tensor (Autograd!)

    def _bump(dist: torch.Tensor) -> torch.Tensor:
        # dist = Abstand vom Kernrand (0 im Kern). f == 0 → harte Kante.
        if f_val <= 0.0:
            return (dist <= 0.0).to(Y.dtype)
        return 1.0 - _smoothstep(0.0, f, dist.clamp_min(0.0).clamp_max(f))

    S = _bump(Y - g)                    # Kern [0, g], Falloff bis g+f
    H = _bump((1.0 - g) - Y)            # Kern [1-g, 1], Falloff bis 1-g-f
    M = _bump(torch.abs(Y - 0.5) - g)   # Kern [0.5-g, 0.5+g], beide Seiten
    return S, M, H


def _lgg(x: torch.Tensor, vals: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Lift/Gamma/Gain in zwei Schritten: erst Helligkeit (Regler unter den Rädern), dann Farbe.

    Nutzer 2026-10-12: die Regler Highlights/Midtones/Shadows (gain_m/gamma_m/lift_m) sind
    Vorverarbeitung — sie verschieben die Helligkeit in ihrer Zone, danach werden die Zonen aus dem
    *neuen* Bild bestimmt und erst dann die Radfarben angewendet. Ein Licht, das man mit dem
    Highlights-Regler abdunkelt, rutscht so in die Mitten und bekommt die Mittenfarbe.
    """
    d = _defaults()
    gain = vals.get("wheel_gain", d["wheel_gain"])
    falloff = vals.get("wheel_falloff", d["wheel_falloff"])
    k = vals.get("wheel_intensity", d["wheel_intensity"])

    def _zones(img: torch.Tensor):
        Y = luma(img).squeeze(-1)
        return _zone_masks(Y, gain, falloff)

    def _apply(ch, lift, gamma, gain, S, M, H):
        ch = (1.0 + (gain - 1.0) * H) * (ch + lift * (1.0 - ch) * S)
        return ch.clamp_min(_EPS) ** (1.0 / (1.0 + (gamma - 1.0) * M))

    # 1) Helligkeit: Master-Regler, Zonen aus dem Eingang, alle Kanäle gleich
    lift_m = vals.get("lift_m", d["lift_m"])
    gamma_m = torch.as_tensor(vals.get("gamma_m", d["gamma_m"])).clamp_min(_EPS)
    gain_m = torch.as_tensor(vals.get("gain_m", d["gain_m"])).clamp_min(_EPS)
    S, M, H = _zones(x)
    out = _apply(x, lift_m, gamma_m, gain_m, S[..., None], M[..., None], H[..., None])

    # 2) Farbe: Zonen aus dem helligkeitskorrigierten Bild, Radfarben je Kanal
    S, M, H = _zones(out)
    out = out.clone()
    for i, c in enumerate(("r", "g", "b")):
        # Intensity skaliert nur die Farbanteile der Räder (um neutral 0 bzw. 1)
        lift_c = k * vals.get(f"lift_{c}", d[f"lift_{c}"])
        gamma_c = torch.as_tensor(vals.get(f"gamma_{c}", d[f"gamma_{c}"])).clamp_min(_EPS) ** k
        gain_c = torch.as_tensor(vals.get(f"gain_{c}", d[f"gain_{c}"])).clamp_min(_EPS) ** k
        out[..., i] = _apply(out[..., i], lift_c, gamma_c, gain_c, S, M, H)
    return out


def _rolloff(y: torch.Tensor, r: float | torch.Tensor) -> torch.Tensor:
    """Highlight-Rolloff als echte Schulter: holt Luma > Knie k = 1 − 0.6·r per tanh weich zurück.

    r == 0 → Eingabe exakt (bitgleich) unverändert. Unter k bleibt jedes Pixel exakt; über k wird
    Yn = k + (1−k)·tanh((Y−k)/(1−k)) und das Pixel mit Yn/max(Y, 1e-6) multipliziert, damit der
    Farbton bleibt. Kein Clamp hier — die Kette entscheidet, was am Ende geclippt wird.
    """
    r_t = torch.as_tensor(r, dtype=y.dtype, device=y.device)
    if r_t.item() == 0.0:
        return y
    k = 1.0 - 0.6 * r_t
    Y = luma(y)
    Yn = torch.where(
        Y <= k,
        Y,
        k + (1.0 - k) * torch.tanh((Y - k) / (1.0 - k).clamp_min(_EPS)),
    )
    # luma ≤ 0 (contrast > 1 pushes darks below 0) stays as is: Yn/clamp(Y) was −Y/1e-6 →
    # huge negative scale → dark pixels turned white/colourful (debug round 2026-10-17)
    scale = torch.where(Y > _EPS, Yn / Y.clamp_min(_EPS), torch.ones_like(Y))
    return y * scale


def _contrast(x: torch.Tensor, vals: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Kontrast um pivot + Highlight-Rolloff."""
    d = _defaults()
    contrast = vals.get("contrast", d["contrast"])
    pivot = vals.get("pivot", d["pivot"])
    rolloff = vals.get("highlight_rolloff", d["highlight_rolloff"])
    y = (x - pivot) * contrast + pivot
    y = _rolloff(y, rolloff)
    return y


def _tone_zones(x: torch.Tensor, vals: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Lightroom-like tone zones: highlights/shadows/whites/blacks in L*."""
    d = _defaults()
    hi = vals.get("highlights", d["highlights"])
    sh = vals.get("shadows", d["shadows"])
    wh = vals.get("whites", d["whites"])
    bl = vals.get("blacks", d["blacks"])

    lab = rgb_to_lab(x)
    lab0 = lab.clone()   # Identitätsregel: nur die Änderung zurückrechnen (2026-10-13)
    L = lab[..., 0] / 100.0  # [0, 1]

    def bump(c: float) -> torch.Tensor:
        return torch.exp(-((L - c) / 0.18) ** 2) * 4 * L * (1 - L) / 0.75

    Lp = L + 0.25 * (sh * bump(0.25) + hi * bump(0.75) + bl * (1 - L) ** 3 + wh * L ** 3)

    # Monotonicity: evaluate on 1024 points, cumulative max, linear interpolate
    xs = torch.linspace(0, 1, 1024, device=x.device, dtype=x.dtype)
    def bump_s(c: float) -> torch.Tensor:
        return torch.exp(-((xs - c) / 0.18) ** 2) * 4 * xs * (1 - xs) / 0.75
    ys = xs + 0.25 * (sh * bump_s(0.25) + hi * bump_s(0.75) + bl * (1 - xs) ** 3 + wh * xs ** 3)
    ys = torch.cummax(ys, dim=0).values.clamp(0.0, 1.0)
    idx = L.clamp(0.0, 1.0) * 1023
    i0 = idx.floor().long().clamp(0, 1022)
    f = idx - i0
    dy = ys - xs                       # nur die Abweichung interpolieren: am Standard exakt 0
    Lp = L + dy[i0] * (1 - f) + dy[i0 + 1] * f

    lab = lab.clone()
    lab[..., 0] = Lp * 100
    return x + (lab_to_rgb(lab) - lab_to_rgb(lab0))


def _saturation(x: torch.Tensor, val: float | torch.Tensor) -> torch.Tensor:
    """Sättigung über Luma: y = Y + s·(x−Y)."""
    Y = luma(x)
    return Y + val * (x - Y)


def _vibrance(x: torch.Tensor, val: float | torch.Tensor) -> torch.Tensor:
    """Vibrance: wie Sättigung, Faktor gewichtet mit 1 − aktuelle Sättigung."""
    Y = luma(x)
    mx = x.amax(dim=-1, keepdim=True)
    mn = x.amin(dim=-1, keepdim=True)
    sat = (mx - mn) / mx.clamp_min(_EPS)
    factor = 1.0 + val * (1.0 - sat)
    return Y + factor * (x - Y)


def _selective_color(x: torch.Tensor, vals: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Selektive Farbe: sechs Bänder mit weichen Cosinus-Fenstern."""
    d = _defaults()
    h, S, V = _rgb_to_hsv(x)

    # Raised-cosine windows of ±60° around each band centre, hue difference wrapped to
    # (-180°, 180°]: neighbouring windows sum to exactly 1. (Until 2026-10-08 the difference was
    # not wrapped and the cosine repeated every 120°, so each band also hit hues 120° away and
    # the bands largely cancelled — "HSL not working".)
    weights = []
    for center in _SEL_CENTERS:
        dh = torch.remainder(h - center + 180.0, 360.0) - 180.0
        w = 0.5 * (1.0 + torch.cos(math.pi * dh / 60.0))
        weights.append(torch.where(dh.abs() < 60.0, w, torch.zeros_like(w)))

    # Gewichtete Summe der Band-Adjustments
    hue_shift = torch.zeros_like(h)
    sat_adj = torch.zeros_like(h)
    light_adj = torch.zeros_like(h)
    for i, band in enumerate(_SEL_BANDS):
        hs = vals.get(f"sel_{band}_hue", d[f"sel_{band}_hue"])
        sa = vals.get(f"sel_{band}_sat", d[f"sel_{band}_sat"])
        la = vals.get(f"sel_{band}_light", d[f"sel_{band}_light"])
        w = weights[i]
        hue_shift = hue_shift + w * hs
        sat_adj = sat_adj + w * sa
        light_adj = light_adj + w * la

    # Anwenden: Hue und Lightness in HSV, Sättigung luminanzerhaltend
    h_new = (h + hue_shift) % 360.0
    v_new = (V + light_adj).clamp(0.0, 1.0)
    rgb_h = _hsv_to_rgb(h_new, S, v_new)
    Y = 0.2126 * rgb_h[..., 0] + 0.7152 * rgb_h[..., 1] + 0.0722 * rgb_h[..., 2]
    Y = Y.unsqueeze(-1)
    rgb_new = (Y + (rgb_h - Y) * (1.0 + sat_adj)).clamp(0.0, 1.0)

    # Only (near-)grey pixels are protected — their hue is meaningless. Full effect from S≈0.15.
    t = ((S - 0.02) / 0.13).clamp(0.0, 1.0)
    guard = t * t * (3.0 - 2.0 * t)
    return x + guard * (rgb_new - x)


def _split_toning(x: torch.Tensor, vals: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Split-Toning in Lab: ab += amount·30·(cos h, sin h), gewichtet mit Zonen."""
    d = _defaults()
    sh_hue = vals.get("shadow_hue", d["shadow_hue"])
    sh_amt = vals.get("shadow_amount", d["shadow_amount"])
    hi_hue = vals.get("highlight_hue", d["highlight_hue"])
    hi_amt = vals.get("highlight_amount", d["highlight_amount"])
    lab = rgb_to_lab(x)
    lab0 = lab.clone()   # Identitätsregel: nur die Änderung zurückrechnen (2026-10-13)
    L = lab[..., 0]  # [0, 100]
    w_sh, w_hi, _ = zone_weights(L)
    w_sh = w_sh.unsqueeze(-1)
    w_hi = w_hi.unsqueeze(-1)

    def _hue_vec(hue: float | torch.Tensor) -> torch.Tensor:
        if isinstance(hue, (int, float)):
            rad = math.radians(hue)
            return torch.tensor([math.cos(rad), math.sin(rad)],
                                device=x.device, dtype=x.dtype)
        rad = torch.deg2rad(hue)
        return torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)

    sh_vec = _hue_vec(sh_hue)
    hi_vec = _hue_vec(hi_hue)

    ab_offset = w_sh * sh_amt * 30.0 * sh_vec + w_hi * hi_amt * 30.0 * hi_vec
    lab = lab.clone()
    lab[..., 1:3] = lab[..., 1:3] + ab_offset
    return x + (lab_to_rgb(lab) - lab_to_rgb(lab0))


def grade(x: torch.Tensor, values: dict[str, float | torch.Tensor]) -> torch.Tensor:
    """Alle manuellen Regler anwenden. Kein finales clamp."""
    y = x
    if not _is_default(values, ["exposure"]):
        y = _exposure(y, values.get("exposure", 0.0))
    if not _is_default(values, ["temperature", "tint"]):
        y = _temp_tint(y, values.get("temperature", 0.0), values.get("tint", 0.0))
    lgg_names = [f"{p}_{c}" for p in ("lift", "gamma", "gain") for c in ("m", "r", "g", "b")]
    if not _is_default(values, lgg_names):
        y = _lgg(y, values)
    if not _is_default(values, ["contrast", "pivot", "highlight_rolloff"]):
        y = _contrast(y, values)
    if not _is_default(values, ["highlights", "shadows", "whites", "blacks"]):
        y = _tone_zones(y, values)
    if not _is_default(values, ["saturation"]):
        y = _saturation(y, values.get("saturation", 1.0))
    if not _is_default(values, ["vibrance"]):
        y = _vibrance(y, values.get("vibrance", 0.0))
    sel_names = [f"sel_{b}_{p}" for b in _SEL_BANDS for p in ("hue", "sat", "light")]
    if not _is_default(values, sel_names):
        y = _selective_color(y, values)
    if not _is_default(values, ["shadow_hue", "shadow_amount", "highlight_hue", "highlight_amount"]):
        y = _split_toning(y, values)
    if not _is_default(values, all_cr_names()):
        y = _color_replace_apply(y, values)
    return y
