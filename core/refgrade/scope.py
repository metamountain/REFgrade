"""Vectorscope: Cb/Cr-Histogramm nach Rec.709 (gamma, wie Resolve)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LO_THRESH = 0.5 / 255.0
_HI_THRESH = 254.5 / 255.0


SCOPE_SIDE = 512


def scope_img(img: torch.Tensor) -> torch.Tensor:
    """Image for scopes/histograms, at most SCOPE_SIDE px (area-averaged).

    A 256-level parade, 192² vectorscope or 64/128-bin histogram gains nothing from more
    pixels, but at 1280 px they cost ~0.3–1 s per preview under GPU load (2026-10-11).
    """
    h, w = img.shape[1], img.shape[2]
    if max(h, w) <= SCOPE_SIDE:
        return img
    k = SCOPE_SIDE / max(h, w)
    return F.interpolate(img.permute(0, 3, 1, 2), size=(max(1, round(h * k)), max(1, round(w * k))),
                         mode="area").permute(0, 2, 3, 1)


def clipping(x: torch.Tensor) -> dict[str, float]:
    """Per-channel clipping fractions (highlight and shadow).

    Args:
        x: [B,H,W,3] sRGB 0..1.

    Returns:
        {"r_hi": f, "r_lo": f, "g_hi": f, "g_lo": f, "b_hi": f, "b_lo": f}
        where f is the fraction of pixels at or beyond the threshold.
    """
    result: dict[str, float] = {}
    total = x.shape[1] * x.shape[2]
    for i, ch in enumerate(("r", "g", "b")):
        c = x[0, :, :, i]
        result[f"{ch}_hi"] = (c >= _HI_THRESH).float().sum().item() / total
        result[f"{ch}_lo"] = (c <= _LO_THRESH).float().sum().item() / total
    return result


def clip_overlay(x: torch.Tensor) -> torch.Tensor:
    """Clipping-Ansicht: überbelichtete Pixel rot, unterbelichtete blau, Rest unverändert.

    Args:
        x: [B,H,W,3] sRGB 0..1.

    Returns:
        Same shape. Pixel mit einem Kanal >= _HI_THRESH → (1,0,0);
        sonst Pixel mit einem Kanal <= _LO_THRESH → (0,0,1); alle anderen unverändert.
    """
    hi = (x >= _HI_THRESH).any(dim=-1, keepdim=True)  # [B,H,W,1]
    lo = (x <= _LO_THRESH).any(dim=-1, keepdim=True)  # [B,H,W,1]
    red = torch.tensor([1.0, 0.0, 0.0], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    blue = torch.tensor([0.0, 0.0, 1.0], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    return torch.where(hi, red, torch.where(lo, blue, x))


def rgb_parade(x: torch.Tensor, cols: int = 256, levels: int = 256) -> dict[str, torch.Tensor]:
    """RGB Parade: per-channel grid levels × cols, normalized to max 1.

    For each channel, the image width is split into `cols` column groups.
    Cell (l, c) = fraction of pixels in column-group c with value in level l.
    Row 0 = value 0 (bottom). Each channel is normalized so its maximum is 1.

    Args:
        x: [1, H, W, 3] sRGB 0..1.
        cols: Number of column groups (default 256).
        levels: Number of value levels (default 256).

    Returns:
        {"r": [levels, cols], "g": [levels, cols], "b": [levels, cols]} tensors.
    """
    result: dict[str, torch.Tensor] = {}
    W = x.shape[2]
    col_idx = torch.arange(W, device=x.device)
    group = (col_idx * cols // W).long()  # [W]

    for i, key in enumerate(("r", "g", "b")):
        ch = x[0, :, :, i]  # [H, W]
        H = ch.shape[0]
        level = (ch * levels).long().clamp(0, levels - 1)  # [H, W]
        pixel_group = group.unsqueeze(0).expand(H, W)  # [H, W]
        flat_idx = level * cols + pixel_group
        hist = torch.bincount(flat_idx.flatten(), minlength=levels * cols).float().view(levels, cols)
        max_val = hist.max()
        if max_val > 0:
            hist = hist / max_val
        result[key] = hist
    return result


def vectorscope(rgb: torch.Tensor, bins: int = 192) -> list[list[float]]:
    """Vectorscope-Daten aus sRGB (gamma, nicht linear, wie Resolve).

    Cb/Cr nach Rec.709 aus sRGB-gamma-Werten, Bereich je −0,5..0,5
    auf bins × bins. Zählung normiert auf Maximum = 1, mit log1p gestaucht.
    Zeilen von oben (+Cr) nach unten.

    Args:
        rgb: [B,H,W,3] sRGB 0..1.
        bins: Gitterauflösung (Default 192).

    Returns:
        Liste von bins Zeilen, je bins Werte (0..1). Zeile 0 = +Cr (oben).
    """
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Rec.709 Luma (gamma-kodiert, wie Resolve)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Cb/Cr mit 0.5-Faktor → Bereich −0,5..0,5
    cb = 0.5 * (b - y) / (1.0 - 0.0722)
    cr = 0.5 * (r - y) / (1.0 - 0.2126)

    # Auf 0..1 mappen (für Histogramm)
    cb = (cb + 0.5).clamp(0.0, 1.0)
    cr = (cr + 0.5).clamp(0.0, 1.0)

    # 2D-Histogramm via bincount
    cb_idx = (cb * bins).long().clamp(0, bins - 1)
    cr_idx = (cr * bins).long().clamp(0, bins - 1)
    flat_idx = cb_idx.flatten() * bins + cr_idx.flatten()
    hist = torch.bincount(flat_idx, minlength=bins * bins).float().view(bins, bins)
    # hist[i_cb][i_cr], i_cb=0 → cb=0 (−0,5), i_cr=0 → cr=0 (−0,5)

    # log1p-Stauchung
    hist = torch.log1p(hist)

    # Normierung: Maximum = 1
    max_val = hist.max()
    if max_val > 0:
        hist = hist / max_val

    # Zeilen von oben (+Cr) nach unten:
    # Zeile 0 = +Cr → cr_index = bins-1
    # Spalte 0 = −Cb → cb_index = 0
    # row i = hist[:, bins-1-i]; one transfer instead of 192 row-wise GPU syncs
    # (2026-10-11: 99 ms -> ~1 ms per preview)
    return hist.T.flip(0).cpu().tolist()


PARADE_COLORS = {"r": (1.0, 0.16, 0.10), "g": (0.15, 1.0, 0.30), "b": (0.20, 0.38, 1.0)}
WAVE_COLOR = (0.62, 0.66, 0.72)      # kühles Grau: dichte Stellen glühen weiß (Kdenlive „White“)


def luma_waveform(x: torch.Tensor, cols: int = 256, levels: int = 256) -> torch.Tensor:
    """Luma-Waveform (Rec.709, gamma): [levels, cols], wie ein Parade-Kanal (Zeile 0 = Wert 0)."""
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return rgb_parade(y[..., None].expand(*y.shape, 3), cols, levels)["r"]


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gauss-Unschärfe für [1,1,H,W]."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32, device=x.device) / sigma) ** 2)
    k = (k / k.sum()).to(x.dtype)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def parade_neon(counts: torch.Tensor, color: tuple[float, float, float],
                glow: float = 2.0, white: float = 0.95) -> torch.Tensor:
    """Ein Parade-Kanal als Neon-Bild; siehe `parade_neon_all` (ein Aufruf für alle drei)."""
    return parade_neon_all({"x": counts}, {"x": color}, glow, white)["x"]


def parade_neon_all(parade: dict[str, torch.Tensor], colors: dict | None = None,
                    glow: float = 2.0, white: float = 0.95) -> dict[str, torch.Tensor]:
    """Parade als Neon-/Plasma-Bilder (DaVinci-artig), Nutzerwahl 2026-10-11:
    glow 2 („etwas weniger Bums“ als 6) + „Plasma-Weiß“.

    - Dichte relativ zum 99,5-%-Perzentil der belegten Zellen (ein Ausreißer setzt nicht die
      Skala), Wurzel-Kurve: dünne Bereiche bleiben dunkel, dichte leuchten heller.
    - Schein nur aus dichten Bereichen (v²) in drei Weiten; die weite auf ¼-Auflösung (spart Zeit).
    - Weiß nur aus dem scharfen Kern (v > 0,6): die dichten Spuren glühen weiß-heiß, linienscharf;
      der Schein färbt nur, er macht nichts weiß (keine Information wird verwischt).
    Alle Kanäle in einem Durchgang ([C,1,H,W]); ~2–5 ms auf der GPU.

    Returns:
        {key: [levels, cols, 3] RGB 0..1, Zeile 0 oben = Wert 1}.
    """
    colors = colors or PARADE_COLORS
    keys = list(parade)
    c0 = torch.stack([parade[k].float() for k in keys])            # [C, L, W]
    flat = c0.flatten(1)
    refs = []
    for row in flat:
        pos = row[row > 0]
        if pos.numel() == 0:
            refs.append(torch.ones((), device=row.device))
            continue
        k = max(1, int(round(0.995 * pos.numel())))
        refs.append(pos.kthvalue(k).values)                        # schneller als quantile
    ref = torch.stack(refs).view(-1, 1, 1).clamp_min(1e-12)
    v = (c0 / ref).clamp(0, 4).sqrt().flip(1)[:, None]               # [C,1,L,W], oben = hell
    d = v * v
    far = F.interpolate(_gauss_blur(F.avg_pool2d(d, 4), 4.0), size=d.shape[-2:], mode="bilinear",
                        align_corners=False)
    bloom = 0.5 * _gauss_blur(d, 2.0) + 0.3 * _gauss_blur(d, 6.0) + 0.2 * far
    e = 0.9 * v ** 1.5 + glow * bloom
    col = torch.tensor([colors[k] for k in keys], dtype=v.dtype, device=v.device).view(-1, 3, 1, 1)
    base = 1.0 - torch.exp(-e * 1.6)
    w = white * (1.0 - torch.exp(-4.0 * torch.relu(v - 0.6)))
    rgb = (base * col + w * (1.0 - col)).clamp(0, 1)                 # [C,3,L,W]
    return {k: rgb[i].permute(1, 2, 0) for i, k in enumerate(keys)}
