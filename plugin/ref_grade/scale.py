"""Responsive scale unit and layout geometry: one source of truth.

No Qt import — pure arithmetic on window dimensions.
"""

from __future__ import annotations


def unit(W: int, H: int) -> float:
    """Scale unit s = min(H/5, W/6). On portrait monitors W/6 limits the unit."""
    return min(H / 5.0, W / 6.0)


def base_px(s: float) -> int:
    """Base font size in pixels, clamped to [9, 16] (user 2026-10-10: 0.06 was too big)."""
    return max(9, min(16, round(s * 0.048)))


def role_px(role: str, s: float) -> int:
    """Font pixel size for a role: small, base, title, big."""
    b = base_px(s)
    if role == "small":
        return max(9, round(b * 0.85))
    if role == "title":
        return round(b * 1.1)
    if role == "big":
        return round(b * 2)
    return b  # "base"


def row_h(s: float) -> int:
    """Height of a slider row, header row, or small button."""
    return round(base_px(s) * 1.7)


def button_h(s: float) -> int:
    """Height of a regular button (RESET, CANCEL, OK, LUT row)."""
    return round(base_px(s) * 1.9)


def value_w(s: float) -> int:
    """Width of a value display field in a slider row."""
    return round(base_px(s) * 3.4)


def layout_rects(W: int, H: int, mode: str, iw: int = 0, ih: int = 0) -> dict:
    """Compute pixel rects for the 4 layout blocks and their 4 squares.

    Returns a dict with keys:
      panel:   (x, y, w, h) — control column
      strip:   (x, y, w, h) — wheel strip (4 squares)
      bottom:  (x, y, w, h) — scope buttons + histogram/parade
      view:    (x, y, w, h) — image view
      squares: list of 4 (x, y, w, h) — VS, Lights, Mids, Lows
    """
    s = unit(W, H)
    m = round(s * 0.04)
    si = round(s)
    s3 = round(3 * s)
    s4 = round(4 * s)

    if mode == "V":
        panel = (m, m, s3 - m, s4 - m)
        strip = (s3, 0, si, s4)
        bottom = (m, s4, s4 - m, si - m)
        view = (s4 + m, 0, W - s4 - m, H)
        # Squares top-to-bottom: VS, Lights, Mids, Lows
        squares = [
            (s3, 0, si, si),
            (s3, si, si, si),
            (s3, 2 * si, si, si),
            (s3, 3 * si, si, si),
        ]
    else:  # H
        panel = (m, m, s3 - m, s4 - m)
        bottom = (m, s4, s4 - m, si - m)

        if iw > 0 and ih > 0:
            # K25h: view fills width, height from aspect ratio (clamped)
            vw = W - s3 - m
            fh = vw * ih / iw
            view_h = max(H - 2 * si, min(round(fh), H - si))
            view = (s3 + m, 0, vw, view_h)
            b = H - view_h
            sq = max(1, b - m)
            # Squares right-aligned to W, in the band below the view
            squares = [
                (W - sq, view_h, sq, sq),
                (W - 2 * sq, view_h, sq, sq),
                (W - 3 * sq, view_h, sq, sq),
                (W - 4 * sq, view_h, sq, sq),
            ]
            strip = (s4, view_h, W - s4, b)
        else:
            # K25b: no image size, fixed geometry
            view_h = (H - si) if H > 5 * s else s4
            view = (s3 + m, 0, W - s3 - m, view_h)
            strip = (W - s4, s4, s4, si)
            squares = [
                (W - s4, s4, si, si),
                (W - s3, s4, si, si),
                (W - round(2 * s), s4, si, si),
                (W - si, s4, si, si),
            ]

    return {"panel": panel, "strip": strip, "bottom": bottom, "view": view,
            "squares": squares}


def choose_mode(W: int, H: int, iw: int, ih: int, current: str | None) -> str | None:
    """Pick V or H by which view rect fits the image at the larger scale.

    Returns None when W/H ≤ 0 or iw/ih ≤ 0 (no decision possible).
    On a tie (±2 %) the current mode is kept.
    """
    if W <= 0 or H <= 0 or iw <= 0 or ih <= 0:
        return None
    rects_v = layout_rects(W, H, "V")
    rects_h = layout_rects(W, H, "H", iw, ih)
    vw, vh = rects_v["view"][2], rects_v["view"][3]
    hw, hh = rects_h["view"][2], rects_h["view"][3]
    scale_v = min(vw / iw, vh / ih)
    scale_h = min(hw / iw, hh / ih)
    cur = current or "V"
    if scale_v >= scale_h * 1.02:
        return "V"
    if scale_h >= scale_v * 1.02:
        return "H"
    return cur
