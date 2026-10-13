"""Kurven: PCHIP-Interpolation (Fritsch-Carlson), apply, validate, is_identity."""

from __future__ import annotations

import torch

_EPS = 1e-6
_CHANNELS = ("rgb", "r", "g", "b")
HUE_KEYS = ("hue_hue", "hue_sat", "hue_lum", "lum_sat", "sat_sat")
_ALL_KEYS = _CHANNELS + HUE_KEYS
_IDENTITY = [[0.0, 1.0]]


def validate(curves: dict) -> None:
    """Prüfe das Kurven-Format (RGB + Hue); wirft ValueError bei Verletzung.

    Args:
        curves: {"rgb": [[x,y],…], "r": […], "g": […], "b": […],
                 "hue_hue": […], "hue_sat": […], "hue_lum": […],
                 "lum_sat": […], "sat_sat": […]}.

    Raises:
        ValueError: wenn Format, Punktezahl, Monotonie oder Wertebereich verletzt.
    """
    if not isinstance(curves, dict):
        raise ValueError("Curves must be a dict")
    for ch in curves:
        if ch not in _ALL_KEYS:
            raise ValueError(f"Unknown curve channel: {ch!r}")
    for ch, pts in curves.items():
        if not isinstance(pts, (list, tuple)):
            raise ValueError(f"Channel {ch!r} must be a list of [x, y] pairs")
        n = len(pts)
        if n < 2:
            raise ValueError(f"Channel {ch!r} needs at least 2 points, got {n}")
        if n > 16:
            raise ValueError(f"Channel {ch!r} allows at most 16 points, got {n}")
        for i, p in enumerate(pts):
            if len(p) != 2:
                raise ValueError(f"Channel {ch!r} point {i} must have exactly 2 values")
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"Channel {ch!r} point {i}: x={x} outside [0, 1]")
            if not (0.0 <= y <= 1.0):
                raise ValueError(f"Channel {ch!r} point {i}: y={y} outside [0, 1]")
        for i in range(1, n):
            if pts[i][0] <= pts[i - 1][0]:
                raise ValueError(f"Channel {ch!r}: x values must be strictly increasing")


def is_identity(curves: dict | None) -> bool:
    """True wenn alle vorhandenen Kanäle identisch sind oder fehlen.

    RGB: genau [[0,0],[1,1]]. Hue: alle y = 0,5.
    """
    if not curves:
        return True
    for ch, pts in curves.items():
        if ch in HUE_KEYS:
            for p in pts:
                if abs(float(p[1]) - 0.5) > _EPS:
                    return False
        else:
            if len(pts) != 2:
                return False
            if abs(pts[0][0] - 0.0) > _EPS or abs(pts[0][1] - 0.0) > _EPS:
                return False
            if abs(pts[1][0] - 1.0) > _EPS or abs(pts[1][1] - 1.0) > _EPS:
                return False
    return True


def pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Tangents of the shape-preserving cubic (same algorithm as scipy's PchipInterpolator):
    weighted harmonic mean inside (0 at a local extremum), 3-point edge formula at the ends."""
    n = len(xs)
    h = [xs[k + 1] - xs[k] for k in range(n - 1)]
    d = [(ys[k + 1] - ys[k]) / h[k] for k in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    m = [0.0] * n
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def edge(h0: float, h1: float, d0: float, d1: float) -> float:
        t = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if (t > 0) != (d0 > 0) or t == 0 or d0 == 0:
            return 0.0
        if (d0 > 0) != (d1 > 0) and abs(t) > 3 * abs(d0):
            return 3 * d0
        return t

    m[0] = edge(h[0], h[1], d[0], d[1])
    m[-1] = edge(h[-1], h[-2], d[-1], d[-2])
    return m


def cyclic_points(pts: list) -> tuple[list[float], list[float]]:
    """Points repeated at x−1 and x+1 (duplicates at the seam dropped): a periodic curve."""
    xs: list[float] = []
    ys: list[float] = []
    for shift in (-1.0, 0.0, 1.0):
        for px, py in pts:
            x = float(px) + shift
            if xs and abs(x - xs[-1]) < 1e-9:
                continue
            xs.append(x)
            ys.append(float(py))
    return xs, ys


def pchip_eval(x: torch.Tensor, xs: list[float], ys: list[float]) -> torch.Tensor:
    """Evaluate the cubic Hermite PCHIP at x (any shape); constant outside [xs[0], xs[-1]]."""
    m = pchip_slopes(xs, ys)
    # One host→device copy instead of three: each copy waits for a GPU slot, which is
    # what made curve drags lag while another process kept the GPU busy.
    xt, yt, mt = torch.tensor([xs, ys, m], device=x.device, dtype=x.dtype)
    xc = x.clamp(xs[0], xs[-1])
    k = (torch.searchsorted(xt, xc.contiguous(), right=True) - 1).clamp(0, len(xs) - 2)
    h = xt[k + 1] - xt[k]
    t = (xc - xt[k]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * yt[k] + (t3 - 2 * t2 + t) * h * mt[k]
            + (-2 * t3 + 3 * t2) * yt[k + 1] + (t3 - t2) * h * mt[k + 1])


def _apply_channel(x_ch: torch.Tensor, pts: list) -> torch.Tensor:
    """PCHIP curve on one channel; constant outside the first/last point."""
    if len(pts) == 2 and abs(pts[0][0]) < _EPS and abs(pts[0][1]) < _EPS        and abs(pts[1][0] - 1.0) < _EPS and abs(pts[1][1] - 1.0) < _EPS:
        return x_ch
    return pchip_eval(x_ch, [float(p[0]) for p in pts], [float(p[1]) for p in pts])


def apply(x: torch.Tensor, curves: dict) -> torch.Tensor:
    """Kurven anwenden: zuerst rgb auf alle Kanäle, dann r, g, b je Kanal.

    Args:
        x: [B,H,W,3] sRGB 0..1.
        curves: {"rgb": [[x,y],…], "r": […], "g": […], "b": […]}.

    Returns:
        [B,H,W,3] sRGB 0..1 (geclampt).
    """
    if not curves:
        return x
    out = x
    if "rgb" in curves:
        out = _apply_channel(out, curves["rgb"])   # same curve on every channel, one call
    for i, ch in enumerate(("r", "g", "b")):
        if ch in curves:
            out = out.clone()
            out[..., i] = _apply_channel(out[..., i:i+1], curves[ch])[..., 0]
    return out.clamp(0.0, 1.0)
