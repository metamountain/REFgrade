"""PCHIP (shape-preserving cubic) in plain Python — no numpy, no Qt — for the curve editor.

Same algorithm as core/refgrade/curves.py (`pchip_slopes`, `cyclic_points`, `pchip_eval`),
which itself matches scipy's PchipInterpolator: the curve never overshoots its points.
Rewritten 2026-10-08 after the first version swung far outside the points (wrong tangent
sign on falling segments, slopes not scaled by the interval width).
"""

from __future__ import annotations


def pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Tangents: weighted harmonic mean inside (0 at a local extremum), 3-point edge formula."""
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


def cyclic_points(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Points repeated at x−1 and x+1 (seam duplicates dropped): a periodic curve."""
    out_x: list[float] = []
    out_y: list[float] = []
    for shift in (-1.0, 0.0, 1.0):
        for px, py in zip(xs, ys):
            x = float(px) + shift
            if out_x and abs(x - out_x[-1]) < 1e-9:
                continue
            out_x.append(x)
            out_y.append(float(py))
    return out_x, out_y


def _hermite(xs: list[float], ys: list[float], m: list[float], px: float) -> float:
    px = min(max(px, xs[0]), xs[-1])                  # constant outside the points
    k = 0
    while k < len(xs) - 2 and px >= xs[k + 1]:
        k += 1
    h = xs[k + 1] - xs[k]
    t = (px - xs[k]) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * ys[k] + (t3 - 2 * t2 + t) * h * m[k]
            + (-2 * t3 + 3 * t2) * ys[k + 1] + (t3 - t2) * h * m[k + 1])


def pchip_eval(x_pts: list[float], y_pts: list[float], px: float) -> float:
    """Value of the curve at px."""
    xs, ys = list(x_pts), list(y_pts)
    return _hermite(xs, ys, pchip_slopes(xs, ys), px)


def pchip_curve(x_pts: list[float], y_pts: list[float], n: int = 256) -> list[float]:
    """n values over [0, 1]."""
    xs, ys = list(x_pts), list(y_pts)
    m = pchip_slopes(xs, ys)
    return [_hermite(xs, ys, m, i / (n - 1) if n > 1 else 0.5) for i in range(max(n, 1))]


def pchip_cyclic_eval(x_pts: list[float], y_pts: list[float], px: float) -> float:
    """Value of the periodic (hue) curve at px; 0 and 1 are the same point."""
    xs, ys = cyclic_points(x_pts, y_pts)
    return _hermite(xs, ys, pchip_slopes(xs, ys), px % 1.0)


def pchip_cyclic_curve(x_pts: list[float], y_pts: list[float], n: int = 256) -> list[float]:
    """n values of the periodic curve over [0, 1] (the extended points cover 0..1 fully)."""
    xs, ys = cyclic_points(x_pts, y_pts)
    m = pchip_slopes(xs, ys)
    return [_hermite(xs, ys, m, i / (n - 1) if n > 1 else 0.5) for i in range(max(n, 1))]
