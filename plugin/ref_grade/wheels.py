"""Colour wheels (Lift / Gamma / Gain) ↔ the core's per-channel LGG sliders. No Qt, no krita.

A wheel point (u, v) in the unit disk is a hue direction and strength. It becomes per-channel
offsets that sum to zero, so a wheel tints without changing brightness; brightness is the
separate master slider (lift_m / gamma_m / gain_m).

    w_c = r · cos(θ − φ_c),  φ = (0°, 120°, 240°) for (R, G, B),  Σ w_c = 0
    lift_c  = LIFT_RANGE · w_c                   (additive, core range −0.5…0.5)
    gamma_c = 2 ** (GAMMA_RANGE · w_c)           (multiplicative around 1)
    gain_c  = 2 ** (GAIN_RANGE · w_c)

`to_point` inverts this for values coming from FIT: the zero-sum part becomes the point, the
common part that a wheel cannot show is returned as `rest` and added back by `to_values`, so
a round trip is exact and no fitted value is lost.
"""

from __future__ import annotations

import math

KINDS = ("lift", "gamma", "gain")
CHANNELS = ("r", "g", "b")
PHASE = (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
LIFT_RANGE, GAMMA_RANGE, GAIN_RANGE = 0.15, 0.5, 0.4


def _w(u: float, v: float) -> tuple[float, float, float]:
    r = min(1.0, math.hypot(u, v))
    t = math.atan2(v, u)
    return tuple(r * math.cos(t - p) for p in PHASE)


def _encode(kind: str, w: float) -> float:
    if kind == "lift":
        return LIFT_RANGE * w
    return 2.0 ** ((GAMMA_RANGE if kind == "gamma" else GAIN_RANGE) * w)


def _decode(kind: str, value: float) -> float:
    if kind == "lift":
        return value / LIFT_RANGE
    return math.log2(max(value, 1e-6)) / (GAMMA_RANGE if kind == "gamma" else GAIN_RANGE)


def to_values(kind: str, u: float, v: float, master: float,
              rest: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, float]:
    """Wheel point + master (+ rest from `to_point`) → the four core slider values."""
    out = {f"{kind}_m": master}
    for c, w, extra in zip(CHANNELS, _w(u, v), rest):
        out[f"{kind}_{c}"] = _encode(kind, w + extra)
    return out


def to_point(kind: str, values: dict[str, float]) -> tuple[float, float, float, tuple]:
    """Core values → (u, v, master, rest); missing channels count as neutral."""
    neutral = 0.0 if kind == "lift" else 1.0
    w = [_decode(kind, values.get(f"{kind}_{c}", neutral)) for c in CHANNELS]
    mean = sum(w) / 3.0
    z = [x - mean for x in w]                      # zero-sum part = what the wheel shows
    u = sum(zc * math.cos(p) for zc, p in zip(z, PHASE)) * 2.0 / 3.0
    v = sum(zc * math.sin(p) for zc, p in zip(z, PHASE)) * 2.0 / 3.0
    r = math.hypot(u, v)
    if r > 1.0:                                     # outside the disk: clip, keep the excess
        u, v = u / r, v / r
    shown = _w(u, v)
    rest = tuple(x - s for x, s in zip(w, shown))
    return u, v, values.get(f"{kind}_m", neutral), rest


def hue_rgb(angle: float) -> tuple[int, int, int]:
    """Display colour at `angle` on the wheel — the same mapping as the tint it produces."""
    w = [math.cos(angle - p) for p in PHASE]
    return tuple(round(255 * (0.5 + 0.5 * x)) for x in w)
