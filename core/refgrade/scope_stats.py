"""Scope-Zahlen für eine KI (MCP): Perzentile, Clipping, Farbstich je Zone, Sättigung, Histogramm.

Eine Maschine liest Zahlen besser als Scope-Bilder; das hier ist die Zahlen-Fassung von
Histogramm, Parade und Vectorscope (K32a, 2026-10-11).
"""

from __future__ import annotations

import torch

from .color import rgb_to_lab
from .scope import clipping

_ZONES = (("shadows", 0.0, 33.0), ("mids", 33.0, 66.0), ("highlights", 66.0, 100.01))


def _r(v: float) -> float:
    return round(float(v), 3)


def stats(img: torch.Tensor) -> dict:
    """img [1,H,W,3] sRGB 0..1 → Kennzahlen (alle Werte 0..1 außer Lab)."""
    x = img.float().clamp(0.0, 1.0)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    out: dict = {"channels": {}}
    for name, ch in (("r", x[..., 0]), ("g", x[..., 1]), ("b", x[..., 2]), ("luma", y)):
        flat = ch.flatten()
        q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], device=flat.device))
        out["channels"][name] = {"p01": _r(q[0]), "p50": _r(q[1]), "p99": _r(q[2]),
                                 "mean": _r(flat.mean())}
    out["clip"] = {k: _r(v) for k, v in clipping(x).items()}
    lab = rgb_to_lab(x)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    cast = {}
    for zone, lo, hi in _ZONES:
        m = (L >= lo) & (L < hi)
        frac = float(m.float().mean())
        cast[zone] = ({"a": _r(a[m].mean()), "b": _r(b[m].mean()), "share": _r(frac)}
                      if frac > 0 else {"a": 0.0, "b": 0.0, "share": 0.0})
    cast["all"] = {"a": _r(a.mean()), "b": _r(b.mean()), "share": 1.0}
    out["cast"] = cast
    out["saturation"] = _r(torch.sqrt(a * a + b * b).mean())
    h = torch.histc(y.flatten(), bins=16, min=0.0, max=1.0)
    out["hist16"] = [round(float(v), 4) for v in (h / h.sum().clamp_min(1.0))]
    return out
