"""AUTO mit klassischen Literaturverfahren (Vergleich zu auto2, Nutzer 2026-10-13: „Literatur statt
ausdenken“). Keine eigenen Ziele, nur die veröffentlichten Verfahren mit ihren Standardparametern:

- Weißabgleich: Shades-of-Gray, Finlayson & Trezzi, „Shades of Gray and Colour Constancy“, CIC 2004:
  Lichtfarbe e_c = (mean(I_c^p))^(1/p) in linearem RGB, p = 6 (p = 1 = Grauwelt). Korrektur von Kries.
- Belichtung: Reinhard et al., „Photographic Tone Reproduction for Digital Images“, SIGGRAPH 2002:
  log-mittlere Leuchtdichte L̄ = exp(mean(log(δ + L))), Schlüssel a = 0,18 (Mittelgrau) →
  Belichtung in Blenden = log2(a / L̄).
- Tonkurve: globale Contrast Limited Histogram Equalisation (Pizer et al. 1987; Mustafa & Finlayson,
  „Contrast limited histogram equalisation revisited“, UEA 2022): Histogramm der Luma, jedes Fach auf
  CLIP · Mittelwert gekappt, Überschuss gleichmäßig verteilt, kumuliert = Kurve (Steigung ≤ CLIP).
"""
from __future__ import annotations

import math

import torch

from .color import luma, srgb_to_linear
from .sections import SECTIONS
from .sliders import SLIDERS
from .whitebalance import _to_values

AUTO_SECTIONS: tuple[str, ...] = ("color", "tone", "curves")
SOG_P = 6.0
KEY = 0.18
CLIP = 2.0
_BINS = 256
_DEF = {s.name: s.default for s in SLIDERS}


def _clamp(name: str, v: float) -> float:
    s = next(x for x in SLIDERS if x.name == name)
    return max(s.min, min(s.max, v))


def _color(x: torch.Tensor, p: float = SOG_P) -> dict:
    lin = srgb_to_linear(x.reshape(-1, 3).clamp(0.0, 1.0)).double()
    e = lin.pow(p).mean(0).pow(1.0 / p) if p != 1.0 else lin.mean(0)
    return {**{n: _DEF[n] for n in SECTIONS["color"]}, **_to_values(e)}


def _tone(x: torch.Tensor) -> dict:
    L = srgb_to_linear(luma(x).clamp(0.0, 1.0)).flatten().double()
    lavg = math.exp(float(torch.log(1e-4 + L).mean()))
    out = {n: _DEF[n] for n in SECTIONS["tone"]}
    out["exposure"] = round(_clamp("exposure", math.log2(KEY / max(lavg, 1e-6))), 4)
    return out


def _curves(x: torch.Tensor) -> dict:
    Y = luma(x).clamp(0.0, 1.0).flatten()
    h = torch.histc(Y.float(), bins=_BINS, min=0.0, max=1.0).double()
    h = h / h.sum()
    limit = CLIP / _BINS
    for _ in range(8):                      # Überschuss verteilen, bis nichts mehr übersteht
        excess = (h - limit).clamp_min(0).sum()
        h = h.clamp(max=limit) + excess / _BINS
        if excess < 1e-9:
            break
    cdf = torch.cumsum(h, 0)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    xs = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    pts = [[0.0, 0.0]] + [[v, round(float(cdf[min(_BINS - 1, int(v * _BINS))]), 4)] for v in xs] + [[1.0, 1.0]]
    for i in range(1, len(pts)):           # streng steigend halten (PCHIP)
        pts[i][1] = max(pts[i][1], pts[i - 1][1] + 1e-3) if i < len(pts) - 1 else 1.0
    return {"curves": {"rgb": pts}}


def auto(chain_in: torch.Tensor, section: str, sog_p: float = SOG_P) -> dict:
    if section not in AUTO_SECTIONS:
        raise ValueError(f"AUTO nicht verfügbar für Bereich {section!r}")
    x = chain_in.clamp(0.0, 1.0)
    if section == "color":
        return _color(x, sog_p)
    if section == "tone":
        return _tone(x)
    return _curves(x)
