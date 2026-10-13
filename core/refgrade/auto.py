"""AUTO je Kapitel, Neuentwurf 2026-10-13 (Opus; alte Formeln in _attic/auto_alt_2026-10-13): Regler werden gegen ein *gemessenes* Ziel gefittet.

Eingang ist das Bild, wie es beim Bereich in der Kette ankommt (`session.chain_input`) — was davor
schon korrigiert wurde, ist enthalten, also korrigiert keine zweite AUTO dasselbe noch einmal.

Ziele (reports/auto_fivek.md, Tuning auf den geraden FiveK-Paaren, Prüfung auf den ungeraden):
- Helligkeit: Luma-Quantile Q liegen im Band, in dem Fotograf:innen sie ablegen (BAND_LO..BAND_HI,
  5.–95. Perzentil der Expertenbilder je Quantil). Innerhalb des Bands: keine Änderung.
- Farbe: Stich der nahezu neutralen Pixel über CAST_OK wird auf CAST_OK zurückgenommen.
Alle Fits: Adam auf den Reglern des Bereichs, Strafterm zurück zum Standard (so bleibt ein schon
gutes Bild unverändert), Ergebnis auf den Reglerbereich geklemmt.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color import luma, rgb_to_lab
from .grade import grade
from .sections import SECTIONS
from .sliders import SLIDERS

# Bewusst nicht automatisch (gemessen 2026-10-13, Schaden auf schon guten Expertenbildern):
# - Farbräder: ΔE 5–11 (Zonenmittel ≠ Farbstich, eine Wiese in den Mitten ist grün).
# - RGB: Kanäle auf die Luma-Quantile → ΔE bis 30 (ein blauer Himmel soll mehr Blau haben);
#   nur Schwarz-/Weißpunkt je Kanal → kein Nutzen (Median 0,00) und bis ΔE 9 Schaden.
# Beides sind Gestaltungswerkzeuge ohne messbares Ziel.
AUTO_SECTIONS: tuple[str, ...] = ("color", "tone", "curves")

Q = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
BAND_LO = (0.050, 0.114, 0.208, 0.573, 0.650, 0.681, 0.746)   # 5. Perzentil der Experten
BAND_HI = (0.409, 0.536, 0.622, 0.682, 0.748, 0.825, 0.909)   # 95. Perzentil
CAST_OK = 2.5            # |(a*, b*)| der Unbunten, den Expert:innen stehen lassen (Median 2,2, p90 3,1)
_N_PIX = 16_384
STAT_SIDE = 512          # Statistikbild (Worker skaliert die Quelle darauf, Kette davor eingerechnet)
_STEPS = 120
_LR = 0.03
_REG = 1e-2           # Bench 2026-10-13: 2e-3 ließ Kontrast 0,35 zu (Schaden ΔE 15,8)

_SLIDER = {s.name: s for s in SLIDERS}
_DEF = {s.name: s.default for s in SLIDERS}


def _pixels(img: torch.Tensor) -> torch.Tensor:
    """Deterministische Teilmenge [1,1,N,3] (gleicher Schritt, kein Zufall)."""
    flat = img.reshape(-1, 3)
    step = max(1, flat.shape[0] // _N_PIX)
    return flat[::step][:_N_PIX].reshape(1, 1, -1, 3).float()


def _band_loss(Y: torch.Tensor) -> torch.Tensor:
    q = torch.quantile(Y.flatten(), torch.tensor(Q, device=Y.device))
    lo = torch.tensor(BAND_LO, device=Y.device)
    hi = torch.tensor(BAND_HI, device=Y.device)
    return (F.relu(lo - q) ** 2 + F.relu(q - hi) ** 2).sum()


# Engere Grenzen als der Regler selbst: AUTO soll sanft eingreifen (Bench 2026-10-13: Kontrast
# 0,35 + Belichtung +0,8 auf einem bewusst dunklen Expertenbild, ΔE 15,8).
_AUTO_RANGE = {"contrast": (0.75, 1.5)}


def _fit(names: list[str], render, loss_fn, device) -> dict[str, float]:
    """Adam über die genannten Regler; render(values) → Bild, loss_fn(Bild) → Skalar."""
    d0 = torch.tensor([_DEF[n] for n in names], device=device)
    span = torch.tensor([_SLIDER[n].max - _SLIDER[n].min for n in names], device=device)
    lo = torch.tensor([_AUTO_RANGE.get(n, (_SLIDER[n].min,))[0] for n in names], device=device)
    hi = torch.tensor([_AUTO_RANGE.get(n, (None, _SLIDER[n].max))[1] for n in names], device=device)
    # ein Hauch neben dem Standard starten: rgb_grade/grade überspringen Regler, die exakt auf dem
    # Standard stehen — dort gäbe es keinen Gradienten und der Fit bliebe stehen
    p = torch.nn.Parameter(d0 + 1e-3 * span)
    opt = torch.optim.Adam([p], lr=_LR)
    with torch.no_grad():
        start = float(loss_fn(render({n: p[i] for i, n in enumerate(names)})))
    if start < 1e-6:                       # schon im Ziel → nichts tun
        return {n: _DEF[n] for n in names}
    for _ in range(_STEPS):
        opt.zero_grad()
        out = render({n: p[i] for i, n in enumerate(names)})
        loss = loss_fn(out) + _REG * (((p - d0) / span) ** 2).sum()
        loss.backward()
        if not torch.isfinite(p.grad).all():
            break
        opt.step()
        with torch.no_grad():
            p.copy_(torch.max(torch.min(p, hi), lo))
    return {n: round(float(p[i].detach()), 4) for i, n in enumerate(names)}


def _auto_tone(x: torch.Tensor) -> dict:
    names = ["exposure", "contrast", "highlights", "shadows", "whites", "blacks"]
    render = lambda v: grade(x, {**_DEF, **v}).clamp(0.0, 1.0)
    vals = _fit(names, render, lambda o: _band_loss(luma(o)), x.device)
    return {**{n: _DEF[n] for n in SECTIONS["tone"]}, **vals}


def _neutral_mask(x: torch.Tensor) -> torch.Tensor:
    lab = rgb_to_lab(x)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    m = (L > 15) & (L < 95) & (torch.sqrt(a * a + b * b) < 12.0)
    if m.float().mean() < 0.02:            # kaum Unbuntes → Grauwelt über alles Mittlere
        m = (L > 15) & (L < 95)
    return m


def _auto_color(x: torch.Tensor) -> dict:
    m = _neutral_mask(x)
    if not m.any():
        return {n: _DEF[n] for n in SECTIONS["color"]}

    def loss(o):
        lab = rgb_to_lab(o)
        cast = torch.sqrt(lab[..., 1][m].mean() ** 2 + lab[..., 2][m].mean() ** 2 + 1e-9)
        return F.relu(cast - CAST_OK) ** 2 / 100.0

    render = lambda v: grade(x, {**_DEF, **v}).clamp(0.0, 1.0)
    vals = _fit(["temperature", "tint"], render, loss, x.device)
    return {**{n: _DEF[n] for n in SECTIONS["color"]}, **vals}


def _auto_curves(x: torch.Tensor) -> dict:
    """Master-Kurve: jedes Luma-Quantil auf den nächsten Punkt im Band (geschlossene Form)."""
    Y = luma(x).flatten()
    q_in = torch.quantile(Y, torch.tensor(Q, device=x.device)).tolist()
    pts = [[0.0, 0.0]]
    for qi, lo, hi in zip(q_in, BAND_LO, BAND_HI):
        x, t = round(qi, 4), round(min(max(qi, lo), hi), 4)
        # x streng steigend und mit Abstand zu beiden Enden (sonst teilt PCHIP durch null —
        # ein sehr helles Bild rundete sein 98-%-Quantil auf 1,0; Bench 2026-10-13)
        if 0.02 < x < 0.98 and x - pts[-1][0] > 0.02 and pts[-1][1] + 0.005 < t < 0.995:
            pts.append([x, t])
    pts.append([1.0, 1.0])
    if all(abs(px - py) < 0.005 for px, py in pts):
        pts = [[0.0, 0.0], [1.0, 1.0]]
    return {"curves": {"rgb": pts}}


def auto(chain_in: torch.Tensor, section: str) -> dict:
    """AUTO-Reglerwerte für einen Bereich.

    Args:
        chain_in: Bild, wie es beim Bereich in der Kette ankommt (sRGB 0..1, [1,H,W,3]).
        section: Schlüssel aus AUTO_SECTIONS.

    Returns:
        Regler des Bereichs (curves: {"curves": {...}}).

    Raises:
        ValueError: unbekannter oder nicht automatisierbarer Bereich.
    """
    if section not in AUTO_SECTIONS:
        raise ValueError(f"AUTO nicht verfügbar für Bereich {section!r}")
    x = _pixels(chain_in.clamp(0.0, 1.0))
    with torch.enable_grad():
        return {"tone": _auto_tone, "color": _auto_color, "curves": _auto_curves}[section](x)
