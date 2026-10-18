"""AUTO against real expert retouches (MIT-Adobe FiveK subset, research licence, local only).

Per pair: the "before" image is graded by
  none        — untouched
  auto_<sec>  — AUTO of one section, applied alone (as in the additive mix with the rest at default)
  auto_all    — all AUTO sections in the additive mix (what pressing every "A" gives)
  ref_*       — established reference methods (grey-world von Kries WB, luminance auto-levels)
  oracle_<sec>— that section's sliders fitted to the expert result (upper bound of the sliders)
and compared with the expert "after": dE76 mean, and split into dL (tone) and dab (colour).

Writes reports/auto_fivek.md and reports/auto_fivek.jsonl.
Usage: python tools/auto_bench_fivek.py [--side 512] [--no-oracle] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade.auto import AUTO_SECTIONS, auto  # noqa: E402  (neue Signatur auto(x, section), 2026-10-13)
from refgrade.color import luma, rgb_to_lab, srgb_to_linear, linear_to_srgb  # noqa: E402
from refgrade.curves import apply as curves_apply  # noqa: E402
from refgrade.grade import grade  # noqa: E402
from refgrade.rgb import rgb_grade  # noqa: E402
from refgrade.sections import SECTIONS, section_values  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "research" / "pairs_fivek"
OUT_MD = ROOT / "reports" / "auto_fivek.md"
OUT_JL = ROOT / "reports" / "auto_fivek.jsonl"
DEFAULTS = {s.name: s.default for s in SLIDERS}


def load(p: Path, side: int) -> torch.Tensor:
    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side))
    return torch.from_numpy(np.asarray(im, np.float32) / 255)[None]


def de(p: torch.Tensor, q: torch.Tensor) -> dict:
    lp, lq = rgb_to_lab(p.clamp(0, 1)), rgb_to_lab(q)
    d = lp - lq
    return {"de": d.norm(dim=-1).mean().item(),
            "dL": d[..., 0].abs().mean().item(),
            "dab": d[..., 1:].norm(dim=-1).mean().item()}


def section_fn(sec: str, vals: dict):
    """sRGB → sRGB for one section, built the same way session.build_lut does."""
    if sec in ("tone", "color", "wheels", "hsl"):
        v = section_values(vals, sec)
        return lambda x: grade(x, v)
    if sec == "rgb":
        v = section_values(vals, "rgb")
        kw = {f"{'master' if c == 'm' else c}_{k}": v[f"rgb_{c}_{k}"]
              for c in ("m", "r", "g", "b") for k in ("gamma", "pedestal", "gain")}
        kw.update({f"use_{z}": v[f"rgb_use_{z}"] for z in ("shadows", "midtones", "highlights")})
        return lambda x: rgb_grade(x, **kw)
    if sec == "curves":
        c = vals.get("curves")
        return (lambda x: curves_apply(x, c)) if c else (lambda x: x)
    raise KeyError(sec)


def additive(x: torch.Tensor, fns: list) -> torch.Tensor:
    total = torch.zeros_like(x)
    for f in fns:
        total = total + (f(x) - x)
    return (x + total).clamp(0, 1)


def ref_grayworld(x: torch.Tensor) -> torch.Tensor:
    """Grey-world von Kries in linear light (the K23 bench winner)."""
    lin = srgb_to_linear(x)
    m = lin.reshape(-1, 3).mean(0)
    gain = m.mean() / m.clamp_min(1e-6)
    return linear_to_srgb((lin * gain).clamp(0, 1))


def ref_autolevels(x: torch.Tensor) -> torch.Tensor:
    """Luminance 0.5/99.5 % stretch, same gain on all channels (hue kept)."""
    Y = luma(x).flatten()
    idx = torch.arange(0, Y.numel(), max(1, Y.numel() // 20000))
    lo, hi = Y[idx].quantile(0.005).item(), Y[idx].quantile(0.995).item()
    return ((x - lo) / max(hi - lo, 1e-3)).clamp(0, 1)


def oracle(x: torch.Tensor, y: torch.Tensor, sec: str) -> torch.Tensor:
    from refgrade.fit_sliders import fit_sliders
    names = [n for n in SECTIONS[sec] if n not in ("clarity", "texture", "dehaze",
                                                   "wheel_gain", "wheel_falloff",
                                                   "wheel_intensity")]
    vals, _ = fit_sliders(x, y, names=names)
    return section_fn(sec, {**DEFAULTS, **vals})(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pairs = sorted(PAIRS.glob("*_before.png"))
    if a.limit:
        pairs = pairs[:a.limit]
    rows = []
    t0 = time.time()
    for b in pairs:
        x = load(b, a.side)
        y = load(Path(str(b).replace("_before", "_after")), a.side)
        if x.shape != y.shape:
            continue
        r = {"pair": b.name[:3], "none": de(x, y)}
        fns, vals_all = [], {}
        for sec in AUTO_SECTIONS:
            out = auto(x, sec)
            vals_all[sec] = out
            f = section_fn(sec, {**DEFAULTS, **out})
            fns.append(f)
            r[f"auto_{sec}"] = de(f(x).clamp(0, 1), y)
        r["auto_all"] = de(additive(x, fns), y)
        r["ref_grayworld"] = de(ref_grayworld(x), y)
        r["ref_autolevels"] = de(ref_autolevels(x), y)
        r["ref_levels+gw"] = de(ref_autolevels(ref_grayworld(x)), y)
        if not a.no_oracle:
            for sec in ("tone", "color", "wheels"):
                r[f"oracle_{sec}"] = de(oracle(x, y, sec), y)
        r["auto_values"] = {k: {n: (round(v, 3) if isinstance(v, float) else v)
                                for n, v in d.items()} for k, d in vals_all.items()}
        rows.append(r)
        print(f"{r['pair']}: none {r['none']['de']:.2f}  auto_all {r['auto_all']['de']:.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)

    keys = [k for k in rows[0] if k not in ("pair", "auto_values")]
    with open(OUT_JL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = [f"# AUTO vs FiveK expert retouches\n",
             f"{len(rows)} pairs, longest side {a.side}. Lower = closer to the expert. "
             f"'worse' = pairs where the config is further from the expert than untouched.\n",
             "| config | dE76 median | dE76 mean | dL median | dab median | worse than none |",
             "|---|---|---|---|---|---|"]
    for k in keys:
        dv = np.array([r[k]["de"] for r in rows])
        dl = np.array([r[k]["dL"] for r in rows])
        da = np.array([r[k]["dab"] for r in rows])
        worse = sum(r[k]["de"] > r["none"]["de"] + 0.1 for r in rows)
        lines.append(f"| {k} | {np.median(dv):.2f} | {dv.mean():.2f} | {np.median(dl):.2f} | "
                     f"{np.median(da):.2f} | {worse}/{len(rows)} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
