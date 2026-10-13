"""Literatur-AUTO (refgrade.auto_lit) im selben Rahmen wie auto_bench2: FiveK, Kette, tune/test, harm.

Varianten: Shades-of-Gray p=6 bzw. Grauwelt p=1 für COLOR, Reinhard-Schlüssel 0,18 für TONE,
CLHE (Steigung ≤ 2) für CURVES — einzeln und als Kette. Schreibt reports/auto_lit_fivek.md (+.jsonl).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade import auto_lit  # noqa: E402
from tools.auto_bench_fivek import DEFAULTS, PAIRS, de, load, section_fn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "auto_lit_fivek.md"


def press(x: torch.Tensor, secs, p: float) -> torch.Tensor:
    y = x
    for sec in secs:
        v = auto_lit.auto(y, sec, sog_p=p)
        y = section_fn(sec, {**DEFAULTS, **v})(y).clamp(0, 1)
    return y


CONFIGS = {
    "only_color_sog6": (["color"], 6.0), "only_color_gw": (["color"], 1.0),
    "only_tone_reinhard": (["tone"], 6.0), "only_curves_clhe": (["curves"], 6.0),
    "chain_sog6": (["color", "tone", "curves"], 6.0), "chain_gw": (["color", "tone", "curves"], 1.0),
    "chain_sog6_no_curves": (["color", "tone"], 6.0),
}


def main() -> None:
    rows = []
    for i, b in enumerate(sorted(PAIRS.glob("*_before.png"))):
        x = load(b, 512)
        y = load(Path(str(b).replace("_before", "_after")), 512)
        if x.shape != y.shape:
            continue
        r = {"pair": b.name[:3], "half": "tune" if i % 2 == 0 else "test", "none": de(x, y)}
        for k, (secs, p) in CONFIGS.items():
            r[k] = de(press(x, secs, p), y)
            r["harm_" + k] = de(press(y, secs, p), y)
        rows.append(r)
        print(r["pair"], {k: round(r[k]["de"], 2) for k in ("none", "chain_sog6", "chain_gw")}, flush=True)
    with open(OUT.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = ["# Literatur-AUTO vs FiveK-Experten\n", f"{len(rows)} Paare, Seite 512, dE76.\n"]
    for half in ("tune", "test"):
        rs = [r for r in rows if r["half"] == half]
        lines += [f"\n## {half} ({len(rs)})\n", "| Konfiguration | dE Median | dL Median | dab Median | "
                  "schlechter | harm Median | harm p90 | harm max |", "|---|---|---|---|---|---|---|---|"]
        for k in ["none"] + list(CONFIGS):
            dv = np.array([r[k]["de"] for r in rs])
            worse = sum(r[k]["de"] > r["none"]["de"] + 0.1 for r in rs)
            h = np.array([r["harm_" + k]["de"] for r in rs]) if k != "none" else np.zeros(1)
            lines.append(f"| {k} | {np.median(dv):.2f} | {np.median([r[k]['dL'] for r in rs]):.2f} | "
                         f"{np.median([r[k]['dab'] for r in rs]):.2f} | {worse}/{len(rs)} | "
                         f"{np.median(h):.2f} | {np.percentile(h, 90):.2f} | {h.max():.2f} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
