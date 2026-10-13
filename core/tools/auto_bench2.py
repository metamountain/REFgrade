"""AUTO-Neuentwurf (refgrade.auto2) gegen FiveK-Expertenbilder, Kette wie im Plugin.

Je Paar: "before" → AUTO der Bereiche in Kettenreihenfolge, jede AUTO auf dem Bild, wie es bei ihr
ankommt (= Drücken aller A-Knöpfe von oben nach unten). Getrennt nach Tuning-Hälfte (gerade
Paare, daraus stammen die Bänder) und Prüf-Hälfte (ungerade). Dazu "harm": dieselbe Kette auf das
Expertenbild selbst — ein schon gutes Bild soll sich kaum ändern (dE76 zum Expertenbild).

Schreibt reports/auto2_fivek.md (+ .jsonl).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade import auto as auto2  # noqa: E402
from tools.auto_bench_fivek import DEFAULTS, PAIRS, de, load, section_fn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "reports" / "auto2_fivek.md"
CHAIN = ("color", "tone", "curves")


def press(x: torch.Tensor, secs) -> tuple[torch.Tensor, dict]:
    y, got = x, {}
    for sec in secs:
        v = auto2.auto(y, sec)
        got[sec] = v
        y = section_fn(sec, {**DEFAULTS, **v})(y).clamp(0, 1)
    return y, got


def main() -> None:
    side = 512
    rows = []
    t0 = time.time()
    for i, b in enumerate(sorted(PAIRS.glob("*_before.png"))):
        x = load(b, side)
        y = load(Path(str(b).replace("_before", "_after")), side)
        if x.shape != y.shape:
            continue
        r = {"pair": b.name[:3], "half": "tune" if i % 2 == 0 else "test", "none": de(x, y)}
        for sec in CHAIN:
            r[f"only_{sec}"] = de(press(x, [sec])[0], y)
        out, vals = press(x, CHAIN)
        r["chain_all"] = de(out, y)
        r["chain_tone_color"] = de(press(x, ["color", "tone"])[0], y)
        r["harm"] = de(press(y, CHAIN)[0], y)
        r["values"] = {k: {n: v for n, v in d.items() if v != DEFAULTS.get(n)} if k != "curves" else d
                       for k, d in vals.items()}
        rows.append(r)
        print(f"{r['pair']} {r['half']}: none {r['none']['de']:.2f} chain {r['chain_all']['de']:.2f} "
              f"harm {r['harm']['de']:.2f} ({time.time() - t0:.0f} s)", flush=True)

    with open(OUT_MD.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    keys = [k for k in rows[0] if k not in ("pair", "half", "values", "harm")]
    lines = ["# AUTO-Neuentwurf (auto2) vs FiveK-Experten\n",
             f"{len(rows)} Paare, Seite {side}. dE76 zum Expertenbild (kleiner = näher). "
             "'schlechter' = weiter weg als unbearbeitet.\n"]
    for half in ("tune", "test"):
        rs = [r for r in rows if r["half"] == half]
        lines += [f"\n## {half} ({len(rs)} Paare)\n",
                  "| Konfiguration | dE76 Median | dE76 Mittel | dL Median | dab Median | schlechter |",
                  "|---|---|---|---|---|---|"]
        for k in keys:
            dv = np.array([r[k]["de"] for r in rs])
            dl = np.array([r[k]["dL"] for r in rs])
            da = np.array([r[k]["dab"] for r in rs])
            worse = sum(r[k]["de"] > r["none"]["de"] + 0.1 for r in rs)
            lines.append(f"| {k} | {np.median(dv):.2f} | {dv.mean():.2f} | {np.median(dl):.2f} | "
                         f"{np.median(da):.2f} | {worse}/{len(rs)} |")
    h = np.array([r["harm"]["de"] for r in rows])
    lines += ["\n## Do-no-harm (Kette auf das Expertenbild selbst)\n",
              f"dE76 zum unveränderten Expertenbild: Median {np.median(h):.2f}, p90 "
              f"{np.percentile(h, 90):.2f}, max {h.max():.2f}"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
