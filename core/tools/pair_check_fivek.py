"""PAIR on real expert retouches (MIT-Adobe FiveK subset, research licence, local only).

Per pair: dE76 without LUT, with PAIR fitted on the whole pair, and held out
(fit on the left half, measured on the right half). Writes reports/pair_fivek.md.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade.color import rgb_to_lab  # noqa: E402
from refgrade.lut import apply as lut_apply  # noqa: E402
from refgrade.pairlut import as_lut, fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "research" / "pairs_fivek"
OUT = ROOT / "reports" / "pair_fivek.md"
SIDE = 768


def load(p):
    im = Image.open(p).convert("RGB"); im.thumbnail((SIDE, SIDE))
    return torch.from_numpy(np.asarray(im, np.float32) / 255)[None]


def de(p, q):
    d = (rgb_to_lab(p) - rgb_to_lab(q)).norm(dim=-1).flatten()
    return d.mean().item(), d.quantile(0.95).item()


def main():
    rows = []
    for b in sorted(PAIRS.glob("*_before.png")):
        x, y = load(b), load(str(b).replace("_before", "_after"))
        lut = as_lut(fit(x, y))
        base, full = de(x, y), de(lut_apply(x, lut).clamp(0, 1), y)
        w = x.shape[2] // 2
        hl = as_lut(fit(x[:, :, :w].contiguous(), y[:, :, :w].contiguous()))
        hold = de(lut_apply(x[:, :, w:].contiguous(), hl).clamp(0, 1), y[:, :, w:])
        rows.append((b.name[:3], base, full, hold))
        print(b.name[:3], "base %.1f  full %.2f  hold %.2f" % (base[0], full[0], hold[0]), flush=True)
    a = np.array([[r[1][0], r[2][0], r[3][0], r[2][1], r[3][1]] for r in rows])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# PAIR on FiveK expert retouches (local, research licence)\n\n")
        f.write(f"{len(rows)} pairs, longest side {SIDE}. dE76 mean (p95).\n\n")
        f.write("| | no LUT | PAIR full | PAIR held out |\n|---|---|---|---|\n")
        f.write(f"| median | {np.median(a[:,0]):.2f} | {np.median(a[:,1]):.2f} ({np.median(a[:,3]):.1f}) | {np.median(a[:,2]):.2f} ({np.median(a[:,4]):.1f}) |\n")
        f.write(f"| worst | {a[:,0].max():.2f} | {a[:,1].max():.2f} | {a[:,2].max():.2f} |\n\n")
        f.write("| pair | no LUT | full | held out |\n|---|---|---|---|\n")
        for n, base, full, hold in rows:
            f.write(f"| {n} | {base[0]:.2f} | {full[0]:.2f} ({full[1]:.1f}) | {hold[0]:.2f} ({hold[1]:.1f}) |\n")
    print(OUT)


if __name__ == "__main__":
    main()
