"""Contact sheet for the POP preset (prototype of spec 38 formulas, before K15 exists).

Rows: test photos. Columns: original, POP at 0.5 / 1.0 / 1.5, POP 1.0 without clarity.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade.color import lab_to_rgb, rgb_to_lab  # noqa: E402
from refgrade.grade import grade  # noqa: E402

POOL = Path(__file__).resolve().parents[2] / "bench" / "pool"
OUT = Path(__file__).resolve().parents[2] / "reports" / "pop_grid.jpg"
PHOTOS = ["01.jpg", "04.jpg", "07.jpg", "09.jpg"]
TILE = 420
POP = {"contrast": 1.10, "vibrance": 0.30, "highlights": -0.35, "shadows": 0.35,
       "whites": 0.20, "blacks": -0.20, "clarity": 0.35}


def tone_zones(lab, v):
    L = lab[..., 0] / 100.0
    bump = lambda c: torch.exp(-((L - c) / 0.18) ** 2) * 4 * L * (1 - L) / 0.75  # noqa: E731
    Lp = L + 0.25 * (v["shadows"] * bump(0.25) + v["highlights"] * bump(0.75)
                     + v["blacks"] * (1 - L) ** 3 + v["whites"] * L ** 3)
    # monotone: cumulative max on a 1024 lookup
    xs = torch.linspace(0, 1, 1024)
    ys = xs + 0.25 * (v["shadows"] * torch.exp(-((xs - .25) / .18) ** 2) * 4 * xs * (1 - xs) / .75
                      + v["highlights"] * torch.exp(-((xs - .75) / .18) ** 2) * 4 * xs * (1 - xs) / .75
                      + v["blacks"] * (1 - xs) ** 3 + v["whites"] * xs ** 3)
    ys = torch.cummax(ys, 0).values.clamp(0, 1)
    idx = (L.clamp(0, 1) * 1023)
    i0 = idx.floor().long().clamp(0, 1022); f = idx - i0
    Lp = ys[i0] * (1 - f) + ys[i0 + 1] * f
    out = lab.clone(); out[..., 0] = Lp * 100
    return out


def clarity(rgb, amount, full_side):
    lab = rgb_to_lab(rgb)
    L = lab[..., 0:1].permute(0, 3, 1, 2)
    half = 0.008 * full_side
    sigma = 0.3 * half + 0.3
    k = int(6 * np.ceil(sigma) + 1)
    x = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    pad = k // 2
    b = F.conv2d(F.pad(L, (pad, pad, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    b = F.conv2d(F.pad(b, (0, 0, pad, pad), mode="reflect"), g.view(1, 1, k, 1))
    sharp = L * (1 + amount) - amount * b
    m = torch.round(255 * 4 * (L / 100) * (1 - L / 100)).clamp(0, 255) / 255
    out = lab.clone(); out[..., 0:1] = (L + m * (sharp - L)).permute(0, 2, 3, 1)
    return lab_to_rgb(out).clamp(0, 1)


def pop(rgb, s, with_clarity=True, full_side=1000):
    v = {k: (1 + (val - 1) * s if k == "contrast" else val * s) for k, val in POP.items()}
    y = grade(rgb, {"contrast": v["contrast"], "vibrance": v["vibrance"]}).clamp(0, 1)
    y = lab_to_rgb(tone_zones(rgb_to_lab(y), v)).clamp(0, 1)
    if with_clarity and v["clarity"]:
        y = clarity(y, 0.6 * v["clarity"], full_side)
    return y


def tile(t):
    a = (t[0].numpy() * 255).round().astype(np.uint8)
    im = Image.fromarray(a); im.thumbnail((TILE, TILE)); c = Image.new("RGB", (TILE, TILE), (0, 0, 0))
    c.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2)); return c


def main():
    cols = [("ORIGINAL", None), ("POP 0.5", (0.5, True)), ("POP 1.0", (1.0, True)),
            ("POP 1.5", (1.5, True)), ("POP 1.0 no clarity", (1.0, False))]
    head = 34
    sheet = Image.new("RGB", (TILE * len(cols), head + TILE * len(PHOTOS)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for j, (name, _) in enumerate(cols):
        d.text((j * TILE + 10, 6), name, fill=(255, 31, 31) if j else (220, 220, 220), font=font)
    for i, p in enumerate(PHOTOS):
        im = Image.open(POOL / p).convert("RGB"); full = max(im.size)
        im.thumbnail((900, 900)); rgb = torch.from_numpy(np.asarray(im, np.float32) / 255)[None]
        for j, (_, cfg) in enumerate(cols):
            out = rgb if cfg is None else pop(rgb, cfg[0], cfg[1], full_side=max(im.size))
            sheet.paste(tile(out), (j * TILE, head + i * TILE))
    sheet.save(OUT, quality=92); print(OUT)


if __name__ == "__main__":
    main()
