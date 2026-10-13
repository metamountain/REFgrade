"""Neon-/Plasma-Parade: Varianten zum Aussuchen rendern (Opus, 2026-10-11).

Nutzer: „Glow wie DaVinci — in verdichteten Bereichen heller leuchten, plasmaartig, Neon.
Render mir Testbilder in Intensitäten.“

python tools/parade_neon_preview.py  →  reports/shots/neon/*.png + Übersicht neon_sheet.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

CORE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CORE))
from refgrade.scope import rgb_parade  # noqa: E402

OUT = CORE.parent / "reports" / "shots" / "neon"
CH_COLOR = {"r": (1.0, 0.16, 0.10), "g": (0.15, 1.0, 0.30), "b": (0.20, 0.38, 1.0)}


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).to(x)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def neon(counts: torch.Tensor, color: tuple, glow: float, hot: float, size: tuple[int, int]) -> torch.Tensor:
    """counts [levels, cols] (Anteile, Zeile 0 = Wert 0) → RGB [H, W, 3] 0..1.

    Nutzer: „Es sollen keine Infos zermatscht werden.“ Darum:
    - Kern scharf: Dichte (rel. 99,5-%-Perzentil, Wurzel) ohne Weichzeichnung, Helligkeit steigt
      mit der Dichte — die Struktur bleibt vollständig lesbar.
    - Halo nur daneben: Bloom aus den dichten Bereichen wirkt mit (1 − Kern), überdeckt also nie
      Daten, sondern füllt nur die dunkle Umgebung (Neon-Schein).
    - Weiß nur an den Spitzen: Überschuss über ~1,3× Perzentil, gedeckelt auf `hot`.
    """
    c0 = counts.float()
    ref = torch.quantile(c0[c0 > 0], 0.995) if (c0 > 0).any() else torch.tensor(1.0)
    v = (c0 / ref.clamp_min(1e-12)).clamp(0, 4) ** 0.5
    v = v.flip(0)[None, None]
    v = F.interpolate(v, size=size, mode="bilinear", align_corners=False).clamp_min(0)
    s = size[1] / 256.0
    core = 1.0 - torch.exp(-1.8 * v ** 1.3)                      # scharf, dichter = heller
    d = v * v
    bloom = 0.5 * _blur(d, 2.5 * s) + 0.3 * _blur(d, 7 * s) + 0.2 * _blur(d, 18 * s)
    halo = (1.0 - torch.exp(-glow * bloom)) * (1.0 - core)       # nur wo der Kern dunkel ist
    c = torch.tensor(color).view(1, 3, 1, 1)
    peak = torch.relu(v - 1.15).clamp(max=1.0)
    white = hot * (1.0 - torch.exp(-3.0 * peak))                 # nur die dichtesten Spitzen
    rgb = (core + 0.75 * halo) * c + white * (1.0 - c)
    return rgb[0].permute(1, 2, 0).clamp(0, 1)


def panel(img: torch.Tensor, glow: float, hot: float, w: int = 800, h: int = 300) -> Image.Image:
    par = rgb_parade(img)
    gap = max(1, int(w * 0.02))
    pw = (w - 2 * gap) // 3
    out = np.zeros((h, w, 3), np.float32)
    for i, ch in enumerate(("r", "g", "b")):
        rgb = neon(par[ch].float().cpu(), CH_COLOR[ch], glow, hot, (h, pw)).numpy()
        x0 = i * (pw + gap)
        out[:, x0:x0 + pw] = rgb
    im = Image.fromarray((out * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(im)
    for frac in (0.25, 0.5, 0.75):                                # dezente Skala
        y = int(h * (1 - frac))
        for i in range(3):
            x0 = i * (pw + gap)
            d.line([(x0, y), (x0 + pw, y)], fill=(38, 38, 38))
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = CORE.parent / "bench" / "pool"
    srcs = [pool / "04.jpg", pool / "06.jpg"]
    variants = [("1_halo_leicht", 1.0, 0.25), ("2_halo_mittel", 2.0, 0.35),
                ("3_neon", 3.5, 0.45), ("4_neon_stark", 6.0, 0.55)]
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (800 * len(srcs) + 20 * (len(srcs) - 1), (300 + 34) * len(variants)), (20, 20, 20))
    for j, src in enumerate(srcs):
        a = np.asarray(Image.open(src).convert("RGB").resize((768, 512)), np.float32) / 255
        img = torch.from_numpy(a)[None]
        for i, (name, glow, hot) in enumerate(variants):
            im = panel(img, glow, hot)
            im.save(OUT / f"{name}_{src.stem}.png")
            y = i * 334
            sheet.paste(im, (j * 820, y + 34))
            ImageDraw.Draw(sheet).text((j * 820 + 6, y + 6), f"{name}  (glow {glow}, hot {hot})  Bild {src.stem}",
                                       fill=(230, 230, 230), font=font)
    sheet.save(OUT / "neon_sheet.png")
    # Zoom: oberes Drittel des R-Panels von Bild 04, 3×, je Variante nebeneinander
    a = np.asarray(Image.open(srcs[0]).convert("RGB").resize((768, 512)), np.float32) / 255
    img = torch.from_numpy(a)[None]
    crops = [panel(img, g, h).crop((0, 0, 256, 110)).resize((512, 220), Image.NEAREST) for _, g, h in variants]
    z = Image.new("RGB", (512 * len(crops) + 10 * (len(crops) - 1), 250), (20, 20, 20))
    for i, (cimg, (name, _, _)) in enumerate(zip(crops, variants)):
        z.paste(cimg, (i * 522, 30))
        ImageDraw.Draw(z).text((i * 522 + 4, 4), name + " (Zoom 2x)", fill=(230, 230, 230), font=font)
    z.save(OUT / "neon_zoom.png")
    print(OUT / "neon_sheet.png")


if __name__ == "__main__":
    main()
