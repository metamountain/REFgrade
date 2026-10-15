"""Direkt-Test Color Replace auf roeper.jpg: blauer Himmel → flashy rot.

Kein Worker, kein Krita — reiner Core-Pfad (refgrade.color_replace.apply).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from refgrade.color_replace import apply  # noqa: E402


def to_tensor(pil: Image.Image) -> torch.Tensor:
    a = np.array(pil.convert("RGB")).astype(np.float32) / 255.0  # [H,W,3]
    return torch.from_numpy(a).unsqueeze(0)  # [1,H,W,3]


def to_pil(t: torch.Tensor) -> Image.Image:
    a = t[0].clamp(0.0, 1.0).numpy()  # [H,W,3]
    return Image.fromarray((a * 255.0).round().astype(np.uint8))


def sky_stats(pil: Image.Image, label: str) -> None:
    a = np.array(pil.convert("RGB")).reshape(-1, 3).astype(np.float32) / 255.0
    # Himmelsregion: obere linke Ecke (sicher Himmel, kein Scheunendach)
    h, w = pil.size[1], pil.size[0]
    sky = np.array(pil.convert("RGB"))[int(h * 0.05):int(h * 0.30), int(w * 0.05):int(w * 0.55)]
    sky = sky.reshape(-1, 3).astype(np.float32) / 255.0
    r, g, b = sky.mean(0)
    mx, mn = sky.max(1), sky.min(1)
    delta = mx - mn
    v = mx
    s = (delta / np.clip(mx, 1e-6, None)).mean()
    # hue mean
    import colorsys
    hh, ss, vv = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    print(f"  [{label}] sky mean RGB=({r:.3f},{g:.3f},{b:.3f})  S={ss:.3f} V={vv:.3f} hue={hh*360:.0f}deg")
    # "red enough"? R deutlich > B und R > G
    red = r > 0.5 and r > b + 0.25 and r > g + 0.15
    print(f"          -> FLASHY RED? {red}  (R={r:.3f} G={g:.3f} B={b:.3f})")


def main() -> None:
    src = Path(__file__).parent.parent / "roeper.jpg"
    out = Path(__file__).parent.parent / "roeper_red_sky.png"

    pil = Image.open(src).convert("RGB")
    print(f"Input:  {src}  {pil.size}")
    print("Sky BEFORE:")
    sky_stats(pil, "before")

    x = to_tensor(pil)
    vals = {f"cr_{i}_{k}": 0.0 for i in range(8)
            for k in ("src_hue", "src_sat", "tgt_hue", "tgt_sat", "intensity", "range")}
    # Blauer Himmel (hue ~212) -> flashy Rot (hue 0, volle Sättigung)
    vals["cr_0_src_hue"] = 212.0
    vals["cr_0_src_sat"] = 1.0
    vals["cr_0_tgt_hue"] = 0.0
    vals["cr_0_tgt_sat"] = 1.0
    vals["cr_0_intensity"] = 1.0
    vals["cr_0_range"] = 90.0

    y = apply(x, vals)
    out_pil = to_pil(y)
    out_pil.save(out)
    print(f"\nOutput: {out}")
    print("Sky AFTER:")
    sky_stats(out_pil, "after")


if __name__ == "__main__":
    main()
