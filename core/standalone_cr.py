"""Standalone RefGrade CR-Test: Bild laden, CR anwenden, speichern.

Kein Krita, kein MCP — reiner Worker-Pipeline-Test.
"""
import sys
import json
import base64
import tempfile
from pathlib import Path
from urllib.request import urlopen, Request
import numpy as np
from PIL import Image

WORKER = "http://127.0.0.1:47123"


def call(path: str, payload: dict | None = None) -> dict:
    url = f"{WORKER}{path}"
    if payload is None:
        req = Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_image(path: str) -> dict:
    """Bild laden → Worker /load (BGRA, depth='U8')."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    a = np.array(img)  # [H, W, 3] RGB
    # Worker erwartet BGRA (4 Kanäle)
    bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)])

    tmp = Path(tempfile.gettempdir()) / "refgrade_standalone"
    tmp.mkdir(exist_ok=True)
    raw_path = tmp / "source.raw"
    bgra.tofile(raw_path)

    source = {
        "raw": str(raw_path),
        "width": w,
        "height": h,
        "depth": "U8",
    }
    resp = call("/load", source)
    print(f"  /load OK: session={resp.get('session', '?')}")
    return resp


def apply_cr(session: str, src_path: str, src_hue=215.0, src_sat=0.5, tgt_hue=0.0,
             tgt_sat=0.7, intensity=1.0, range_deg=35.0, index=0) -> Image.Image:
    """CR anwenden → /preview → PIL Image."""
    values = {f"cr_{i}_{k}": 0.0 for i in range(8)
              for k in ("src_hue", "src_sat", "tgt_hue", "tgt_sat", "intensity", "range")}
    values[f"cr_{index}_src_hue"] = src_hue
    values[f"cr_{index}_src_sat"] = src_sat
    values[f"cr_{index}_tgt_hue"] = tgt_hue
    values[f"cr_{index}_tgt_sat"] = tgt_sat
    values[f"cr_{index}_intensity"] = intensity
    values[f"cr_{index}_range"] = range_deg

    mix = {"weights": {k: 1.0 for k in ("tone", "color", "wheels", "rgb", "hsl",
                                         "color_replace", "curves", "lut", "reference")},
           "master": 1.0, "solo": None}
    resp = call("/preview", {
        "session": session, "references": [src_path], "method": "reinhard",
        "values": values, "max_side": 1600, "mix": mix, "stage": "full",
    })
    img_path = resp["image"]
    return Image.open(img_path).convert("RGB")


def save_result(img: Image.Image, out_path: str, orig_path: str | None = None) -> None:
    """Bild speichern + optional Vergleich."""
    img.save(out_path, quality=95)
    print(f"  Gespeichert: {out_path}")

    if orig_path:
        before = Image.open(orig_path).convert("RGB").resize(img.size)
        w, h = img.size
        canvas = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
        canvas.paste(before, (0, 0))
        canvas.paste(img, (w + 10, 0))
        cmp_path = str(Path(out_path).with_name(Path(out_path).stem + "_vergleich" + Path(out_path).suffix))
        canvas.save(cmp_path)
        print(f"  Vergleich: {cmp_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python standalone_cr.py <input.jpg> [output.jpg] [--src-hue 215 --tgt-hue 0 ...]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else in_path.replace(".jpg", "_cr.jpg")

    # Parse CR args
    args = sys.argv[2:]
    cr = {"src_hue": 215.0, "src_sat": 0.5, "tgt_hue": 0.0, "tgt_sat": 0.7,
          "intensity": 1.0, "range_deg": 35.0}
    i = 0
    while i < len(args):
        if args[i] == "--src-hue": cr["src_hue"] = float(args[i+1]); i += 2
        elif args[i] == "--src-sat": cr["src_sat"] = float(args[i+1]); i += 2
        elif args[i] == "--tgt-hue": cr["tgt_hue"] = float(args[i+1]); i += 2
        elif args[i] == "--tgt-sat": cr["tgt_sat"] = float(args[i+1]); i += 2
        elif args[i] == "--intensity": cr["intensity"] = float(args[i+1]); i += 2
        elif args[i] == "--range": cr["range_deg"] = float(args[i+1]); i += 2
        else: i += 1

    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"CR: src=({cr['src_hue']}°, {cr['src_sat']}) → tgt=({cr['tgt_hue']}°, {cr['tgt_sat']}), "
          f"int={cr['intensity']}, range={cr['range_deg']}°")

    print("\n1. Worker /load ...")
    load_resp = load_image(in_path)
    session = load_resp["session"]

    print("2. Worker /preview (CR) ...")
    result = apply_cr(session, in_path, **cr)
    print(f"   Ergebnis: {result.size[0]}x{result.size[1]}")

    print("3. Speichern ...")
    save_result(result, out_path, in_path)
    print("\nFertig.")


if __name__ == "__main__":
    main()
