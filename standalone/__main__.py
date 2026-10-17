"""Standalone CLI: grade an image by reference without Krita.

Usage:
    python -m standalone input.png ref1.png [ref2.png ...] [options]

Examples:
    python -m standalone photo.jpg sunset.jpg --method reinhard -o graded.jpg
    python -m standalone photo.jpg sunset.jpg --cube sunset.cube
    python -m standalone photo.jpg sunset.jpg --method zonal --no-fit -o out.png
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# core/ auf den Pfad legen (refgrade-Package)
_CORE = Path(__file__).resolve().parent.parent / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import numpy as np
import torch

from .image_bridge import read_image, write_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="standalone",
        description="Grade an image by reference (no Krita needed).",
    )
    parser.add_argument("input", help="Source image file (PNG, JPG, TIFF, …)")
    parser.add_argument("refs", nargs="+", help="Reference image file(s)")
    parser.add_argument("--method", choices=("reinhard", "mkl", "zonal", "pair"),
                        default="reinhard", help="Grading method (default: reinhard)")
    parser.add_argument("--no-fit", action="store_true",
                        help="Skip auto-fit; use default slider values")
    parser.add_argument("-o", "--output", help="Output image file (default: <input>_rg.<ext>)")
    parser.add_argument("--cube", metavar="PATH", help="Also export a .cube LUT to this path")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        help="Force device (default: auto)")
    parser.add_argument("--max-side", type=int, default=0,
                        help="Scale source down so longest side ≤ N pixels (0 = full res)")
    args = parser.parse_args(argv)

    # Resolve output path
    src_path = Path(args.input)
    if not src_path.is_file():
        print(f"error: input not found: {src_path}", file=sys.stderr)
        return 1
    for r in args.refs:
        if not Path(r).is_file():
            print(f"error: reference not found: {r}", file=sys.stderr)
            return 1

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = src_path.suffix or ".png"
        out_path = src_path.with_name(f"{src_path.stem}_rg{suffix}")

    # Device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("error: --device cuda requested but CUDA is not available", file=sys.stderr)
            return 1
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"device: {device}")

    # Read source
    with tempfile.TemporaryDirectory(prefix="refgrade_cli_") as tmp:
        tmp_path = Path(tmp)
        source = read_image(str(src_path), tmp_path)
        print(f"source: {source['width']}x{source['height']} {source['depth']}")

        # Refs: pass actual image file paths (session loads them via PIL)
        ref_paths = [str(Path(r).resolve()) for r in args.refs]
        for i, r in enumerate(args.refs):
            print(f"ref {i + 1}: {Path(r).name}")

        # Open session
        from refgrade import session as sess

        print("opening session…")
        s = sess.open_session(
            source["raw"], source["width"], source["height"], source["depth"],
            device=device,
        )

        # Fit or defaults
        if args.no_fit:
            from refgrade.sliders import SLIDERS
            values = {sl.name: sl.default for sl in SLIDERS}
            print("sliders: defaults (--no-fit)")
        else:
            print(f"fitting ({args.method})…")
            values = sess.fit(s, ref_paths, args.method)
            fitted = {k: v for k, v in values.items() if v != 0.0}
            print(f"sliders: {len(fitted)} non-zero")

        # Apply
        print("applying…")
        result = sess.apply_full(s, ref_paths, args.method, values)

        # Scale down if requested
        if args.max_side > 0:
            import torch.nn.functional as F
            h, w = result.shape[1], result.shape[2]
            scale = args.max_side / max(h, w)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                result = F.interpolate(
                    result.permute(0, 3, 1, 2), size=(new_h, new_w),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
                print(f"scaled to {new_w}x{new_h}")

        # Save image: tensor → raw bytes → image file
        from refgrade import rawio

        h, w = result.shape[1], result.shape[2]
        alpha = np.full((h, w), 255 if source["depth"] == "U8" else 65535,
                        dtype=np.uint8 if source["depth"] == "U8" else np.uint16)
        raw_out = tmp_path / "result.raw"
        rawio.write_raw(str(raw_out), result, alpha, source["depth"])
        write_image(source, raw_out, out_path)
        print(f"saved: {out_path}")

        # Export .cube
        if args.cube:
            cube_path = Path(args.cube)
            lut = sess.build_lut(s, ref_paths, args.method, values)
            sess.lut_write_cube(lut, str(cube_path))
            print(f"cube:  {cube_path}")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
