"""End-to-end check against the REAL worker (no Krita, no window): load a pool image in the
raw format Krita delivers, then compute previews with each section and check the effect
arrives in the pixels. Prints one line per check; exit code 1 if any fails.

    python tools/e2e_check.py        (from core\\)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules["ref_grade"] = pkg
from ref_grade import config, worker_client, worker_process  # noqa: E402

PORT = 47199
POOL = ROOT / "bench" / "pool"
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")


def preview(session: str, mix: dict, values: dict | None = None) -> tuple[np.ndarray, dict]:
    r = worker_client.call(PORT, "/preview", {
        "session": session, "references": [str(POOL / "01.jpg")], "method": "reinhard",
        "values": values or {}, "max_side": 256, "mix": mix})
    return np.asarray(Image.open(r["image"]).convert("RGB"), dtype=np.float32) / 255, r


worker_process.ensure(PORT, config.python_exe(), config.CORE, timeout=60)
try:
    rgb = np.asarray(Image.open(POOL / "07.jpg").convert("RGB"))
    h, w = rgb.shape[:2]
    raw = config.temp_dir() / "e2e_source.raw"
    raw.write_bytes(np.concatenate([rgb[..., ::-1], np.full((h, w, 1), 255, np.uint8)], 2).tobytes())
    sess = worker_client.call(PORT, "/load", {"raw": str(raw), "width": w, "height": h,
                                              "depth": "U8"})["session"]

    off = {"weights": {}, "master": 1.0}
    base, r0 = preview(sess, off)
    # identity reference: master 0 (byte-identical to the source per test_mix/apply_full);
    # a PIL resize is no reference — it resamples differently (~0.03 mean difference)
    ident, _ = preview(sess, {"weights": {"tone": 1.0}, "master": 0.0}, {"exposure": 1.0})
    check("nothing on = identity", float(np.abs(base - ident).max()) < 1 / 255,
          f"max diff {np.abs(base - ident).max():.4f}")
    check("vectorscope in /preview", len(r0.get("vectorscope", [])) == 96,
          f"rows={len(r0.get('vectorscope', []))}")

    img, _ = preview(sess, {"weights": {"reference": 1.0}, "master": 1.0})
    check("reference changes the image", float(np.abs(img - base).mean()) > 0.005,
          f"mean diff {np.abs(img - base).mean():.4f}")

    curves = {"rgb": [[0, 0], [0.5, 0.25], [1, 1]]}
    img, _ = preview(sess, {"weights": {"curves": 1.0}, "master": 1.0, "curves": curves})
    check("RGB curve darkens midtones", float(img.mean()) < float(base.mean()) - 0.03,
          f"mean {base.mean():.3f} -> {img.mean():.3f}")

    hue = {"hue_sat": [[0, 0.0], [0.5, 0.0], [1, 0.0]]}
    img, _ = preview(sess, {"weights": {"curves": 1.0}, "master": 1.0, "curves": hue})
    sat = lambda a: float((a.max(2) - a.min(2)).mean())  # noqa: E731
    check("HvS = 0 desaturates", sat(img) < 0.5 * sat(base), f"sat {sat(base):.3f} -> {sat(img):.3f}")

    lut = sorted((ROOT / "luts").glob("*.cube"))[0]
    img, _ = preview(sess, {"weights": {"lut": 1.0}, "master": 1.0, "lut": str(lut)})
    check("LUT changes the image", float(np.abs(img - base).mean()) > 0.005, lut.name)

    img, _ = preview(sess, {"weights": {"tone": 1.0}, "master": 1.0}, {"exposure": 1.0})
    check("tone exposure +1 brightens", float(img.mean()) > float(base.mean()) + 0.05,
          f"mean {base.mean():.3f} -> {img.mean():.3f}")
finally:
    worker_process.stop(PORT)
print("FAILURES:", failures)
sys.exit(1 if failures else 0)
