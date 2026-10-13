r"""Bench: Renderzeiten für preview/apply/cube messen (Paket K27e).

Aufruf (aus core\):
    python tools/bench_render.py          # 0,5/2/8/24 MP, je 3 Läufe
    python tools/bench_render.py --quick  # nur 0,5 MP, 1 Lauf

Schreibt:
    reports/render_times.md   (Tabelle)
    plugin/ref_grade/render_seed.json  (lineare Anpassung a + b·MP)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade import session as sess
from refgrade.rawio import save_png

_CORE = Path(__file__).resolve().parents[1]
_REPORTS = _CORE / "reports"
_SEED_OUT = _CORE.parent / "plugin" / "ref_grade" / "render_seed.json"

# Megapixel-Stufen: (breite, höhe) bei 4:3
_SIZES: list[tuple[int, int]] = [
    (800, 600),      # 0.48 MP ≈ 0.5
    (1600, 1200),    # 1.92 MP ≈ 2
    (3200, 2400),    # 7.68 MP ≈ 8
    (5600, 4200),    # 23.52 MP ≈ 24
]
_PREVIEWS = [0.5, 2, 8, 24]  # MP labels
_MAX_SIDE = 1536  # Vorschaugröße wie im Plugin (typisches Viewport)
_METHOD = "reinhard"
_VALUES = {"light_strength": 0.7, "color_strength": 0.5}


def _make_raw(path: str, w: int, h: int, shift: float = 0.0) -> None:
    """Synthetisches BGRA-Rohdatenbild (Diagonalgradient + Versatz)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.clip(0.2 + 0.6 * x / w + shift, 0, 1)
    g = np.clip(0.3 + 0.4 * y / h + shift, 0, 1)
    b = np.clip(0.5 + 0.3 * (x + y) / (w + h) + shift, 0, 1)
    a = np.full((h, w), 255, dtype=np.uint8)
    bgr = np.stack([b, g, r, np.ones_like(r)], axis=-1) * 255
    bgr.astype(np.uint8).tofile(path)


def _make_ref_png(path: str, w: int = 512, h: int = 384) -> None:
    """Synthetische Vorlage als PNG (für refs-Liste)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (0.3 + 0.5 * x / w) % 1.0
    g = (0.4 + 0.4 * y / h) % 1.0
    b = (0.6 + 0.3 * (x - y) / w) % 1.0
    rgb = np.stack([r, g, b], axis=-1).clip(0, 1)
    tensor = torch.from_numpy(rgb).unsqueeze(0)
    save_png(tensor, path)


def _time_op(fn, device: torch.device) -> float:
    """Eine Operation timen (ms); bei CUDA OOM auf CPU wiederholen."""
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(device)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        fn(torch.device("cpu"))
        return (time.perf_counter() - t0) * 1000.0


def _run_bench(sizes: list[tuple[int, int]], runs: int, device: torch.device) -> dict:
    """Alle Größen × alle Wege × alle Läufe; liefert {op: [ms pro Größe]}."""
    results: dict[str, list[float]] = {"preview": [], "apply": [], "cube": []}
    tmpdir = tempfile.mkdtemp(prefix="refgrade_bench_")
    ref_path = os.path.join(tmpdir, "ref.png")
    _make_ref_png(ref_path)
    refs = [ref_path]

    for w, h in sizes:
        mp = w * h / 1_000_000.0
        raw_path = os.path.join(tmpdir, f"img_{w}x{h}.raw")
        _make_raw(raw_path, w, h)

        for i in range(runs):
            s = sess.open_session(raw_path, w, h, "U8", device=device)

            # preview
            def _preview(dev):
                sess.preview(s, refs, _METHOD, _VALUES, _MAX_SIDE, None)
            results["preview"].append(_time_op(_preview, device))

            # apply (volle Größe)
            def _apply(dev):
                sess.apply_full(s, refs, _METHOD, _VALUES, None)
            results["apply"].append(_time_op(_apply, device))

            # cube (LUT-Export, größenunabhängig)
            def _cube(dev):
                sess.write_cube(s, refs, _METHOD, _VALUES, tmpdir, None)
            results["cube"].append(_time_op(_cube, device))

            del s
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  {w}×{h} ({mp:.1f} MP) fertig")

    return results


def _fit_linear(mps: list[float], times: list[float]) -> tuple[float, float]:
    """Lineare Anpassung ms = a + b·MP (kleinste Quadrate)."""
    n = len(mps)
    if n < 2:
        med = statistics.median(times) if times else 0.0
        return med, 0.0
    x = np.array(mps, dtype=np.float64)
    y = np.array(times, dtype=np.float64)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx < 1e-12:
        return float(y_mean), 0.0
    b = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    a = float(y_mean - b * x_mean)
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Renderzeiten messen")
    parser.add_argument("--quick", action="store_true", help="Nur 0,5 MP, 1 Lauf")
    args = parser.parse_args()

    if args.quick:
        sizes = [_SIZES[0]]
        runs = 1
    else:
        sizes = _SIZES
        runs = 3

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Gerät: {device}")
    print(f"Größen: {[(w, h) for w, h in sizes]}, Läufe: {runs}")

    results = _run_bench(sizes, runs, device)

    # Lineare Anpassung pro Operation
    mps_per_size = [w * h / 1_000_000.0 for w, h in sizes]
    seed: dict[str, dict[str, float]] = {}
    for op in ("preview", "apply", "cube"):
        times = results[op]
        if op == "cube":
            # Größenunabhängig: Median
            med = statistics.median(times) if times else 50.0
            seed[op] = {"a": med, "b": 0.0}
        else:
            # Jede Größe hat `runs` Messungen; Median pro Größe nehmen
            n = len(sizes)
            per_size = []
            for i in range(n):
                chunk = times[i * runs:(i + 1) * runs]
                per_size.append(statistics.median(chunk) if chunk else 0.0)
            a, b = _fit_linear(mps_per_size, per_size)
            seed[op] = {"a": round(a, 1), "b": round(b, 1)}

    # render_seed.json schreiben
    _SEED_OUT.parent.mkdir(parents=True, exist_ok=True)
    _SEED_OUT.write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
    print(f"\nrender_seed.json:\n{json.dumps(seed, indent=2)}")

    # render_times.md schreiben
    _REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Renderzeiten (K27e)", "",
             f"Gerät: {device}", f"Vorschau max_side: {_MAX_SIDE} px",
             f"Methode: {_METHOD}", "",
             "| Größe | MP | preview (ms) | apply (ms) | cube (ms) |",
             "|-------|-----|-------------|-----------|-----------|"]
    for i, (w, h) in enumerate(sizes):
        mp = w * h / 1_000_000.0
        pv = statistics.median(results["preview"][i * runs:(i + 1) * runs])
        ap = statistics.median(results["apply"][i * runs:(i + 1) * runs])
        cu = statistics.median(results["cube"][i * runs:(i + 1) * runs])
        lines.append(f"| {w}×{h} | {mp:.1f} | {pv:.0f} | {ap:.0f} | {cu:.0f} |")
    lines += ["",
              "## Lineare Anpassung (ms = a + b·MP)", "",
              "| Operation | a (ms) | b (ms/MP) |",
              "|-----------|--------|-----------|"]
    for op in ("preview", "apply", "cube"):
        lines.append(f"| {op} | {seed[op]['a']:.1f} | {seed[op]['b']:.1f} |")
    (_REPORTS / "render_times.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreports/render_times.md geschrieben")


if __name__ == "__main__":
    main()
