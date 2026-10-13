"""Reaktionszeit je Regler + Teilzeiten der Vorschau (Opus, 2026-10-11).

python tools/latency_quick.py            # Zustand aus RG_GPU_STATE (normal|compressed|paused)
Hängt einen Abschnitt an reports/latency_quick.md (Projektwurzel) an.

Teil A (im Prozess, wie worker._handle_preview): je Schritt ms, 1280 und 2560, 6 frische Werte.
Teil B (HTTP, echter Worker): jeder sichtbare Regler 3 frische Werte, max_side 1280, stage lut.
"""

from __future__ import annotations

import json
import os
import socket
import statistics as st
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CORE = Path(__file__).resolve().parent.parent
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

from refgrade import session as sess  # noqa: E402
from refgrade.scope import clipping, rgb_parade, vectorscope  # noqa: E402
from refgrade.session import _histogram_l, _histogram_rgb  # noqa: E402
from refgrade.sections import VISIBLE  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402

POOL = ROOT / "bench" / "pool"
SRC, REF = POOL / "04.jpg", str(POOL / "02.jpg")
STATE = os.environ.get("RG_GPU_STATE", "normal")
OUT = ROOT / "reports" / "latency_quick.md"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def raw_of(jpg: Path, tmp: str) -> tuple[str, int, int]:
    a = np.array(Image.open(jpg).convert("RGB"), dtype=np.uint8)
    h, w = a.shape[:2]
    bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)])
    p = os.path.join(tmp, "src.raw")
    bgra.tofile(p)
    return p, w, h


def fresh(name: str, i: int) -> float:
    s = next(x for x in SLIDERS if x.name == name)
    # wechselnde Werte, nie Standard, kein Cache-Treffer
    f = (0.15 + 0.11 * i) % 0.9 + 0.05
    v = s.min + f * (s.max - s.min)
    return round(v if abs(v - s.default) > 1e-3 else v + s.step, 4)


def part_a(tmp: str) -> list[str]:
    raw, w, h = raw_of(SRC, tmp)
    s = sess.open_session(raw, w, h, "U8")
    rows = ["| max_side | Schritt | Median ms | P90 ms |", "|---|---|---|---|"]
    for ms in (1280, 2560):
        steps: dict[str, list[float]] = {}
        sess.preview(s, [REF], "reinhard", {}, ms, None, stage="lut")  # Aufwärmen (Methode, Cache)
        for i in range(6):
            vals = {"exposure": fresh("exposure", i), "gain_r": fresh("gain_r", i)}
            t = {}
            sync(); t0 = time.perf_counter()
            lut = sess.build_lut(s, [REF], "reinhard", vals, None); sync(); t["build_lut"] = time.perf_counter()
            img = sess._lut_apply(sess._preview_source(s, ms), lut); sync(); t["lut_apply"] = time.perf_counter()
            _histogram_l(img); _histogram_rgb(img); t["histogramme"] = time.perf_counter()
            path = os.path.join(tmp, f"p{i}.png")
            from refgrade.rawio import save_png  # type: ignore
            save_png(img.cpu(), path); t["png_schreiben"] = time.perf_counter()
            vectorscope(img); sync(); t["vectorscope_ergebnis"] = time.perf_counter()
            vectorscope(s.source); sync(); t["vectorscope_original_voll"] = time.perf_counter()
            rgb_parade(img); sync(); t["parade"] = time.perf_counter()
            clipping(img); sync(); t["clipping"] = time.perf_counter()
            prev = t0
            for k, v in t.items():
                steps.setdefault(k, []).append((v - prev) * 1000)
                prev = v
            steps.setdefault("SUMME", []).append((prev - t0) * 1000)
        for k, v in steps.items():
            v = sorted(v)
            rows.append(f"| {ms} | {k} | {st.median(v):.1f} | {v[int(0.9 * (len(v) - 1))]:.1f} |")
    return rows


def _port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def _post(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def part_b(tmp: str) -> list[str]:
    port = _port()
    proc = subprocess.Popen([sys.executable, "-m", "worker", "--port", str(port), "--tmp", tmp],
                            cwd=str(CORE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except OSError:
                time.sleep(0.3)
        raw, w, h = raw_of(SRC, tmp)
        sid = _post(base + "/load", {"raw": raw, "width": w, "height": h, "depth": "U8"})["session"]
        body = {"session": sid, "references": [REF], "method": "reinhard", "max_side": 1280, "stage": "lut"}
        _post(base + "/preview", {**body, "values": {}})
        res = []
        for name in VISIBLE:
            ts = []
            for i in range(3):
                t0 = time.perf_counter()
                _post(base + "/preview", {**body, "values": {name: fresh(name, i)}})
                ts.append((time.perf_counter() - t0) * 1000)
            res.append((name, st.median(ts), max(ts)))
        res.sort(key=lambda r: -r[1])
        rows = ["| Regler | Median ms | Max ms |", "|---|---|---|"]
        rows += [f"| {n} | {m:.0f} | {x:.0f} |" for n, m, x in res]
        allm = [r[1] for r in res]
        rows.insert(0, f"Alle {len(res)} Regler: Median {st.median(allm):.0f} ms, "
                       f"min {min(allm):.0f}, max {max(allm):.0f}\n")
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="refgrade_latq_") as tmp:
        a = part_a(tmp)
        b = part_b(tmp)
    text = [f"\n## Zustand `{STATE}` — {time.strftime('%Y-%m-%d %H:%M')} — VRAM vor Start: {vram}\n",
            "### A — Teilzeiten im Worker-Ablauf (exposure + gain_r, frische Werte)\n", *a,
            "\n### B — HTTP Ende-zu-Ende je Regler (1280 px, stage lut)\n", *b, ""]
    with open(OUT, "a", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
