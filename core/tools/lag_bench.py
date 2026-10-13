r"""K29c: Vorschau-Lag messen (Paket K29c).

Aufruf (aus core\):
    python tools/lag_bench.py

Startet einen echten Worker, sendet Preview-Anfragen mit wechselnden Reglerwerten
für verschiedene max_side-Werte und Stufen. Misst die Gesamtzeit (HTTP) und
misst die Teilzeiten direkt (sess.preview, Scopes, PNG, JSON).

Schreibt:
    reports/lag_bench.md   (Tabelle)
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
REPORTS_DIR = PROJECT_DIR / "reports"
PYTHON = sys.executable

MAX_SIDES = [2560, 1714, 1280, 856, 428]
STAGES = ["lut", "full"]
N_REQUESTS = 10
CLARITY = 0.3
SCOPE_SMALL = 512  # Scopes-Kandidat: verkleinertes Bild


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 30) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _p90(vals: list[float]) -> float:
    vals = sorted(vals)
    idx = 0.9 * (len(vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _varying_values(i: int) -> dict:
    """Different slider values per request to prevent caching."""
    return {
        "light_strength": 0.5 + 0.05 * (i % 5),
        "color_strength": 0.3 + 0.05 * (i % 3),
        "clarity": CLARITY if i % 2 == 0 else 0.0,
    }


def _nvidia_smi() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip().replace("\n", "; ")
    except Exception:
        return "nvidia-smi nicht verfügbar"


def _png_decode_time(path: str) -> float:
    """PNG decode via PIL (Proxy für QImage(path) im Plugin-Prozess)."""
    from PIL import Image
    t0 = time.perf_counter()
    img = Image.open(path)
    img.load()
    return (time.perf_counter() - t0) * 1000.0


def _png_write_bench(arr_rgb: np.ndarray, n: int = 5) -> dict[str, float]:
    """PNG-Write-Zeit für Kompressionsstufen default/0/1 messen."""
    from PIL import Image
    pil_img = Image.fromarray(arr_rgb, mode="RGB")
    times: dict[str, float] = {}
    for label, kw in [("default", {}), ("level0", {"compress_level": 0}),
                      ("level1", {"compress_level": 1})]:
        samples: list[float] = []
        for _ in range(n):
            p = str(Path(tempfile.gettempdir()) / f"lagbench_{label}_{id(pil_img)}.png")
            t0 = time.perf_counter()
            pil_img.save(p, format="PNG", **kw)
            samples.append((time.perf_counter() - t0) * 1000.0)
            os.remove(p)
        times[label] = statistics.median(samples)
    return times


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="refgrade_lagbench_")
    env = os.environ.copy()

    print(f"Starte Worker auf Port {port} ...")
    proc = subprocess.Popen(
        [PYTHON, "-m", "worker", "--port", str(port), "--tmp", tmp],
        cwd=str(CORE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(f"Worker sofort beendet: {err.decode()}")
        try:
            r = _get(f"{base}/health")
            if r.get("ok"):
                break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        raise RuntimeError("Worker nicht innerhalb von 30 s erreichbar")

    gpu_info = _nvidia_smi()
    print(f"GPU: {gpu_info}")

    # Load source image (2000 px) and a 4000 px upscaled version
    from PIL import Image
    jpg_2000 = str(POOL_DIR / "04.jpg")
    if not Path(jpg_2000).exists():
        jpg_2000 = str(POOL_DIR / "01.jpg")

    def _jpg_to_raw(jpg_path: str, tag: str) -> tuple[str, int, int]:
        """Convert JPEG to BGRA U8 raw file; return (raw_path, width, height)."""
        img = Image.open(jpg_path).convert("RGB")
        w, h = img.size
        arr = np.array(img, dtype=np.uint8)  # H,W,3 RGB
        bgra = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0],
                         np.full((h, w), 255, dtype=np.uint8)], axis=-1)
        raw_path = str(Path(tmp) / f"{tag}.raw")
        bgra.tofile(raw_path)
        return raw_path, w, h

    src_2000, w2000, h2000 = _jpg_to_raw(jpg_2000, "src_2000")

    # Upscale to 4000 px
    img = Image.open(jpg_2000).convert("RGB")
    w, h = img.size
    scale = 4000 / max(w, h)
    img_4000 = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    jpg_4000 = str(Path(tmp) / "upscaled_4000.jpg")
    img_4000.save(jpg_4000, quality=95)
    src_4000, w4000, h4000 = _jpg_to_raw(jpg_4000, "src_4000")

    ref_path = str(POOL_DIR / "01.jpg")

    results: list[dict] = []

    for src_label, src_path, sw, sh in [("2000px", src_2000, w2000, h2000),
                                         ("4000px", src_4000, w4000, h4000)]:
        # Load session
        r = _post(f"{base}/load", {"raw": src_path, "width": sw, "height": sh, "depth": "U8"})
        if "error" in r:
            print(f"  WARN: load {src_label} fehlgeschlagen: {r['error']}")
            continue
        sid = r["session"]
        print(f"  {src_label}: session {sid[:8]}...")

        for max_side in MAX_SIDES:
            for stage in STAGES:
                times_http: list[float] = []
                times_server: list[float] = []
                times_png_decode: list[float] = []
                oom = False
                for i in range(N_REQUESTS):
                    payload = {
                        "session": sid,
                        "references": [ref_path],
                        "method": "reinhard",
                        "values": _varying_values(i),
                        "max_side": max_side,
                        "mix": {"weights": {}, "master": 0.5},
                        "stage": stage,
                    }
                    t0 = time.perf_counter()
                    try:
                        resp = _post(f"{base}/preview", payload, timeout=120)
                    except Exception as e:
                        if "OutOfMemory" in str(e) or "out of memory" in str(e).lower():
                            oom = True
                            break
                        print(f"  WARN: {max_side}/{stage} req {i}: {e}")
                        break
                    t_http = (time.perf_counter() - t0) * 1000.0
                    t_server = resp.get("ms", 0.0)
                    img_path = resp.get("image", "")
                    t_png = 0.0
                    if img_path and Path(img_path).exists():
                        t_png = _png_decode_time(img_path)
                    times_http.append(t_http)
                    times_server.append(t_server)
                    times_png_decode.append(t_png)
                if oom:
                    results.append({"src": src_label, "max_side": max_side, "stage": stage,
                                    "oom": True, "n": len(times_http)})
                    print(f"  {src_label} {max_side}/{stage}: CUDA-OOM (Umgebungsgrenze)")
                    continue
                if not times_http:
                    continue
                results.append({
                    "src": src_label, "max_side": max_side, "stage": stage,
                    "n": len(times_http),
                    "http_median": statistics.median(times_http),
                    "http_p90": _p90(times_http),
                    "server_median": statistics.median(times_server),
                    "server_p90": _p90(times_server),
                    "png_decode_median": statistics.median(times_png_decode),
                })
                print(f"  {src_label} {max_side}/{stage}: "
                      f"HTTP med={statistics.median(times_http):.0f}ms "
                      f"P90={_p90(times_http):.0f}ms | "
                      f"Server med={statistics.median(times_server):.0f}ms | "
                      f"PNG decode med={statistics.median(times_png_decode):.1f}ms")

    # Direct sub-timing: in-process, no HTTP overhead
    print("\nTeilzeiten (direkt, ohne HTTP):")
    from refgrade import session as sess
    from refgrade.scope import vectorscope, rgb_parade, clipping
    from refgrade.session import _histogram_l, _histogram_rgb
    import worker as _worker
    from worker import _save_png_tmp
    import torch
    # in-process worker module: its temp dir is unset (cwd) — left 222 files / 486 MB in core\
    # (2026-10-11); write into the bench's own temp dir instead
    _worker._tmp_dir = tmp

    sub_results: list[dict] = []
    for src_label, src_path, sw, sh in [("2000px", src_2000, w2000, h2000),
                                         ("4000px", src_4000, w4000, h4000)]:
        try:
            s = sess.open_session(src_path, sw, sh, "U8")
        except Exception as e:
            print(f"  WARN: open_session {src_label}: {e}")
            continue
        for max_side in MAX_SIDES:
            for stage in STAGES:
                t_compute: list[float] = []
                t_scopes: list[float] = []
                t_png: list[float] = []
                t_json: list[float] = []
                for i in range(N_REQUESTS):
                    vals = _varying_values(i)
                    # 1. Compute
                    t0 = time.perf_counter()
                    img, hists = sess.preview(s, [ref_path], "reinhard", vals, max_side,
                                              {"weights": {}, "master": 0.5}, stage=stage)
                    t_compute.append((time.perf_counter() - t0) * 1000.0)
                    # 2. Scopes (nur Ergebnisbild, ohne Quellbild-Vectorscope)
                    t0 = time.perf_counter()
                    vs = vectorscope(img)
                    parade = rgb_parade(img)
                    clip = clipping(img)
                    hists["result"] = _histogram_l(img)
                    hists["rgb"]["result"] = _histogram_rgb(img)
                    t_scopes.append((time.perf_counter() - t0) * 1000.0)
                    # 3. PNG write
                    t0 = time.perf_counter()
                    img_path = _save_png_tmp(img, "lagbench")
                    t_png.append((time.perf_counter() - t0) * 1000.0)
                    # 4. JSON (Parade-Tensoren in Listen; der Worker sendet sie als PNG-Pfade)
                    t0 = time.perf_counter()
                    parade_json = {k: v.tolist() for k, v in parade.items()}
                    _ = json.dumps({"image": img_path, "histogram": hists,
                                    "vectorscope": vs, "parade": parade_json, "clip": clip})
                    t_json.append((time.perf_counter() - t0) * 1000.0)
                sub_results.append({
                    "src": src_label, "max_side": max_side, "stage": stage,
                    "compute_median": statistics.median(t_compute),
                    "scopes_median": statistics.median(t_scopes),
                    "png_median": statistics.median(t_png),
                    "json_median": statistics.median(t_json),
                })
                print(f"  {src_label} {max_side}/{stage}: "
                      f"compute={statistics.median(t_compute):.0f}ms "
                      f"scopes={statistics.median(t_scopes):.0f}ms "
                      f"png={statistics.median(t_png):.0f}ms "
                      f"json={statistics.median(t_json):.0f}ms")

    # --- Kandidaten-Messungen ---
    print("\nKandidaten-Messungen:")

    # 1) PNG-Kompression: Write-Zeit bei 2560 px mit verschiedenen Stufen
    png_comp_results: dict[str, float] = {}
    try:
        s2000 = sess.open_session(src_2000, w2000, h2000, "U8")
        img_2560, _ = sess.preview(s2000, [ref_path], "reinhard",
                                   {"light_strength": 0.5, "color_strength": 0.3, "clarity": 0.0},
                                   2560, {"weights": {}, "master": 0.5}, stage="lut")
        arr = np.clip(img_2560.squeeze(0).cpu().numpy(), 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        png_comp_results = _png_write_bench(arr, n=5)
        print(f"  PNG-Write 2560px: default={png_comp_results['default']:.0f}ms "
              f"level0={png_comp_results['level0']:.0f}ms level1={png_comp_results['level1']:.0f}ms")
    except Exception as e:
        print(f"  WARN: PNG-Kompression-Messung fehlgeschlagen: {e}")

    # 2) Scopes bei 512 px (verkleinertes Bild)
    scope_small_results: dict[str, float] = {}
    try:
        s2000 = sess.open_session(src_2000, w2000, h2000, "U8")
        img_512, _ = sess.preview(s2000, [ref_path], "reinhard",
                                  {"light_strength": 0.5, "color_strength": 0.3, "clarity": 0.0},
                                  SCOPE_SMALL, {"weights": {}, "master": 0.5}, stage="lut")
        t_list: list[float] = []
        for _ in range(N_REQUESTS):
            t0 = time.perf_counter()
            vs = vectorscope(img_512)
            parade = rgb_parade(img_512)
            clip = clipping(img_512)
            h = _histogram_l(img_512)
            hr = _histogram_rgb(img_512)
            t_list.append((time.perf_counter() - t0) * 1000.0)
        scope_small_results["512px_median"] = statistics.median(t_list)
        # Vergleich: Scopes bei 2560 px (aus sub_results)
        ref_2560 = next((r for r in sub_results
                         if r["src"] == "2000px" and r["max_side"] == 2560 and r["stage"] == "lut"),
                        None)
        scope_small_results["2560px_median"] = ref_2560["scopes_median"] if ref_2560 else 0.0
        print(f"  Scopes 512px med={scope_small_results['512px_median']:.0f}ms "
              f"vs 2560px med={scope_small_results['2560px_median']:.0f}ms")
    except Exception as e:
        print(f"  WARN: Scopes-512-Messung fehlgeschlagen: {e}")

    # --- Bericht schreiben ---
    lines = [
        "# Lag-Bench (K29c)",
        "",
        f"GPU: {gpu_info}",
        "",
        "**Alle Zahlen gemessen unter LLM-GPU-Last (BeeLlama belegt ~21 GB VRAM).**",
        "",
        "## HTTP-Gesamtzeit (Worker-Anfrage, inkl. Netzwerk)",
        "",
        "PNG-Decode = PIL-Decode der Antwort-PNG (Proxy für QImage(path) im Plugin-Prozess).",
        "",
        "| Quelle | max_side | Stufe | n | HTTP Median (ms) | HTTP P90 (ms) | Server ms (Median) | PNG-Decode (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("oom"):
            lines.append(f"| {r['src']} | {r['max_side']} | {r['stage']} | – | CUDA-OOM | – | – | – |")
        else:
            lines.append(
                f"| {r['src']} | {r['max_side']} | {r['stage']} | {r['n']} "
                f"| {r['http_median']:.0f} | {r['http_p90']:.0f} "
                f"| {r['server_median']:.0f} | {r['png_decode_median']:.1f} |")

    lines += [
        "",
        "## Teilzeiten (in-process, ohne HTTP-Overhead)",
        "",
        "Teilzeiten werden direkt in `sess.preview` / Scope-Funktionen / `_save_png_tmp` / "
        "`json.dumps` gemessen (kein Worker, kein Netzwerk).",
        "",
        "| Quelle | max_side | Stufe | Rechnen (ms) | Scopes (ms) | PNG-Write (ms) | JSON (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sub_results:
        lines.append(
            f"| {r['src']} | {r['max_side']} | {r['stage']} "
            f"| {r['compute_median']:.0f} | {r['scopes_median']:.0f} "
            f"| {r['png_median']:.0f} | {r['json_median']:.0f} |")

    # Bewertung mit gemessenen Zahlen
    lines += ["", "## Bewertung weiterer Beschleunigungskandidaten", ""]

    # Drag-Halbierung: echte Zahlen aus den Tabellen
    ref_2560 = next((r for r in sub_results
                     if r["src"] == "2000px" and r["max_side"] == 2560 and r["stage"] == "lut"), None)
    ref_1280 = next((r for r in sub_results
                     if r["src"] == "2000px" and r["max_side"] == 1280 and r["stage"] == "lut"), None)
    if ref_2560 and ref_1280:
        sum_2560 = ref_2560["compute_median"] + ref_2560["scopes_median"] + ref_2560["png_median"] + ref_2560["json_median"]
        sum_1280 = ref_1280["compute_median"] + ref_1280["scopes_median"] + ref_1280["png_median"] + ref_1280["json_median"]
        speedup = sum_2560 / sum_1280 if sum_1280 > 0 else 0
        lines.append(
            f"- **Drag-Halbierung (eingebaut):** max_side // 2 beim Ziehen. "
            f"Teilzeiten-Summe 2560 px = {sum_2560:.0f} ms, 1280 px = {sum_1280:.0f} ms "
            f"→ Faktor {speedup:.2f}× schneller. "
            f"(Rechnen: {ref_2560['compute_median']:.0f}→{ref_1280['compute_median']:.0f} ms, "
            f"Scopes: {ref_2560['scopes_median']:.0f}→{ref_1280['scopes_median']:.0f} ms, "
            f"PNG: {ref_2560['png_median']:.0f}→{ref_1280['png_median']:.0f} ms.)")

    # PNG-Kompression
    if png_comp_results:
        lines.append(
            f"- **PNG-Kompression:** Write-Zeit 2560 px — default: "
            f"{png_comp_results['default']:.0f} ms, level 0: {png_comp_results['level0']:.0f} ms, "
            f"level 1: {png_comp_results['level1']:.0f} ms. "
            f"Level 0 spart {png_comp_results['default'] - png_comp_results['level0']:.0f} ms "
            f"({(png_comp_results['default'] - png_comp_results['level0']) / png_comp_results['default'] * 100:.0f} %).")

    # Scopes 512 px
    if scope_small_results:
        s512 = scope_small_results["512px_median"]
        s2560 = scope_small_results["2560px_median"]
        factor = s2560 / s512 if s512 > 0 else 0
        lines.append(
            f"- **Scopes aus verkleinertem Bild (≤ 512 px):** Scope-Zeit 512 px = {s512:.0f} ms "
            f"vs 2560 px = {s2560:.0f} ms → Faktor {factor:.1f}× kleiner. "
            f"Beim Ziehen wären die Scopes ungenauer, aber die Latenz sinkt um "
            f"{s2560 - s512:.0f} ms.")

    lines.append("")

    report = "\n".join(lines)
    out_path = REPORTS_DIR / "lag_bench.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nBericht: {out_path}")

    # Cleanup
    try:
        _post(f"{base}/shutdown", {})
    except Exception:
        pass
    proc.wait(timeout=5)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
