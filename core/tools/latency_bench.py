"""K31a: Reaktionszeit jedes Reglers messen (Bench + Bericht).

Aufruf (aus core\\):
    python tools/latency_bench.py              # voll: 1280+2560, 10 Werte je Regler
    python tools/latency_bench.py --quick      # schnell: nur 1280, 3 Werte je Regler

Liest RG_GPU_STATE (normal|compressed|paused) und RG_GPU_VRAM aus der Umgebung.
Hängt einen Abschnitt an reports/latency.md an (überschreibt nicht).
Schreibt reports/latency.png (gestapelte Balken je Bereich).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
REPORTS_DIR = PROJECT_DIR / "reports"
PYTHON = sys.executable

sys.path.insert(0, str(CORE_DIR))

from refgrade.sections import SECTIONS, VISIBLE  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402


def _free_port() -> int:
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


def _p95(vals: list[float]) -> float:
    vals = sorted(vals)
    idx = 0.95 * (len(vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _fresh_value(name: str, i: int) -> float:
    """Wechselnder Wert, nie Standard, kein Cache-Treffer."""
    s = next(x for x in SLIDERS if x.name == name)
    f = (0.15 + 0.11 * i) % 0.9 + 0.05
    v = s.min + f * (s.max - s.min)
    if abs(v - s.default) < 1e-3:
        v = v + s.step
    return round(v, 4)


def _nvidia_smi() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip().replace("\n", "; ")
    except Exception:
        return "nvidia-smi nicht verfügbar"


def _jpg_to_raw(jpg_path: str, tmp: str) -> tuple[str, int, int]:
    from PIL import Image
    img = Image.open(jpg_path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)
    bgra = np.stack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0],
                     np.full((h, w), 255, dtype=np.uint8)], axis=-1)
    raw_path = os.path.join(tmp, "src.raw")
    bgra.tofile(raw_path)
    return raw_path, w, h


def _slider_group(name: str) -> str:
    """Bereichsschlüssel eines Reglers (aus SECTIONS)."""
    for key, names in SECTIONS.items():
        if name in names:
            return key
    return "other"


def _bench_sliders(base: str, sid: str, ref: str, max_side: int, stage: str,
                   n: int) -> list[dict]:
    """Jeder sichtbare Regler: n Änderungen, timing:true."""
    results: list[dict] = []
    for name in VISIBLE:
        times: list[float] = []
        timings: list[dict] = []
        for i in range(n):
            body = {
                "session": sid,
                "references": [ref],
                "method": "reinhard",
                "values": {name: _fresh_value(name, i)},
                "max_side": max_side,
                "mix": {"weights": {}, "master": 0.5},
                "stage": stage,
                "timing": True,
            }
            t0 = time.perf_counter()
            resp = _post(f"{base}/preview", body)
            t_e2e = (time.perf_counter() - t0) * 1000.0
            times.append(t_e2e)
            timings.append(resp.get("timing", {}))
        med = statistics.median(times)
        p90 = _p90(times)
        # Aggregierte Teilzeiten (Median)
        agg: dict[str, float] = {}
        for t in timings:
            for k, v in t.items():
                agg.setdefault(k, []).append(v)
        agg_med = {k: statistics.median(v) for k, v in agg.items()}
        results.append({
            "name": name,
            "group": _slider_group(name),
            "median": med,
            "p90": p90,
            "timing": agg_med,
        })
    return results


def _bench_curves(base: str, sid: str, ref: str, max_side: int, stage: str,
                  n: int) -> list[dict]:
    """Kurve: ein Punkt verschoben, n Änderungen."""
    results: list[dict] = []
    times: list[float] = []
    timings: list[dict] = []
    for i in range(n):
        # Kurve mit einem verschobenen Punkt (R-Kanal)
        x = 0.2 + 0.1 * (i % 5)
        y = 0.5 + 0.05 * (i % 3)
        mix = {
            "weights": {"curves": 1.0},
            "master": 0.5,
            "curves": {"r": [[0.0, 0.0], [x, y], [1.0, 1.0]]},
        }
        body = {
            "session": sid,
            "references": [ref],
            "method": "reinhard",
            "values": {},
            "max_side": max_side,
            "mix": mix,
            "stage": stage,
            "timing": True,
        }
        t0 = time.perf_counter()
        resp = _post(f"{base}/preview", body)
        times.append((time.perf_counter() - t0) * 1000.0)
        timings.append(resp.get("timing", {}))
    agg: dict[str, list[float]] = {}
    for t in timings:
        for k, v in t.items():
            agg.setdefault(k, []).append(v)
    agg_med = {k: statistics.median(v) for k, v in agg.items()}
    results.append({
        "name": "curves (1 Punkt)",
        "group": "curves",
        "median": statistics.median(times),
        "p90": _p90(times),
        "timing": agg_med,
    })
    return results


def _bench_lut_weight(base: str, sid: str, ref: str, max_side: int, stage: str,
                      n: int, lut_path: str) -> list[dict]:
    """LUT-Gewicht: n Änderungen."""
    results: list[dict] = []
    times: list[float] = []
    timings: list[dict] = []
    for i in range(n):
        w = 0.1 + 0.15 * (i % 6)
        mix = {
            "weights": {"lut": w},
            "master": 0.5,
            "lut": lut_path,
        }
        body = {
            "session": sid,
            "references": [ref],
            "method": "reinhard",
            "values": {},
            "max_side": max_side,
            "mix": mix,
            "stage": stage,
            "timing": True,
        }
        t0 = time.perf_counter()
        resp = _post(f"{base}/preview", body)
        times.append((time.perf_counter() - t0) * 1000.0)
        timings.append(resp.get("timing", {}))
    agg: dict[str, list[float]] = {}
    for t in timings:
        for k, v in t.items():
            agg.setdefault(k, []).append(v)
    agg_med = {k: statistics.median(v) for k, v in agg.items()}
    results.append({
        "name": "LUT-Gewicht",
        "group": "lut",
        "median": statistics.median(times),
        "p90": _p90(times),
        "timing": agg_med,
    })
    return results


def _bench_master(base: str, sid: str, ref: str, max_side: int, stage: str,
                  n: int) -> list[dict]:
    """Master: n Änderungen."""
    times: list[float] = []
    timings: list[dict] = []
    for i in range(n):
        m = 0.1 + 0.15 * (i % 6)
        body = {
            "session": sid,
            "references": [ref],
            "method": "reinhard",
            "values": {"exposure": 0.3, "saturation": 1.5},
            "max_side": max_side,
            "mix": {"weights": {}, "master": m},
            "stage": stage,
            "timing": True,
        }
        t0 = time.perf_counter()
        resp = _post(f"{base}/preview", body)
        times.append((time.perf_counter() - t0) * 1000.0)
        timings.append(resp.get("timing", {}))
    agg: dict[str, list[float]] = {}
    for t in timings:
        for k, v in t.items():
            agg.setdefault(k, []).append(v)
    agg_med = {k: statistics.median(v) for k, v in agg.items()}
    return [{
        "name": "Master",
        "group": "master",
        "median": statistics.median(times),
        "p90": _p90(times),
        "timing": agg_med,
    }]


def _bench_sections_onoff(base: str, sid: str, ref: str, max_side: int,
                          stage: str, n: int) -> list[dict]:
    """Jeder Bereich ON/OFF: n Änderungen."""
    results: list[dict] = []
    for sec_key in SECTIONS:
        if sec_key in ("curves", "lut"):
            continue
        times: list[float] = []
        timings: list[dict] = []
        for i in range(n):
            on = (i % 2 == 0)
            weights = {sec_key: 1.0 if on else 0.0}
            body = {
                "session": sid,
                "references": [ref],
                "method": "reinhard",
                "values": {name: _fresh_value(name, i)
                           for name in SECTIONS[sec_key][:3]},
                "max_side": max_side,
                "mix": {"weights": weights, "master": 0.5},
                "stage": stage,
                "timing": True,
            }
            t0 = time.perf_counter()
            resp = _post(f"{base}/preview", body)
            times.append((time.perf_counter() - t0) * 1000.0)
            timings.append(resp.get("timing", {}))
        agg: dict[str, list[float]] = {}
        for t in timings:
            for k, v in t.items():
                agg.setdefault(k, []).append(v)
        agg_med = {k: statistics.median(v) for k, v in agg.items()}
        results.append({
            "name": f"Bereich {sec_key} ON/OFF",
            "group": sec_key,
            "median": statistics.median(times),
            "p90": _p90(times),
            "timing": agg_med,
        })
    return results


def _bench_drag(base: str, sid: str, ref: str, max_side: int,
                n: int = 30) -> list[float]:
    """Ziehen: n /preview nacheinander, exposure 0.00 in Schritten 0.05 (echte
    Wertänderung, kein LUT-Cache-Treffer), lut_size 17, scopes nur Histogramm."""
    times: list[float] = []
    for i in range(n):
        exposure = round(0.05 * i, 4)
        body = {
            "session": sid,
            "references": [ref],
            "method": "reinhard",
            "values": {"exposure": exposure},
            "max_side": max_side,
            "mix": {"weights": {}, "master": 0.5},
            "scopes": ["hist"],
            "lut_size": 17,
            "timing": True,
        }
        t0 = time.perf_counter()
        _post(f"{base}/preview", body)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _make_chart(all_results: list[dict], path: Path) -> None:
    """Gestapelte Balken je Bereich (PIL)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  PIL nicht verfügbar, keine Grafik")
        return

    # Aggregiere Teilzeiten je Bereich
    group_timings: dict[str, dict[str, float]] = {}
    for r in all_results:
        g = r["group"]
        if g not in group_timings:
            group_timings[g] = {}
        for k, v in r["timing"].items():
            group_timings[g][k] = group_timings[g].get(k, 0.0) + v

    # Top-10 Bereiche nach Gesamtzeit
    sorted_groups = sorted(group_timings.items(),
                           key=lambda x: -sum(x[1].values()))[:10]

    # Farben für die Top-8 Teilschritte
    colors = [
        (220, 50, 50), (50, 150, 50), (50, 100, 220), (220, 180, 50),
        (180, 50, 180), (50, 180, 180), (180, 120, 50), (100, 100, 100),
    ]

    # Top-8 Teilschritte über alle Bereiche
    all_keys: dict[str, float] = {}
    for _, t in sorted_groups:
        for k, v in t.items():
            all_keys[k] = all_keys.get(k, 0.0) + v
    top_keys = [k for k, _ in sorted(all_keys.items(), key=lambda x: -x[1])[:8]]

    W, H = 900, 500
    margin_l, margin_b, margin_t, margin_r = 120, 60, 40, 20
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    bar_h = min(40, plot_h // len(sorted_groups) - 4)

    max_val = max(sum(t.values()) for _, t in sorted_groups) if sorted_groups else 1.0
    max_val = max(max_val, 1.0)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()

    for idx, (gname, timings) in enumerate(sorted_groups):
        y = margin_t + idx * (bar_h + 4)
        x = margin_l
        for k in top_keys:
            v = timings.get(k, 0.0)
            w = int(v / max_val * plot_w)
            ci = top_keys.index(k)
            draw.rectangle([x, y, x + w, y + bar_h], fill=colors[ci % len(colors)])
            x += w
        draw.text((margin_l - 5, y + bar_h // 2 - 6), gname, fill=(0, 0, 0),
                  font=font, anchor="ra")

    # Legende
    ly = H - margin_b + 10
    lx = margin_l
    for k in top_keys:
        ci = top_keys.index(k)
        draw.rectangle([lx, ly, lx + 12, ly + 12], fill=colors[ci % len(colors)])
        draw.text((lx + 15, ly), k, fill=(0, 0, 0), font=font)
        lx += 15 + len(k) * 7 + 15

    img.save(str(path))
    print(f"  Grafik: {path}")


def _run_drag() -> int:
    """--drag: 30 /preview mit echter Wertänderung, Median + p95 der Antwortzeit."""
    max_side = 640  # halbe normale Größe (1280)
    n = 30
    print(f"K31a Latency-Bench (drag: {n} /preview, exposure 0.00..1.45, "
          f"lut_size 17, max_side {max_side}, scopes hist)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="refgrade_latbench_")
    env = os.environ.copy()

    print(f"  Worker auf Port {port} ...")
    proc = subprocess.Popen(
        [PYTHON, "-m", "worker", "--port", str(port), "--tmp", tmp],
        cwd=str(CORE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
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
            raise RuntimeError("Worker nicht innerhalb von 30 s erreichbar")

        src_jpg = str(POOL_DIR / "04.jpg")
        if not Path(src_jpg).exists():
            src_jpg = str(POOL_DIR / "01.jpg")
        ref_jpg = str(POOL_DIR / "02.jpg")
        if not Path(ref_jpg).exists():
            ref_jpg = str(POOL_DIR / "01.jpg")

        raw, w, h = _jpg_to_raw(src_jpg, tmp)
        r = _post(f"{base}/load", {"raw": raw, "width": w, "height": h, "depth": "U8"})
        sid = r["session"]
        print(f"  Session: {sid[:8]}... ({w}x{h})")
        print(f"  GPU: {_nvidia_smi()}")

        times = _bench_drag(base, sid, ref_jpg, max_side, n)
        med = statistics.median(times)
        p95 = _p95(times)
        print(f"\n  Ziehen (echte Wertänderung): Median {med:.1f} ms, p95 {p95:.1f} ms "
              f"({n} Anfragen, max_side {max_side}, lut_size 17, scopes hist)")
        return 0
    except Exception as e:
        print(f"  FEHLER: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            _post(f"{base}/shutdown", {})
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="K31a: Reaktionszeit-Bench")
    parser.add_argument("--quick", action="store_true",
                        help="Nur 1280 px, 3 Werte je Regler")
    parser.add_argument("--drag", action="store_true",
                        help="Ziehen: 30 /preview mit echter Wertänderung (exposure)")
    args = parser.parse_args()

    if args.drag:
        return _run_drag()

    quick = args.quick
    max_sides = [1280] if quick else [1280, 2560]
    n_values = 3 if quick else 10

    gpu_state = os.environ.get("RG_GPU_STATE", "normal")
    gpu_vram = os.environ.get("RG_GPU_VRAM", _nvidia_smi())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="refgrade_latbench_")
    env = os.environ.copy()

    print(f"K31a Latency-Bench ({'quick' if quick else 'voll'})")
    print(f"  GPU-Zustand: {gpu_state}")
    print(f"  VRAM: {gpu_vram}")
    print(f"  max_sides: {max_sides}, n_werte: {n_values}")
    print(f"  Worker auf Port {port} ...")

    proc = subprocess.Popen(
        [PYTHON, "-m", "worker", "--port", str(port), "--tmp", tmp],
        cwd=str(CORE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"

    try:
        # Warten bis Worker bereit
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
            raise RuntimeError("Worker nicht innerhalb von 30 s erreichbar")

        gpu_info = _nvidia_smi()
        print(f"  GPU: {gpu_info}")

        # Bild laden
        src_jpg = str(POOL_DIR / "04.jpg")
        if not Path(src_jpg).exists():
            src_jpg = str(POOL_DIR / "01.jpg")
        ref_jpg = str(POOL_DIR / "02.jpg")
        if not Path(ref_jpg).exists():
            ref_jpg = str(POOL_DIR / "01.jpg")

        raw, w, h = _jpg_to_raw(src_jpg, tmp)
        r = _post(f"{base}/load", {"raw": raw, "width": w, "height": h, "depth": "U8"})
        sid = r["session"]
        print(f"  Session: {sid[:8]}... ({w}x{h})")

        # LUT-Datei für LUT-Gewicht-Bench
        luts_dir = PROJECT_DIR / "luts"
        lut_path = ""
        if luts_dir.exists():
            for f in luts_dir.rglob("*.cube"):
                lut_path = str(f)
                break

        all_results: list[dict] = []

        for max_side in max_sides:
            for stage in ("lut", "full"):
                label = f"{max_side} px / {stage}"
                print(f"\n  [{label}] Regler ...")
                t0 = time.perf_counter()
                results = _bench_sliders(base, sid, ref_jpg, max_side, stage, n_values)
                print(f"    {len(results)} Regler in {time.perf_counter() - t0:.1f} s")
                all_results.extend(results)

                print(f"  [{label}] Kurven ...")
                results = _bench_curves(base, sid, ref_jpg, max_side, stage, n_values)
                all_results.extend(results)

                if lut_path:
                    print(f"  [{label}] LUT-Gewicht ...")
                    results = _bench_lut_weight(base, sid, ref_jpg, max_side, stage,
                                                n_values, lut_path)
                    all_results.extend(results)

                print(f"  [{label}] Master ...")
                results = _bench_master(base, sid, ref_jpg, max_side, stage, n_values)
                all_results.extend(results)

                print(f"  [{label}] Bereiche ON/OFF ...")
                results = _bench_sections_onoff(base, sid, ref_jpg, max_side, stage, n_values)
                all_results.extend(results)

    except Exception as e:
        print(f"  FEHLER: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Worker sicher beenden
        try:
            _post(f"{base}/shutdown", {})
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)

    if not all_results:
        print("  Keine Ergebnisse.")
        return 1

    # ── Bericht schreiben ──────────────────────────────────────────────────
    ts = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n## Zustand `{gpu_state}` — {ts} — VRAM: {gpu_vram}",
        "",
        f"GPU: {gpu_info}",
        f"Modus: {'quick' if quick else 'voll'} (max_sides={max_sides}, n={n_values})",
        "",
        "**Alle Zahlen gemessen unter LLM-GPU-Last (BeeLlama belegt VRAM).**",
        "",
        "### Tabelle je Regler",
        "",
        "| Regler | Bereich | Median ms | P90 ms | Größter Anteil |",
        "|---|---|---|---|---|",
    ]

    for r in all_results:
        top_key = max(r["timing"].items(), key=lambda x: x[1])[0] if r["timing"] else "–"
        top_val = max(r["timing"].values()) if r["timing"] else 0.0
        lines.append(
            f"| {r['name']} | {r['group']} | {r['median']:.0f} | {r['p90']:.0f} "
            f"| {top_key} ({top_val:.0f} ms) |"
        )

    # Rangliste der teuersten Teilschritte
    lines += ["", "### Rangliste der teuersten Teilschritte (Median über alle Regler)", ""]
    step_totals: dict[str, list[float]] = {}
    for r in all_results:
        for k, v in r["timing"].items():
            step_totals.setdefault(k, []).append(v)
    step_medians = {k: statistics.median(v) for k, v in step_totals.items()}
    sorted_steps = sorted(step_medians.items(), key=lambda x: -x[1])
    lines.append("| Rang | Teilschritt | Median ms |")
    lines.append("|---|---|---|")
    for i, (k, v) in enumerate(sorted_steps, 1):
        lines.append(f"| {i} | {k} | {v:.1f} |")

    report = "\n".join(lines) + "\n"

    # Anhängen (nicht überschreiben)
    out_path = REPORTS_DIR / "latency.md"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Bericht: {out_path}")

    # Grafik
    chart_path = REPORTS_DIR / "latency.png"
    _make_chart(all_results, chart_path)

    print("  Fertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
