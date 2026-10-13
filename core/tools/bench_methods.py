"""Bench: automatischer Methodenvergleich (T1-T5).

Aufruf:
    python.exe tools/bench_methods.py --all
    python.exe tools/bench_methods.py --method reinhard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import data as skdata
import colour

# sys.path für refgrade
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.methods import get, list_all, is_stable
from refgrade.methods import classic  # noqa: F401
from refgrade.metrics import coverage, swd, delta_e00, structure, banding, lut_fidelity
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from refgrade.color import rgb_to_lab, luma
from refgrade.lut import build, apply as lut_apply
from refgrade.auto_ignore import suggest
from refgrade.stats import sample

_POOL_DIR = Path(__file__).resolve().parents[2] / "bench" / "pool"
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "bench" / "results"
_HISTORY_FILE = Path(__file__).resolve().parents[1] / "bench" / "history.jsonl"
_LEADERBOARD = Path(__file__).resolve().parents[1] / "bench" / "LEADERBOARD.md"
_MAX_SIZE = 1024
_SEED = 0
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load_pool() -> list[tuple[str, torch.Tensor]]:
    """Pool laden: alle Bilder, ≤ 1024 px, Reihenfolge nach Dateiname."""
    if not os.path.isdir(_POOL_DIR):
        print(f"{_POOL_DIR} existiert nicht — Nutzer legt die Bilder ab")
        sys.exit(2)
    files = sorted(
        f for f in os.listdir(_POOL_DIR)
        if os.path.splitext(f)[1].lower() in _VALID_EXTS
    )
    if not files:
        print(f"{_POOL_DIR} ist leer — Nutzer legt die Bilder ab")
        sys.exit(2)

    dev = _get_device()
    pool = []
    for fname in files:
        fpath = os.path.join(_POOL_DIR, fname)
        img = Image.open(fpath).convert("RGB")
        w, h = img.size
        scale = min(1.0, _MAX_SIZE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).to(dev)
        pool.append((fpath, tensor))
    return pool


def _random_slider_values(seed: int) -> dict[str, float]:
    """Seed-feste Zufalls-Reglerwerte aus SLIDERS."""
    rng = np.random.default_rng(seed)
    values = {}
    for s in SLIDERS:
        if s.core:
            continue  # match-Regler nicht variieren
        lo, hi = s.min, s.max
        # Begrenzte Variation
        if s.name == "exposure":
            val = float(rng.uniform(-1.0, 1.0))
        elif s.name in ("temperature", "tint"):
            val = float(rng.uniform(-0.5, 0.5))
        elif s.name == "saturation":
            val = float(rng.uniform(0.5, 1.5))
        elif s.name == "contrast":
            val = float(rng.uniform(0.8, 1.3))
        elif s.name.startswith("sel_"):
            val = float(rng.uniform(-0.3, 0.3))
        elif s.name in ("shadow_hue", "highlight_hue"):
            val = float(rng.uniform(0, 360))
        elif s.name in ("shadow_amount", "highlight_amount"):
            val = float(rng.uniform(0, 0.4))
        else:
            val = float(rng.uniform(lo, hi))
        values[s.name] = round(val, 3)
    return values


def _crop_40(img: torch.Tensor, position: str) -> torch.Tensor:
    """40 % Ausschnitt: 'tl' = oben-links, 'br' = unten-rechts."""
    h, w = img.shape[1], img.shape[2]
    ch, cw = h * 40 // 100, w * 40 // 100
    if position == "tl":
        return img[:, :ch, :cw, :]
    else:  # br
        return img[:, h - ch:, w - cw:, :]


def _time_and_vram(fn):
    """Zeit (mit cuda.synchronize) und Spitzen-VRAM messen."""
    dev = _get_device()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = 0.0
    if dev.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return result, elapsed, peak_vram


def _make_tree_mask(h: int, w: int, device) -> torch.Tensor:
    """T4: Baum in der linken Hälfte."""
    mask = torch.zeros(1, h, w, device=device)
    mask[:, :, :w // 2] = 1.0
    return mask


def _run_t1(pool, method_name, seed):
    """T1: bekannte Zufalls-Graduierung auf X → Vorlage; ΔE00 gegen Wahrheit."""
    dev = _get_device()
    results = []
    for i, (path, img) in enumerate(pool):
        slider_vals = _random_slider_values(seed + i)
        truth = grade(img, slider_vals)
        ref = truth  # Vorlage = gegradetes Bild

        fit_fn = get(method_name)
        out, elapsed, vram = _time_and_vram(
            lambda: fit_fn(img, ref)(img)
        )
        de_mean, de_p95 = delta_e00(out, truth)
        results.append({
            "image": os.path.basename(path),
            "delta_e00_mean": round(de_mean, 4),
            "delta_e00_p95": round(de_p95, 4),
            "time_s": round(elapsed, 4),
            "peak_vram_mb": round(vram, 1),
        })
    return results


def _run_t1b(pool, method_name, seed):
    """T1b: wie T1, Vorlage nur Ausschnitt 40 %; Fehler auf ganzem X."""
    dev = _get_device()
    results = []
    for i, (path, img) in enumerate(pool):
        slider_vals = _random_slider_values(seed + i)
        truth = grade(img, slider_vals)

        # Vorlage: Ausschnitt
        ref_tl = _crop_40(truth, "tl")
        ref_br = _crop_40(truth, "br")

        fit_fn = get(method_name)
        out_tl, t1, v1 = _time_and_vram(
            lambda: fit_fn(img, ref_tl)(img)
        )
        de_tl_mean, _ = delta_e00(out_tl, truth)

        out_br, t2, v2 = _time_and_vram(
            lambda: fit_fn(img, ref_br)(img)
        )
        de_br_mean, _ = delta_e00(out_br, truth)

        results.append({
            "image": os.path.basename(path),
            "tl_delta_e00": round(de_tl_mean, 4),
            "br_delta_e00": round(de_br_mean, 4),
            "time_s": round((t1 + t2) / 2, 4),
            "peak_vram_mb": round(max(v1, v2), 1),
        })
    return results


def _run_t2(pool, method_name, seed):
    """T2: gleiche Graduierung auf anderes Bild Y; LUT aus T1 auf Y."""
    dev = _get_device()
    if len(pool) < 2:
        return []
    results = []
    for i, (path_x, img_x) in enumerate(pool[:-1]):
        path_y, img_y = pool[i + 1]
        slider_vals = _random_slider_values(seed + i)

        # T1: LUT aus X
        truth_x = grade(img_x, slider_vals)
        fit_fn = get(method_name)
        lut_fn = fit_fn(img_x, truth_x)
        lut = build(lut_fn, size=33, device=dev)

        # Auf Y anwenden
        truth_y = grade(img_y, slider_vals)
        out_y, elapsed, vram = _time_and_vram(
            lambda: lut_apply(img_y, lut)
        )
        de_mean, de_p95 = delta_e00(out_y, truth_y)
        results.append({
            "image_x": os.path.basename(path_x),
            "image_y": os.path.basename(path_y),
            "delta_e00_mean": round(de_mean, 4),
            "delta_e00_p95": round(de_p95, 4),
            "time_s": round(elapsed, 4),
            "peak_vram_mb": round(vram, 1),
        })
    return results


def _run_t3(pool, method_name, seed):
    """T3: Vorlage mit anderem Inhalt: coverage, swd, structure, banding."""
    dev = _get_device()
    if len(pool) < 2:
        return []
    results = []
    for i in range(0, len(pool) - 1, 2):
        path_x, img_x = pool[i]
        path_y, img_y = pool[i + 1]

        fit_fn = get(method_name)
        out, elapsed, vram = _time_and_vram(
            lambda: fit_fn(img_x, img_y)(img_x)
        )

        cov = coverage(img_x, img_y)
        swd_val = swd(img_x, out)
        struct = structure(img_x, out)
        band = banding(img_x, out)

        results.append({
            "image": os.path.basename(path_x),
            "reference": os.path.basename(path_y),
            "coverage": round(cov, 2),
            "swd": round(swd_val, 5),
            "structure": round(struct, 4),
            "banding": round(band, 4),
            "time_s": round(elapsed, 4),
            "peak_vram_mb": round(vram, 1),
        })
    return results


def _run_t4(pool, method_name, seed):
    """T4: dominante Fremdfläche (Baum) — ohne Maske, mit Maske, auto_ignore."""
    dev = _get_device()
    results = []
    for i, (path, img) in enumerate(pool):
        h, w = img.shape[1], img.shape[2]
        slider_vals = _random_slider_values(seed + i)

        # Baum erzeugen
        torch.manual_seed(0)
        tree = torch.zeros(1, h, w, 3, device=dev)
        tree[..., 1] = 0.55 + 0.25 * torch.rand(1, h, w, device=dev)
        tree[..., 0] = 0.10 + 0.10 * torch.rand(1, h, w, device=dev)
        tree[..., 2] = 0.05 + 0.08 * torch.rand(1, h, w, device=dev)
        tree_mask = _make_tree_mask(h, w, dev)

        # Bild mit Baum
        src = img.clone()
        src[tree_mask > 0.5] = tree[tree_mask > 0.5]

        # Wahrheit: Grade auf src
        truth = grade(src, slider_vals)

        # Vorlage: Grade auf img (ohne Baum)
        ref = grade(img, slider_vals)

        fit_fn = get(method_name)

        # A) Ohne Maske
        out_a, t_a, v_a = _time_and_vram(
            lambda: fit_fn(src, ref)(src)
        )
        de_a, _ = delta_e00(out_a, truth)

        # B) Mit wahrer Ignore-Maske (nur Statistik)
        keep = 1.0 - tree_mask  # 1 = gewichtet, 0 = ignoriert
        out_b, t_b, v_b = _time_and_vram(
            lambda: fit_fn(src, ref, image_weights=keep)(src)
        )
        de_b, _ = delta_e00(out_b, truth)

        # C) auto_ignore
        auto_mask = suggest(src, ref, chroma_percentile=99, min_area=0.02, feather_px=4)
        auto_weights = 1.0 - auto_mask
        out_c, t_c, v_c = _time_and_vram(
            lambda: fit_fn(src, ref, image_weights=auto_weights)(src)
        )
        de_c, _ = delta_e00(out_c, truth)

        results.append({
            "image": os.path.basename(path),
            "no_mask_delta_e00": round(de_a, 4),
            "true_mask_delta_e00": round(de_b, 4),
            "auto_ignore_delta_e00": round(de_c, 4),
            "time_s": round((t_a + t_b + t_c) / 3, 4),
            "peak_vram_mb": round(max(v_a, v_b, v_c), 1),
        })
    return results


def _run_t5(pool, seed):
    """T5: Ranking taugt? Spearman-Korrelation zwischen rank-Score und ΔE00."""
    dev = _get_device()
    from refgrade.reference_rank import rank as ref_rank, _load_image_tensor

    results = []
    for i, (path, img) in enumerate(pool[:3]):  # Nur 3 Bilder für T5
        slider_vals = _random_slider_values(seed + i)
        truth = grade(img, slider_vals)

        # Vorlagen: dieselbe Graduierung auf anderen Pool-Bildern
        refs = []
        for j, (path_j, img_j) in enumerate(pool):
            if j == i:
                continue
            ref_j = grade(img_j, slider_vals)
            refs.append((path_j, ref_j))

        # Ranking
        # Für das Ranking brauchen wir einen Ordner — wir nutzen den Pool direkt
        ranked = ref_rank(img, _POOL_DIR, k=len(refs))

        # Für jede Top-Vorlage: Reinhard/MKL, ΔE00
        scores = []
        de_values = []
        for rank_path, rank_score in ranked:
            # Finden wir die Vorlage in unseren refs?
            basename = os.path.basename(rank_path)
            found_ref = None
            for p, r in refs:
                if os.path.basename(p) == basename:
                    found_ref = r
                    break
            if found_ref is None:
                continue
            fn = get("reinhard")
            out = fn(img, found_ref)(img)
            de_mean, _ = delta_e00(out, truth)
            scores.append(rank_score)
            de_values.append(de_mean)

        if len(scores) >= 3:
            # Spearman-Korrelation
            from scipy.stats import spearmanr
            rho, pval = spearmanr(scores, de_values)
        else:
            rho, pval = 0.0, 1.0

        # Top-1 ΔE00 vs. Median
        top1_de = de_values[0] if de_values else 0.0
        median_de = float(np.median(de_values)) if de_values else 0.0

        results.append({
            "image": os.path.basename(path),
            "spearman_rho": round(rho, 4),
            "p_value": round(pval, 4),
            "top1_delta_e00": round(top1_de, 4),
            "median_delta_e00": round(median_de, 4),
            "n_references": len(de_values),
        })
    return results


def _check_regression(history: list[dict], current: dict) -> bool:
    """Regression: stable-Methode > 2 % schlechter als letzter Eintrag → True."""
    for entry in current.get("methods", []):
        name = entry["method"]
        if not is_stable(name):
            continue
        # Letzten Eintrag in der History finden
        prev = None
        for h in reversed(history):
            if h.get("method") == name:
                prev = h
                break
        if prev is None:
            continue
        # T1b ΔE00 vergleichen
        prev_val = prev.get("t1b_delta_e00_mean", 0)
        curr_val = entry.get("t1b_delta_e00_mean", 0)
        if prev_val > 0 and curr_val > prev_val * 1.02:
            print(f"  ⚠️  REGRESSION: {name} T1b ΔE00 {curr_val:.4f} > {prev_val:.4f} (+2%)")
            return True
    return False


def _update_leaderboard(results: dict):
    """LEADERBOARD.md schreiben."""
    os.makedirs(os.path.dirname(_LEADERBOARD), exist_ok=True)
    lines = ["# Leaderboard", "",
             f"Letzter Lauf: {results.get('timestamp', '?')}", ""]

    # T1b Tabelle
    lines.append("## T1b (Hauptkriterium) — ΔE00 (niedriger = besser)")
    lines.append("")
    lines.append("| Methode | ΔE00 tl | ΔE00 br | stable |")
    lines.append("|---------|---------|---------|--------|")
    for m in results.get("methods", []):
        name = m["method"]
        stable = "✓" if is_stable(name) else ""
        tl = m.get("t1b_delta_e00_tl_mean", "-")
        br = m.get("t1b_delta_e00_br_mean", "-")
        lines.append(f"| {name} | {tl} | {br} | {stable} |")

    lines.append("")
    lines.append("## T2 — LUT auf anderes Bild")
    lines.append("")
    lines.append("| Methode | ΔE00 mean |")
    lines.append("|---------|-----------|")
    for m in results.get("methods", []):
        name = m["method"]
        val = m.get("t2_delta_e00_mean", "-")
        lines.append(f"| {name} | {val} |")

    with open(_LEADERBOARD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Bench: Methodenvergleich")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Alle Methoden")
    group.add_argument("--method", type=str, help="Einzelne Methode")
    args = parser.parse_args()

    if args.all:
        methods = list_all()
    else:
        methods = [args.method]

    dev = _get_device()
    print(f"Gerät: {dev}")
    print(f"Methoden: {methods}")

    # Pool laden
    pool = _load_pool()
    print(f"Pool: {len(pool)} Bilder")

    seed = _SEED
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ergebnisse sammeln
    all_results = []
    for method_name in methods:
        print(f"\n=== {method_name} ===")
        entry = {"method": method_name, "stable": is_stable(method_name)}

        # T1
        print("  T1: bekannte Graduierung...")
        t1 = _run_t1(pool, method_name, seed)
        entry["t1"] = t1
        entry["t1_delta_e00_mean"] = round(
            float(np.mean([r["delta_e00_mean"] for r in t1])), 4
        ) if t1 else None

        # T1b
        print("  T1b: Ausschnitt-Vorlage...")
        t1b = _run_t1b(pool, method_name, seed)
        entry["t1b"] = t1b
        if t1b:
            entry["t1b_delta_e00_tl_mean"] = round(
                float(np.mean([r["tl_delta_e00"] for r in t1b])), 4
            )
            entry["t1b_delta_e00_br_mean"] = round(
                float(np.mean([r["br_delta_e00"] for r in t1b])), 4
            )
            entry["t1b_delta_e00_mean"] = round(
                (entry["t1b_delta_e00_tl_mean"] + entry["t1b_delta_e00_br_mean"]) / 2, 4
            )

        # T2
        print("  T2: LUT auf anderes Bild...")
        t2 = _run_t2(pool, method_name, seed)
        entry["t2"] = t2
        entry["t2_delta_e00_mean"] = round(
            float(np.mean([r["delta_e00_mean"] for r in t2])), 4
        ) if t2 else None

        # T3
        print("  T3: anderer Inhalt...")
        t3 = _run_t3(pool, method_name, seed)
        entry["t3"] = t3
        if t3:
            entry["t3_structure_mean"] = round(
                float(np.mean([r["structure"] for r in t3])), 4
            )

        # T4
        print("  T4: dominante Fremdfläche...")
        t4 = _run_t4(pool, method_name, seed)
        entry["t4"] = t4
        if t4:
            entry["t4_no_mask_mean"] = round(
                float(np.mean([r["no_mask_delta_e00"] for r in t4])), 4
            )
            entry["t4_true_mask_mean"] = round(
                float(np.mean([r["true_mask_delta_e00"] for r in t4])), 4
            )
            entry["t4_auto_ignore_mean"] = round(
                float(np.mean([r["auto_ignore_delta_e00"] for r in t4])), 4
            )

        all_results.append(entry)
        print(f"  T1 ΔE00: {entry.get('t1_delta_e00_mean')}")
        print(f"  T1b ΔE00: {entry.get('t1b_delta_e00_mean')}")
        print(f"  T2 ΔE00: {entry.get('t2_delta_e00_mean')}")

    # T5 (einmalig, nicht pro Methode)
    print("\n=== T5: Ranking taugt? ===")
    t5 = _run_t5(pool, seed)
    for r in t5:
        print(f"  {r['image']}: ρ={r['spearman_rho']}, "
              f"Top1 ΔE00={r['top1_delta_e00']}, Median={r['median_delta_e00']}")

    # Zusammenfassung
    results = {
        "timestamp": timestamp,
        "device": str(dev),
        "pool_size": len(pool),
        "methods": all_results,
        "t5": t5,
    }

    # Regression prüfen
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    history = []
    if os.path.exists(_HISTORY_FILE):
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    regression = _check_regression(history, results)

    # Ergebnis speichern
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(_RESULTS_DIR, f"{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nErgebnis: {result_path}")

    # History
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        for entry in all_results:
            summary = {
                "timestamp": timestamp,
                "method": entry["method"],
                "stable": entry["stable"],
                "t1_delta_e00_mean": entry.get("t1_delta_e00_mean"),
                "t1b_delta_e00_mean": entry.get("t1b_delta_e00_mean"),
                "t2_delta_e00_mean": entry.get("t2_delta_e00_mean"),
                "t3_structure_mean": entry.get("t3_structure_mean"),
                "t4_no_mask_mean": entry.get("t4_no_mask_mean"),
                "t4_true_mask_mean": entry.get("t4_true_mask_mean"),
                "t4_auto_ignore_mean": entry.get("t4_auto_ignore_mean"),
            }
            f.write(json.dumps(summary) + "\n")

    # Leaderboard
    _update_leaderboard(results)
    print(f"Leaderboard: {_LEADERBOARD}")

    if regression:
        print("\n⚠️  REGRESSION erkannt — Exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
