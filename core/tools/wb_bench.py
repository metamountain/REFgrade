"""WB Messreihe: Winkelfehler aller Schätzer × Lichtfarben, Korrekturweg, Laufzeit.

Aufruf:
    python tools/wb_bench.py            # volle Messreihe
    python tools/wb_bench.py --quick    # 2 Bilder, 3 Lichtfarben, 3 Schätzer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.color import srgb_to_linear, linear_to_srgb, rgb_to_lab, lab_to_rgb
from refgrade.grade import _temp_tint
from tools.wb_methods import ESTIMATORS, _d65_illuminant, _blackbody_illuminant, _lab_to_illuminant

_POOL = Path(__file__).resolve().parents[2] / "bench" / "pool"
_REPORTS = Path(__file__).resolve().parents[2] / "reports"
_SEED = 0
_MAX_SIZE = 768  # downscale for speed


# ---------------------------------------------------------------------------
# Illuminants
# ---------------------------------------------------------------------------

def _make_illuminants() -> dict[str, np.ndarray]:
    """12 illuminants: 7 Planck + 5 off-curve, all linear RGB sum=1."""
    ills: dict[str, np.ndarray] = {}
    for T in (2500, 3200, 4000, 5000, 6500, 8000, 10000):
        ills[f"{T}K"] = _blackbody_illuminant(T)
    # Off-curve: defined by Lab chromaticity
    off = {
        "green+": (50, -15, 0),
        "green-": (50, 15, 0),
        "magenta+": (50, 15, -10),
        "magenta-": (50, -15, 10),
        "cyan": (50, -10, -15),
    }
    for name, (L, a, b) in off.items():
        ills[name] = _lab_to_illuminant(L, a, b)
    return ills


# ---------------------------------------------------------------------------
# Image loading and cast application
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """Load image, downscale to ≤ _MAX_SIZE, return float32 [H,W,3] in 0..1."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, _MAX_SIZE / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _apply_cast(rgb: np.ndarray, ill: np.ndarray, d65: np.ndarray) -> np.ndarray:
    """Apply illuminant: sRGB→linear, multiply by ill/d65, →sRGB, 16-bit quantize."""
    linear = _srgb_to_linear_np(rgb)
    ratio = ill / d65
    cast = linear * ratio
    srgb = _linear_to_srgb_np(cast)
    # 16-bit quantize
    q = np.round(srgb * 65535.0) / 65535.0
    return q.astype(np.float32)


def _jpeg_variant(rgb: np.ndarray) -> np.ndarray:
    """Encode to JPEG quality 90 and decode back."""
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def _srgb_to_linear_np(t: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → linear (numpy)."""
    return np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb_np(t: np.ndarray) -> np.ndarray:
    """linear → sRGB [0,1] (numpy)."""
    t = np.clip(t, 0, None)
    return np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.clip(t, 0.0031308, None) ** (1/2.4) - 0.055)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def angular_error_deg(est: np.ndarray, true: np.ndarray) -> float:
    """Angle in degrees between two illuminant vectors."""
    cos = np.dot(est, true) / (np.linalg.norm(est) * np.linalg.norm(true) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def delta_e76(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Mean ΔE76 in CIELAB between two sRGB images [H,W,3]."""
    ta = torch.from_numpy(img_a).float().unsqueeze(0)  # [1,H,W,3]
    tb = torch.from_numpy(img_b).float().unsqueeze(0)
    lab_a = rgb_to_lab(ta).squeeze(0).numpy()
    lab_b = rgb_to_lab(tb).squeeze(0).numpy()
    return float(np.sqrt(((lab_a - lab_b) ** 2).sum(axis=-1)).mean())


def delta_e76_p95(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """95th percentile ΔE76 between two sRGB images."""
    ta = torch.from_numpy(img_a).float().unsqueeze(0)
    tb = torch.from_numpy(img_b).float().unsqueeze(0)
    lab_a = rgb_to_lab(ta).squeeze(0).numpy()
    lab_b = rgb_to_lab(tb).squeeze(0).numpy()
    de = np.sqrt(((lab_a - lab_b) ** 2).sum(axis=-1))
    return float(np.percentile(de, 95))


# ---------------------------------------------------------------------------
# Correction paths
# ---------------------------------------------------------------------------

def correct_von_kries(cast: np.ndarray, est_ill: np.ndarray, d65: np.ndarray) -> np.ndarray:
    """Von Kries: divide per-channel in linear RGB by est_ill, then re-apply D65."""
    linear = _srgb_to_linear_np(cast)
    corrected = linear / (est_ill + 1e-9)
    # Normalize to match D65 luminance
    ratio = d65 / est_ill
    corrected = linear * ratio
    return _linear_to_srgb_np(corrected)


def correct_our_path(cast: np.ndarray, est_ill: np.ndarray, d65: np.ndarray) -> np.ndarray:
    """Our path: est_ill → temp/tint (exakte von-Kries-Umkehrung, 2026-10-13) → _temp_tint."""
    from refgrade.whitebalance import _to_values
    v = _to_values(torch.from_numpy(np.asarray(est_ill, dtype=np.float64)))
    temp, tint = v["temperature"], v["tint"]
    t = torch.from_numpy(cast).float().unsqueeze(0)
    corrected = _temp_tint(t, temp, tint).squeeze(0).numpy()
    return corrected


# ---------------------------------------------------------------------------
# Main bench
# ---------------------------------------------------------------------------

def run_bench(quick: bool = False) -> list[dict]:
    """Run the full (or quick) measurement and return results as list of dicts."""
    np.random.seed(_SEED)
    d65 = _d65_illuminant()
    ills = _make_illuminants()

    if quick:
        ills = dict(list(ills.items())[:3])
        est_names = ["ours", "grayworld_0", "greyedge"]
    else:
        est_names = list(ESTIMATORS.keys())

    # Load images
    files = sorted(_POOL.glob("*.jpg"))
    if quick:
        files = files[:2]
    print(f"Images: {len(files)}, Illuminants: {len(ills)}, Estimators: {len(est_names)}")

    results: list[dict] = []
    # Runtime tracking
    runtimes: dict[str, list[float]] = {n: [] for n in est_names}

    for fpath in files:
        orig = _load_image(fpath)
        jpeg = _jpeg_variant(orig)
        variants = [("orig", orig), ("jpeg", jpeg)]
        for vname, base in variants:
            for ill_name, ill in ills.items():
                cast = _apply_cast(base, ill, d65)
                row: dict = {
                    "image": fpath.name, "variant": vname,
                    "illuminant": ill_name,
                }
                for est_name in est_names:
                    t0 = time.perf_counter()
                    est = ESTIMATORS[est_name](cast[np.newaxis, ...])
                    dt_ms = (time.perf_counter() - t0) * 1000
                    runtimes[est_name].append(dt_ms)
                    err = angular_error_deg(est, ill)
                    row[est_name] = err
                results.append(row)

    # Runtime summary
    runtime_summary = {
        n: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
        for n, v in runtimes.items()
    }

    # Angular error summary per estimator
    err_summary: dict[str, dict[str, float]] = {}
    for est_name in est_names:
        errs = np.array([r[est_name] for r in results])
        err_summary[est_name] = {
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "worst25_mean": float(np.mean(np.sort(errs)[-max(1, len(errs) // 4):])),
            "max": float(errs.max()),
        }

    # Correction path for best 3 (by median angular error)
    best3 = sorted(est_names, key=lambda n: err_summary[n]["median"])[:3]
    corr_results: dict[str, dict[str, float]] = {}
    for est_name in best3:
        de_a_list, de_b_list, de_a95_list, de_b95_list = [], [], [], []
        for fpath in files:
            orig = _load_image(fpath)
            for ill_name, ill in list(ills.items())[:2]:  # first 2 illuminants only
                cast = _apply_cast(orig, ill, d65)
                est = ESTIMATORS[est_name](cast[np.newaxis, ...])
                corr_a = correct_von_kries(cast, est, d65)
                corr_b = correct_our_path(cast, est, d65)
                de_a_list.append(delta_e76(corr_a, orig))
                de_b_list.append(delta_e76(corr_b, orig))
                de_a95_list.append(delta_e76_p95(corr_a, orig))
                de_b95_list.append(delta_e76_p95(corr_b, orig))
        corr_results[est_name] = {
            "von_kries_mean": float(np.mean(de_a_list)),
            "von_kries_p95": float(np.mean(de_a95_list)),
            "our_path_mean": float(np.mean(de_b_list)),
            "our_path_p95": float(np.mean(de_b95_list)),
        }

    return results, err_summary, runtime_summary, corr_results, best3, est_names


def write_outputs(results: list[dict], err_summary: dict, runtime_summary: dict,
                  corr_results: dict, best3: list, est_names: list) -> None:
    """Write reports/wb_bench.jsonl and reports/wb_bench.md."""
    _REPORTS.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = _REPORTS / "wb_bench.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_meta": "runtime", **runtime_summary}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_meta": "error_summary", **err_summary}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_meta": "correction", **corr_results}, ensure_ascii=False) + "\n")

    # Markdown report
    _write_markdown(err_summary, runtime_summary, corr_results, best3, est_names, results)


def _write_markdown(err_summary: dict, runtime_summary: dict, corr_results: dict,
                    best3: list, est_names: list, results: list) -> None:
    """Write the markdown report."""
    lines: list[str] = []
    lines.append("# WB Messreihe (K23)\n")
    lines.append(f"Bilder: {len(set(r['image'] for r in results))} aus `bench/pool/` "
                 f"(10 verfügbar, Spec nennt 12)\n")
    lines.append(f"Varianten: original + JPEG-90\n")
    lines.append(f"Schätzer: {len(est_names)}\n")

    # 1. Ranking
    lines.append("## 1. Rangliste nach Median-Winkelfehler\n")
    lines.append("| Rang | Schätzer | Median (°) | Mittel (°) | Worst-25 % (°) | Max (°) |")
    lines.append("|---|---|---|---|---|---|")
    ranked = sorted(est_names, key=lambda n: err_summary[n]["median"])
    for i, name in enumerate(ranked, 1):
        e = err_summary[name]
        lines.append(f"| {i} | {name} | {e['median']:.2f} | {e['mean']:.2f} | {e['worst25_mean']:.2f} | {e['max']:.2f} |")
    lines.append("")

    # 2. Table estimator × illuminant
    lines.append("## 2. Winkelfehler (°) Schätzer × Lichtfarbe\n")
    ill_names = list(dict.fromkeys(r["illuminant"] for r in results))
    header = "| Schätzer | " + " | ".join(ill_names) + " |"
    sep = "|---|" + "---|" * len(ill_names)
    lines.append(header)
    lines.append(sep)
    for name in est_names:
        row = [name]
        for ill in ill_names:
            errs = [r[name] for r in results if r["illuminant"] == ill]
            row.append(f"{np.mean(errs):.2f}" if errs else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 3. Correction path
    lines.append("## 3. Korrekturweg (a) von Kries vs (b) unser Weg\n")
    lines.append("ΔE76 zum unverfälschten Original (Mittel / p95):\n")
    lines.append("| Schätzer | (a) von Kries Mittel | (a) p95 | (b) unser Weg Mittel | (b) p95 |")
    lines.append("|---|---|---|---|---|")
    for name in best3:
        c = corr_results[name]
        lines.append(f"| {name} | {c['von_kries_mean']:.3f} | {c['von_kries_p95']:.3f} "
                     f"| {c['our_path_mean']:.3f} | {c['our_path_p95']:.3f} |")
    lines.append("")

    # 4. Runtime
    lines.append("## 4. Laufzeit (CPU, ms)\n")
    lines.append("| Schätzer | Mittel | p95 |")
    lines.append("|---|---|---|")
    for name in est_names:
        rt = runtime_summary[name]
        lines.append(f"| {name} | {rt['mean_ms']:.1f} | {rt['p95_ms']:.1f} |")
    lines.append("")

    # 5. Recommendation
    lines.append("## 5. Empfehlung\n")
    best = best3[0]
    lines.append(f"**Empfohlener Schätzer:** `{best}` (Median {err_summary[best]['median']:.2f}°)")
    # Determine best correction path
    for name in best3:
        c = corr_results[name]
        if c["von_kries_mean"] < c["our_path_mean"]:
            lines.append(f"- Korrekturweg: **von Kries** ist besser "
                         f"({c['von_kries_mean']:.3f} vs {c['our_path_mean']:.3f} ΔE76)")
        else:
            lines.append(f"- Korrekturweg: **unser Weg** (temp/tint) ist besser "
                         f"({c['our_path_mean']:.3f} vs {c['von_kries_mean']:.3f} ΔE76)")
        break
    lines.append("- Integration: in `whitebalance.py` als neue `estimate_illuminant()`; "
                 "Anbindung an Temperature/Tint-Regler in späterem Paket.")
    lines.append("")

    report_path = _REPORTS / "wb_bench.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"JSONL:  {_REPORTS / 'wb_bench.jsonl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WB Messreihe")
    parser.add_argument("--quick", action="store_true", help="2 Bilder, 3 Lichtfarben, 3 Schätzer")
    args = parser.parse_args()

    t0 = time.perf_counter()
    out = run_bench(quick=args.quick)
    results, err_summary, runtime_summary, corr_results, best3, est_names = out
    write_outputs(results, err_summary, runtime_summary, corr_results, best3, est_names)
    dt = time.perf_counter() - t0
    print(f"Fertig in {dt:.1f} s")


if __name__ == "__main__":
    main()
