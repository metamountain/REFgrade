"""PAIR-Bench: Messung von Fit-Strategien für LUTs aus Vorher/Nachher-Paaren.

Aufruf:
    python tools/pair_bench.py              # voller Lauf (≤ 20 min)
    python tools/pair_bench.py --quick      # Schnelltest (1 Fitbild, 3 Gradings, 2 Strategien)
    python tools/pair_bench.py --phase1     # nur Kernraster
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from refgrade.color import rgb_to_lab, luma
from refgrade.lut import apply as lut_apply
from pair_grades import GRADES
from pair_strategies import STRATEGIES

# --- Konstanten ---

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POOL = os.path.join(ROOT, "bench", "pool")
LUTS = os.path.join(ROOT, "luts")
REPORTS = os.path.join(ROOT, "reports")

FIT_IMAGES = ["01.jpg", "03.jpg", "05.jpg", "07.jpg", "09.jpg"]
TEST_IMAGES = ["02.jpg", "04.jpg", "06.jpg", "08.jpg", "10.jpg"]
REAL_LUTS = ["OrangeAndBlue.cube", "Candlelight.CUBE", "ColdChrome.cube"]
PERTURBATIONS = ["u16", "u8", "jpeg90", "jpeg75", "noise"]
CORE_PERTURBATIONS = ["u16", "jpeg90"]

SKIN_TONES = np.array([
    [0.278, 0.204, 0.165], [0.341, 0.253, 0.204], [0.384, 0.290, 0.235],
    [0.447, 0.333, 0.271], [0.482, 0.361, 0.298], [0.525, 0.392, 0.325],
    [0.573, 0.424, 0.353], [0.616, 0.455, 0.380], [0.667, 0.490, 0.412],
    [0.718, 0.525, 0.443], [0.765, 0.557, 0.471], [0.812, 0.592, 0.502],
    [0.855, 0.624, 0.530], [0.902, 0.659, 0.561],
], dtype=np.float32)

SEED = 42


# --- Helper ---

def load_image(path: str, max_side: int = 768) -> torch.Tensor:
    """Bild laden, auf max_side skalieren, als [H,W,3] float32 0..1."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def perturb(before: torch.Tensor, after: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Störung auf beide Bilder anwenden."""
    if kind == "u16":
        return (torch.round(before * 65535.0) / 65535.0,
                torch.round(after * 65535.0) / 65535.0)
    elif kind == "u8":
        return (torch.round(before * 255.0) / 255.0,
                torch.round(after * 255.0) / 255.0)
    elif kind in ("jpeg90", "jpeg75"):
        q = 90 if kind == "jpeg90" else 75
        return _jpeg_roundtrip(before, q), _jpeg_roundtrip(after, q)
    elif kind == "noise":
        gen = torch.Generator().manual_seed(SEED)
        nb = before + torch.randn(before.shape, generator=gen) * 0.01
        na = after + torch.randn(after.shape, generator=gen) * 0.01
        return (torch.round(nb.clamp(0, 1) * 65535.0) / 65535.0,
                torch.round(na.clamp(0, 1) * 65535.0) / 65535.0)
    raise ValueError(f"Unbekannte Störung: {kind}")


def _jpeg_roundtrip(img: torch.Tensor, quality: int) -> torch.Tensor:
    import io
    arr = (img.numpy().clip(0, 1) * 255).round().astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return torch.from_numpy(np.asarray(Image.open(buf), dtype=np.float32) / 255.0)


def make_lut_dict(table_rgb: torch.Tensor, N: int) -> dict:
    """Tensor [r][g][b] → lut.build-Format [b][g][r]."""
    table_bgr = table_rgb.permute(2, 1, 0, 3).float()
    return {"size": N, "table": table_bgr, "title": "pair_bench"}


def apply_lut_2d(img: torch.Tensor, lut_dict: dict) -> torch.Tensor:
    """LUT anwenden für [H,W,3] Input (nutzt refgrade.lut.apply)."""
    return lut_apply(img.unsqueeze(0), lut_dict).squeeze(0)


def apply_lut_flat(pts: torch.Tensor, lut_dict: dict) -> torch.Tensor:
    """LUT anwenden für [P,3] Input (eigene trilineare Interpolation)."""
    N = lut_dict["size"]
    table = lut_dict["table"]  # [b][g][r] → permutiere zu [r][g][b]
    table_rgb = table.permute(2, 1, 0, 3)  # [r][g][b]
    g = pts * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    f = g - i0
    out = torch.zeros(pts.shape[0], 3, dtype=table_rgb.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) *
                     (f[:, 1] if dy else 1 - f[:, 1]) *
                     (f[:, 2] if dz else 1 - f[:, 2]))[:, None]
                out += w * table_rgb[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def _to_4d(t: torch.Tensor) -> torch.Tensor:
    """Stellt sicher, dass t 4D [B,H,W,3] ist für rgb_to_lab."""
    if t.dim() == 2:  # [P,3] → [1,1,P,3]
        return t.unsqueeze(0).unsqueeze(1)
    elif t.dim() == 3:  # [H,W,3] → [1,H,W,3]
        return t.unsqueeze(0)
    return t


def delta_e76(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """ΔE76 Mittel und p95. a, b: [...,3] sRGB 0..1."""
    lab_a = rgb_to_lab(_to_4d(a))
    lab_b = rgb_to_lab(_to_4d(b))
    d = (lab_a - lab_b).norm(dim=-1).flatten()
    return d.mean().item(), d.quantile(0.95).item()


def make_gray_ramp() -> torch.Tensor:
    """256-Stufen-Graukeil als [256,3]."""
    v = torch.linspace(0, 1, 256).view(256, 1, 1)
    return v.expand(256, 1, 3).reshape(256, 3).contiguous()


def make_rgb_grid(N: int = 17) -> torch.Tensor:
    """N³-RGB-Gitter als [N³, 3]."""
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N)] * 3, indexing="ij"), -1
    )
    return grid.reshape(-1, 3)


def banding_metric(lut_dict: dict) -> tuple[int, float]:
    """Banding auf 256er-Graukeil: Vorzeichenwechsel + größter L*-Sprung."""
    ramp = make_gray_ramp()  # [256,3]
    out = apply_lut_flat(ramp, lut_dict)  # [256,3]
    out_16 = torch.round(out * 65535.0) / 65535.0
    Y = luma(out_16).squeeze(-1)  # [256]
    dY = Y[1:] - Y[:-1]
    signs = torch.sign(dY)
    sign_changes = (signs[1:] * signs[:-1] < 0).sum().item()
    lab = rgb_to_lab(_to_4d(out_16))
    L = lab[..., 0].flatten()  # [256]
    dL = (L[1:] - L[:-1]).abs()
    return int(sign_changes), float(dL.max().item())


def get_all_gradings() -> dict[str, callable]:
    """Alle Gradings: synthetisch + echte LUTs."""
    grades = dict(GRADES)
    for i, name in enumerate(REAL_LUTS):
        path = os.path.join(LUTS, name)
        grades[f"real_{i}_{name}"] = (lambda x, p=path: _real_lut_grade(x, p))
    return grades


def _real_lut_grade(x: torch.Tensor, path: str) -> torch.Tensor:
    from refgrade.lut import read_cube
    lut = read_cube(path)
    shape = x.shape
    if x.dim() == 2:
        out = apply_lut_flat(x, lut)
    else:
        x4 = x.unsqueeze(0) if x.dim() == 3 else x
        out = lut_apply(x4, lut).squeeze(0) if x.dim() == 3 else lut_apply(x, lut)
    return out.reshape(shape).clamp(0, 1)


# --- Messung ---

def measure_one(
    strategy_fn: callable,
    grade_fn: callable,
    fit_img: torch.Tensor,
    test_img: torch.Tensor,
    perturbation: str,
    N: int = 33,
) -> dict:
    """Eine Messung: Fit + alle Metriken."""
    before = fit_img
    after = grade_fn(fit_img)

    before_p, after_p = perturb(before, after, perturbation)

    b_flat = before_p.reshape(-1, 3)
    a_flat = after_p.reshape(-1, 3)

    t0 = time.perf_counter()
    lut_table = strategy_fn(b_flat, a_flat, N)
    fit_ms = (time.perf_counter() - t0) * 1000.0

    lut_dict = make_lut_dict(lut_table, N)

    # dE_fit auf A
    out_a = apply_lut_2d(before_p, lut_dict)
    dE_fit_mean, dE_fit_p95 = delta_e76(out_a, after_p)

    # dE_other auf B
    after_b = grade_fn(test_img)
    out_b = apply_lut_2d(test_img, lut_dict)
    dE_other_mean, dE_other_p95 = delta_e76(out_b, after_b)

    # dE_cube auf 17³-Gitter
    grid = make_rgb_grid(17)
    grid_target = grade_fn(grid)
    grid_out = apply_lut_flat(grid, lut_dict)
    dE_cube_mean, dE_cube_p95 = delta_e76(grid_out, grid_target)

    # dE_skin
    skin = torch.from_numpy(SKIN_TONES)
    skin_target = grade_fn(skin)
    skin_out = apply_lut_flat(skin, lut_dict)
    dE_skin_mean, _ = delta_e76(skin_out, skin_target)

    # Banding
    banding_changes, banding_jump = banding_metric(lut_dict)

    return {
        "dE_fit_mean": round(dE_fit_mean, 3),
        "dE_fit_p95": round(dE_fit_p95, 3),
        "dE_other_mean": round(dE_other_mean, 3),
        "dE_other_p95": round(dE_other_p95, 3),
        "dE_cube_mean": round(dE_cube_mean, 3),
        "dE_cube_p95": round(dE_cube_p95, 3),
        "dE_skin_mean": round(dE_skin_mean, 3),
        "banding_changes": banding_changes,
        "banding_max_jump": round(banding_jump, 4),
        "fit_ms": round(fit_ms, 1),
    }


# --- Hauptlauf ---

def run_bench(quick: bool = False, phase1_only: bool = False) -> list[dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Lade Bilder...")
    fit_imgs = {n: load_image(os.path.join(POOL, n)) for n in (FIT_IMAGES[:1] if quick else FIT_IMAGES)}
    test_imgs = {n: load_image(os.path.join(POOL, n)) for n in (TEST_IMAGES[:1] if quick else TEST_IMAGES)}

    all_grades = get_all_gradings()
    grade_names = ["gamma_up", "s_curve", "split_tone"] if quick else list(all_grades.keys())
    strat_names = ["splat", "hybrid"] if quick else list(STRATEGIES.keys())

    results: list[dict] = []

    # Phase 1
    n_pert = len(CORE_PERTURBATIONS)
    n_total = len(strat_names) * len(grade_names) * n_pert * len(fit_imgs)
    print(f"\n=== Phase 1: {len(strat_names)}×{len(grade_names)}×{n_pert}×{len(fit_imgs)} = {n_total} Fits ===")

    t_start = time.perf_counter()
    n_done = 0
    for sname in strat_names:
        sfn = STRATEGIES[sname]
        for gname in grade_names:
            gfn = all_grades[gname]
            for pert in CORE_PERTURBATIONS:
                for fname, fimg in fit_imgs.items():
                    tname = list(test_imgs.keys())[0]
                    timg = test_imgs[tname]
                    try:
                        m = measure_one(sfn, gfn, fimg, timg, pert)
                    except Exception as e:
                        print(f"  ERR {sname}/{gname}/{pert}/{fname}: {e}")
                        m = {"error": str(e)}
                    results.append({"strategy": sname, "grading": gname, "perturbation": pert,
                                    "fit_image": fname, "phase": 1, **m})
                    n_done += 1
                    if n_done % 30 == 0 or n_done == n_total:
                        el = time.perf_counter() - t_start
                        print(f"  {n_done}/{n_total} ({el:.0f}s, ETA {el/n_done*(n_total-n_done):.0f}s)")

    # Phase 2
    if not phase1_only and not quick:
        strat_scores: dict[str, list[float]] = {}
        for rec in results:
            if "dE_other_mean" in rec:
                strat_scores.setdefault(rec["strategy"], []).append(rec["dE_other_mean"])
        ranked = sorted(strat_scores.items(), key=lambda kv: np.mean(kv[1]))
        top4 = [name for name, _ in ranked[:4]]
        print(f"\nTop 4: {top4}")
        for name, scores in ranked:
            print(f"  {name}: {np.mean(scores):.3f}")

        n_total2 = len(top4) * len(grade_names) * len(PERTURBATIONS) * len(fit_imgs)
        print(f"\n=== Phase 2: {len(top4)}×{len(grade_names)}×{len(PERTURBATIONS)}×{len(fit_imgs)} = {n_total2} Fits ===")
        t_start2 = time.perf_counter()
        n_done2 = 0

        for sname in top4:
            sfn = STRATEGIES[sname]
            for gname in grade_names:
                gfn = all_grades[gname]
                for pert in PERTURBATIONS:
                    for fname, fimg in fit_imgs.items():
                        tidx = list(fit_imgs.keys()).index(fname) % len(test_imgs)
                        tname = list(test_imgs.keys())[tidx]
                        timg = test_imgs[tname]
                        try:
                            m = measure_one(sfn, gfn, fimg, timg, pert)
                        except Exception as e:
                            print(f"  ERR {sname}/{gname}/{pert}/{fname}: {e}")
                            m = {"error": str(e)}
                        results.append({"strategy": sname, "grading": gname, "perturbation": pert,
                                        "fit_image": fname, "phase": 2, **m})
                        n_done2 += 1
                        if n_done2 % 30 == 0 or n_done2 == n_total2:
                            el = time.perf_counter() - t_start2
                            print(f"  {n_done2}/{n_total2} ({el:.0f}s, ETA {el/n_done2*(n_total2-n_done2):.0f}s)")

    print(f"\nGesamt: {len(results)} Messungen")
    return results


def write_jsonl(results: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"JSONL: {path} ({len(results)} Zeilen)")


def generate_report(results: list[dict], path: str) -> None:
    ok = [r for r in results if "dE_other_mean" in r]
    lines: list[str] = []
    lines.append("# PAIR-Bench: Rangliste der Fit-Strategien\n")
    lines.append(f"Lauf: {time.strftime('%Y-%m-%d %H:%M')}, {len(ok)} Messungen\n")

    # 1. Rangliste
    lines.append("## 1. Rangliste (dE_other Mittel)\n")
    lines.append("| Rang | Strategie | dE_other | p95 | dE_cube | dE_skin | banding | ms |")
    lines.append("|---|---|---|---|---|---|---|---|")

    strat_data: dict[str, dict] = {}
    for sname in STRATEGIES:
        recs = [r for r in ok if r["strategy"] == sname]
        if not recs:
            continue
        strat_data[sname] = {
            "dE_other": np.mean([r["dE_other_mean"] for r in recs]),
            "dE_other_p95": np.mean([r["dE_other_p95"] for r in recs]),
            "dE_cube": np.mean([r["dE_cube_mean"] for r in recs]),
            "dE_skin": np.mean([r["dE_skin_mean"] for r in recs]),
            "banding": np.mean([r["banding_changes"] for r in recs]),
            "ms": np.mean([r["fit_ms"] for r in recs]),
        }

    ranked = sorted(strat_data.items(), key=lambda kv: kv[1]["dE_other"])
    for i, (name, d) in enumerate(ranked, 1):
        lines.append(f"| {i} | {name} | {d['dE_other']:.3f} | {d['dE_other_p95']:.3f} | "
                     f"{d['dE_cube']:.3f} | {d['dE_skin']:.3f} | {d['banding']:.1f} | {d['ms']:.0f} |")

    # 2. Strategie × Grading
    lines.append("\n## 2. Strategie × Grading (dE_other Mittel)\n")
    grade_names = sorted(set(r["grading"] for r in ok))
    lines.append("| Strategie | " + " | ".join(g.replace(".", "")[:12] for g in grade_names) + " |")
    lines.append("|---|" + "---|" * len(grade_names))
    for sname, _ in ranked:
        row = f"| {sname} |"
        for gname in grade_names:
            recs = [r for r in ok if r["strategy"] == sname and r["grading"] == gname]
            row += f" {np.mean([r['dE_other_mean'] for r in recs]):.2f} |" if recs else " – |"
        lines.append(row)

    # 3. Strategie × Störung
    lines.append("\n## 3. Strategie × Störung (dE_other Mittel)\n")
    pert_names = ["u16", "u8", "jpeg90", "jpeg75", "noise"]
    lines.append("| Strategie | u16 | u8 | jpeg90 | jpeg75 | noise |")
    lines.append("|---|---|---|---|---|---|")
    for sname, _ in ranked:
        row = f"| {sname} |"
        for pert in pert_names:
            recs = [r for r in ok if r["strategy"] == sname and r["perturbation"] == pert]
            row += f" {np.mean([r['dE_other_mean'] for r in recs]):.2f} |" if recs else " – |"
        lines.append(row)

    # 16-Bit vs 8-Bit
    lines.append("\n### 16 Bit vs 8 Bit\n")
    lines.append("| Strategie | u16 | u8 | Δ |")
    lines.append("|---|---|---|---|")
    for sname, _ in ranked:
        r16 = [r for r in ok if r["strategy"] == sname and r["perturbation"] == "u16"]
        r8 = [r for r in ok if r["strategy"] == sname and r["perturbation"] == "u8"]
        if r16 and r8:
            m16 = np.mean([r["dE_other_mean"] for r in r16])
            m8 = np.mean([r["dE_other_mean"] for r in r8])
            lines.append(f"| {sname} | {m16:.3f} | {m8:.3f} | {m8-m16:+.3f} |")

    # 4. Empfehlung
    lines.append("\n## 4. Empfehlung\n")
    if ranked:
        best_name, best_d = ranked[0]
        lines.append(f"**Standard: `{best_name}`**")
        lines.append(f"\ndE_other={best_d['dE_other']:.3f}, dE_cube={best_d['dE_cube']:.3f}, ms={best_d['ms']:.0f}")
        if len(ranked) > 1:
            sn, sd = ranked[1]
            pct = (sd["dE_other"] - best_d["dE_other"]) / max(best_d["dE_other"], 0.01) * 100
            if pct < 3.0:
                lines.append(f"\n{sn} nur {pct:.1f}% schlechter ({sd['dE_other']:.3f}), {sd['ms']:.0f} ms.")

    # 5. Grenzen
    lines.append("\n## 5. Grenzen\n")
    lines.append("- Synthetische Gradings decken lokale Effekte und Rauschen im Original nicht ab.")
    lines.append("- JPEG ist immer 8-Bit; 16-Bit-JPEG nicht getestet.")
    lines.append("- Kein Alignment-Test (Paar ist per Definition perfekt aligned).")
    lines.append("- CPU-only; GPU-Effekte nicht getestet.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Bericht: {path}")


def main():
    parser = argparse.ArgumentParser(description="PAIR-Bench")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--phase1", action="store_true")
    args = parser.parse_args()

    os.makedirs(REPORTS, exist_ok=True)
    results = run_bench(quick=args.quick, phase1_only=args.phase1)

    suffix = "_quick" if args.quick else ""
    write_jsonl(results, os.path.join(REPORTS, f"pair_bench{suffix}.jsonl"))
    generate_report(results, os.path.join(REPORTS, f"pair_bench{suffix}.md"))


if __name__ == "__main__":
    main()
