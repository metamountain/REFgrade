"""K29b: Auto-WB „zu warm?" prüfen — Beleuchtungs-Invarianz in b* + Cast-freie Pool-Bilder.

Aufruf:
    python tools/autowb_check.py

Teil 3 führt `run_bench` bewusst neu aus (Quelle der Winkelfehler-Zahlen).
Private Importe aus `wb_bench` bleiben (kein Umbau nötig).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.color import rgb_to_lab
from refgrade.grade import _temp_tint
from refgrade.whitebalance import L_RANGE, NEUTRAL_CHROMA, estimate as wb_estimate
from tools.wb_bench import (
    _apply_cast,
    _load_image,
    _make_illuminants,
    run_bench,
)
from tools.wb_methods import _d65_illuminant

_POOL = Path(__file__).resolve().parents[2] / "bench" / "pool"
_REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _b_error_after_correction(
    orig: np.ndarray,
    cast: np.ndarray,
    orig_vals: dict[str, float] | None = None,
    orig_clamped: bool = False,
) -> tuple[float, bool, bool]:
    """Beleuchtungs-Invarianz-Fehler in b*.

    Soll = Auto-WB-Ergebnis des unverfälschten Bildes.
    Positiv = Ergebnis wärmer als das Auto-WB des Originals.
    clamped_cast  = True, wenn der Cast-WB das ±1-Limit erreicht.
    clamped_basis = True, wenn der Original-WB das ±1-Limit erreicht.
    """
    orig_t = torch.from_numpy(orig).float().unsqueeze(0)
    cast_t = torch.from_numpy(cast).float().unsqueeze(0)

    if orig_vals is None:
        orig_vals = wb_estimate(orig_t)
        orig_clamped = abs(orig_vals["temperature"]) >= 0.999 or abs(orig_vals["tint"]) >= 0.999
    ref = _temp_tint(orig_t, orig_vals["temperature"], orig_vals["tint"])

    vals = wb_estimate(cast_t)
    clamped_cast = abs(vals["temperature"]) >= 0.999 or abs(vals["tint"]) >= 0.999
    corr = _temp_tint(cast_t, vals["temperature"], vals["tint"])

    lab_corr = rgb_to_lab(corr)
    lab_ref = rgb_to_lab(ref)
    err = float((lab_corr[..., 2] - lab_ref[..., 2]).mean())
    return err, clamped_cast, orig_clamped


def _neutral_mask_stats(rgb: np.ndarray) -> dict:
    """Neutral-Masken-Statistik für ein Cast-freies Bild (gleiche Konstanten wie estimate)."""
    t = torch.from_numpy(rgb).float().unsqueeze(0)
    lab = rgb_to_lab(t)  # [1, H, W, 3]
    L, a, b = lab[0, :, :, 0], lab[0, :, :, 1], lab[0, :, :, 2]

    keep = (L > L_RANGE[0]) & (L < L_RANGE[1])
    neutral = keep & (torch.sqrt(a * a + b * b) < NEUTRAL_CHROMA)

    n_neutral = int(neutral.sum().item())
    n_keep = int(keep.sum().item())

    blue_share = float((b[neutral] < 0).sum().item()) / max(n_neutral, 1)
    median_b_neutral = float(b[neutral].median().item()) if n_neutral > 0 else 0.0
    median_b_keep = float(b[keep].median().item()) if n_keep > 0 else 0.0

    return {
        "blue_share": blue_share,
        "median_b_neutral": median_b_neutral,
        "median_b_keep": median_b_keep,
    }


def main() -> None:
    t0 = time.perf_counter()
    _REPORTS.mkdir(parents=True, exist_ok=True)
    d65 = _d65_illuminant()
    ills = _make_illuminants()

    # --- Part 1: Pool images without cast ---
    print("=== Part 1: Auto-WB on cast-free pool images ===")
    files = sorted(_POOL.glob("*.jpg"))
    print(f"Images: {len(files)}")

    pool_results: list[dict] = []
    for fpath in files:
        rgb = _load_image(fpath)
        t = torch.from_numpy(rgb).float().unsqueeze(0)
        vals = wb_estimate(t)
        stats = _neutral_mask_stats(rgb)
        pool_results.append({"image": fpath.name, **vals, **stats})
        print(
            f"  {fpath.name}: temp={vals['temperature']:+.3f}  tint={vals['tint']:+.3f}  "
            f"blue={stats['blue_share']:.2f}  med_b_neut={stats['median_b_neutral']:+.1f}"
        )

    # --- Part 2: Invariance error across all images × illuminants ---
    print("\n=== Part 2: Invariance b* error (all images × illuminants) ===")
    b_err_signed: list[float] = []
    b_err_abs: list[float] = []
    clamped_any: list[bool] = []
    clamped_cast_count = 0
    clamped_basis_count = 0
    per_ill_signed: dict[str, list[float]] = {name: [] for name in ills}
    per_ill_abs: dict[str, list[float]] = {name: [] for name in ills}
    per_ill_clamped_cast: dict[str, int] = {name: 0 for name in ills}
    per_ill_clamped_basis: dict[str, int] = {name: 0 for name in ills}

    for fpath in files:
        rgb = _load_image(fpath)
        orig_t = torch.from_numpy(rgb).float().unsqueeze(0)
        orig_vals = wb_estimate(orig_t)
        orig_clamped = abs(orig_vals["temperature"]) >= 0.999 or abs(orig_vals["tint"]) >= 0.999

        for ill_name in ills:
            cast = _apply_cast(rgb, ills[ill_name], d65)
            err, cc, cb = _b_error_after_correction(rgb, cast, orig_vals, orig_clamped)
            b_err_signed.append(err)
            b_err_abs.append(abs(err))
            clamped_any.append(cc or cb)
            per_ill_signed[ill_name].append(err)
            per_ill_abs[ill_name].append(abs(err))
            if cc:
                clamped_cast_count += 1
                per_ill_clamped_cast[ill_name] += 1
            if cb:
                clamped_basis_count += 1
                per_ill_clamped_basis[ill_name] += 1

    n_total = len(b_err_signed)

    # Main number: only K-illuminants (Farbtemperaturen)
    k_ills = [n for n in ills if n.endswith("K")]
    k_signed = [e for n in k_ills for e in per_ill_signed[n]]
    mean_k_signed = float(np.mean(k_signed))
    mean_k_abs = float(np.mean([abs(e) for e in k_signed]))
    clamped_cast_k = sum(per_ill_clamped_cast[n] for n in k_ills)
    clamped_basis_k = sum(per_ill_clamped_basis[n] for n in k_ills)
    n_k = len(k_signed)

    # Non-clamped K-illuminant errors (for unbiased verdict)
    k_nonclamped_errors: list[float] = []
    for fi in range(len(files)):
        for ii, ill_name in enumerate(ills):
            if ill_name.endswith("K"):
                idx = fi * len(ills) + ii
                if not clamped_any[idx]:
                    k_nonclamped_errors.append(b_err_signed[idx])
    mean_k_nonclamped = float(np.mean(k_nonclamped_errors)) if k_nonclamped_errors else 0.0
    n_k_nonclamped = len(k_nonclamped_errors)

    print(f"  K-illuminants ({len(k_ills)}): signed {mean_k_signed:+.3f}, |err| {mean_k_abs:.3f}, "
          f"clamped_cast {clamped_cast_k}/{n_k}, clamped_basis {clamped_basis_k}/{n_k}")
    print(f"  K-illuminants non-clamped ({n_k_nonclamped}): signed {mean_k_nonclamped:+.3f}")
    print(f"  All {n_total} combos: clamped_cast {clamped_cast_count}, clamped_basis {clamped_basis_count}")

    warm_ills = [n for n in k_ills if int(n[:-1]) <= 4000]
    cool_ills = [n for n in k_ills if int(n[:-1]) >= 5000]
    mean_signed_warm = float(np.mean([e for n in warm_ills for e in per_ill_signed[n]]))
    mean_signed_cool = float(np.mean([e for n in cool_ills for e in per_ill_signed[n]]))
    mean_abs_warm = float(np.mean([abs(e) for n in warm_ills for e in per_ill_signed[n]]))
    mean_abs_cool = float(np.mean([abs(e) for n in cool_ills for e in per_ill_signed[n]]))
    clamped_cast_warm = sum(per_ill_clamped_cast[n] for n in warm_ills)
    clamped_basis_warm = sum(per_ill_clamped_basis[n] for n in warm_ills)
    clamped_cast_cool = sum(per_ill_clamped_cast[n] for n in cool_ills)
    clamped_basis_cool = sum(per_ill_clamped_basis[n] for n in cool_ills)
    n_warm = len(warm_ills) * len(files)
    n_cool = len(cool_ills) * len(files)
    print(f"  Warm (<=4000K): signed {mean_signed_warm:+.3f}, clamped_cast {clamped_cast_warm}/{n_warm}")
    print(f"  Cool (>=5000K): signed {mean_signed_cool:+.3f}, clamped_cast {clamped_cast_cool}/{n_cool}")

    # --- Part 3: Run wb_bench for real angle/correction numbers ---
    print("\n=== Part 3: Running wb_bench (full) for angle + correction path ===")
    bench_results, err_summary, runtime_summary, corr_results, best3, est_names = run_bench(
        quick=False
    )

    # --- Write report ---
    lines: list[str] = []
    lines.append('# Auto-WB „zu warm?" — Messung (K29b)\n')
    lines.append(f"Zeitpunkt: {time.strftime('%Y-%m-%d %H:%M')}\n")

    # Part 1
    lines.append("## 1. Auto-WB auf Cast-freien Pool-Bildern\n")
    lines.append(
        "Ein neutral fotografiertes Bild sollte |temp| < 0,1 ergeben. "
        "Die Neutralmaske verwendet dieselben Konstanten wie `estimate()` "
        f"(L ∈ {L_RANGE}, Chroma < {NEUTRAL_CHROMA:.0f}).\n"
    )
    lines.append(
        "| Bild | temperature | tint | blue_share | med_b_neutral | med_b_keep (L_RANGE) | |temp| < 0,1? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in pool_results:
        ok = "✓" if abs(r["temperature"]) < 0.1 else "**✗**"
        lines.append(
            f"| {r['image']} | {r['temperature']:+.3f} | {r['tint']:+.3f} "
            f"| {r['blue_share']:.2f} | {r['median_b_neutral']:+.1f} "
            f"| {r['median_b_keep']:+.1f} | {ok} |"
        )

    n_too_warm = sum(1 for r in pool_results if r["temperature"] < -0.1)
    n_too_cool = sum(1 for r in pool_results if r["temperature"] > 0.1)
    n_ok = sum(1 for r in pool_results if abs(r["temperature"]) < 0.1)
    n_fail = len(files) - n_ok
    lines.append(
        f"\nZusammenfassung: {n_ok} neutral, {n_too_warm} mit Warmstich (temp < −0,1), "
        f"{n_too_cool} mit Kaltstich (temp > +0,1). {n_fail}/{len(files)} verfehlen |temp| < 0,1.\n"
    )

    # Hypothesis check
    lines.append("### Hypothese der Spec direkt prüfen\n")
    lines.append(
        'Spec-Verdacht: bläuliche Schatten/Himmel gelten als „neutral" → Median zu blau → '
        'Korrektur wärmt zu stark.\n'
    )
    lines.append(
        "Bestätigungskriterium: Mehrheit der Bilder ergibt temperature > +0,1 (Korrektur wärmt).\n"
    )
    if n_too_cool > len(files) / 2:
        lines.append(
            f"**Verdacht bestätigt:** {n_too_cool}/{len(files)} Bilder haben "
            f"temperature > +0,1 (Korrektur wärmt).\n"
        )
    else:
        n_pos_median = sum(1 for r in pool_results if r["median_b_neutral"] > 0)
        lines.append(
            f"**Verdacht nicht bestätigt:** {n_too_warm}/{len(files)} Bilder haben "
            f"temperature < −0,1 (Korrektur kühlt), nur "
            f"{n_too_cool}/{len(files)} wärmen. "
            f"Der Median-b* der Neutral-Pixel ist in {n_pos_median}/{len(files)} Bildern positiv.\n"
        )
    lines.append(
        "Die blue_share-Spalte zeigt den Anteil der Neutral-Pixel mit b* < 0. "
        "Der Median-b* (med_b_neutral) bestimmt das Vorzeichen der Korrektur.\n"
    )

    # Part 2
    lines.append("## 2. Invarianz-Fehler nach Korrektur (Beleuchtungs-Invarianz)\n")
    lines.append(
        "Fehler = mittlere b*-Differenz zwischen WB(cast) und WB(orig). "
        "Positiv = Ergebnis wärmer als das Auto-WB des Originals; negativ = kälter.\n"
    )
    lines.append(f"Alle {len(files)} Bilder × {len(ills)} Lichtfarben = {n_total} Kombinationen.\n")
    lines.append("### Hauptzahl (nur Farbtemperaturen, Namen enden auf K)\n")
    lines.append("| Metrik | Wert |")
    lines.append("|---|---|")
    lines.append(f"| Mittlerer b*-Fehler (nur K-Lichtfarben, vorzeichenbehaftet) | {mean_k_signed:+.3f} |")
    lines.append(f"| Mittlerer |b*-Fehler| (nur K-Lichtfarben) | {mean_k_abs:.3f} |")
    lines.append(f"| Mittlerer b*-Fehler (nur K, **ohne geklemmte**) | {mean_k_nonclamped:+.3f} ({n_k_nonclamped}/{n_k}) |")
    lines.append(f"| Geklemmt Cast (±1-Limit) | {clamped_cast_k}/{n_k} Kombinationen |")
    lines.append(f"| Geklemmt Basis (Original ±1-Limit) | {clamped_basis_k}/{n_k} Kombinationen |\n")

    lines.append("### Je Lichtfarbe (alle 12)\n")
    lines.append(
        "| Lichtfarbe | b*-Fehler (Mittel) | |b*-Fehler| (Mittel) "
        "| Geklemmt Cast | Geklemmt Basis |"
    )
    lines.append("|---|---|---|---|---|")
    for name in ills:
        s = float(np.mean(per_ill_signed[name]))
        a = float(np.mean(per_ill_abs[name]))
        cc = per_ill_clamped_cast[name]
        cb = per_ill_clamped_basis[name]
        cc_str = f"{cc}/{len(files)}" if cc else "0"
        cb_str = f"{cb}/{len(files)}" if cb else "0"
        lines.append(f"| {name} | {s:+.3f} | {a:.3f} | {cc_str} | {cb_str} |")
    lines.append("")

    lines.append("### Aufschlüsselung warm vs. kalt (nur K-Lichtfarben)\n")
    lines.append(
        "| Gruppe | b*-Fehler (Mittel) | |b*-Fehler| (Mittel) "
        "| Geklemmt Cast | Geklemmt Basis |"
    )
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Warm (≤4000K: {', '.join(warm_ills)}) | {mean_signed_warm:+.3f} "
        f"| {mean_abs_warm:.3f} | {clamped_cast_warm}/{n_warm} | {clamped_basis_warm}/{n_warm} |"
    )
    lines.append(
        f"| Kalt (≥5000K: {', '.join(cool_ills)}) | {mean_signed_cool:+.3f} "
        f"| {mean_abs_cool:.3f} | {clamped_cast_cool}/{n_cool} | {clamped_basis_cool}/{n_cool} |"
    )
    lines.append("")

    # Part 3: real wb_bench numbers
    lines.append("## 3. Winkelfehler + Korrekturweg (aus `wb_bench.py`, neu ausgeführt)\n")
    lines.append("### Winkelfehler (Median) je Schätzer\n")
    lines.append("| Schätzer | Median (°) | Mittel (°) |")
    lines.append("|---|---|---|")
    ranked = sorted(est_names, key=lambda n: err_summary[n]["median"])
    for name in ranked:
        e = err_summary[name]
        lines.append(f"| {name} | {e['median']:.2f} | {e['mean']:.2f} |")
    lines.append("")
    ours_median = err_summary["ours"]["median"]
    worse_than_ours = [n for n in est_names if err_summary[n]["median"] > ours_median]
    worse_str = ", ".join(f"`{n}`" for n in worse_than_ours) if worse_than_ours else "kein anderer"
    lines.append(
        f"Der in `whitebalance.estimate()` verwendete Schätzer (`ours`) hat einen "
        f"Median-Winkelfehler von {ours_median:.2f}°. Schlechter: {worse_str}.\n"
    )

    lines.append("### Korrekturweg (bester Schätzer × Korrekturweg, ΔE76 zum Original)\n")
    lines.append(
        'Die Tabelle zeigt den ΔE76 zwischen korrigiertem und unverfälschtem Original '
        'für die drei besten Schätzer (nach Winkelfehler). '
        '„unser Weg" = Lab-b*-Shift wie in `whitebalance._temp_tint`.\n'
    )
    lines.append("| Schätzer | von Kries (Mittel) | unser Weg (Mittel) |")
    lines.append("|---|---|---|")
    for name in best3:
        c = corr_results[name]
        lines.append(
            f"| {name} | {c['von_kries_mean']:.3f} | {c['our_path_mean']:.3f} |"
        )
    lines.append("")
    lines.append(
        f'Hinweis: `best3[0]` = `{best3[0]}` ist der beste Schätzer nach Winkelfehler, '
        f'nicht der in `estimate()` verwendete (`ours`, {ours_median:.2f}°). '
        f'Die ΔE-Zahlen hier beschreiben die Kombination „bester Schätzer × Korrekturweg", '
        f'keine Aussage über `estimate()` allein.\n'
    )

    # Part 4: Verdict
    lines.append("## 4. Urteil\n")

    # Verdict based on non-clamped mean (clamping is a range limit, not an estimator bias)
    if mean_k_nonclamped > 1.0:
        lines.append(
            f"**Warm-Bias im Schätzer: JA.** Der mittlere b*-Fehler über die "
            f"K-Lichtfarben ohne geklemmte Fälle beträgt {mean_k_nonclamped:+.3f} "
            f"(Schwelle: +1,0).\n"
        )
    else:
        lines.append(
            f"**Warm-Bias im Schätzer: NEIN** (Hauptzahl {mean_k_signed:+.3f} nur durch Klemmung). "
            f"Ohne die {clamped_cast_k} geklemmten K-Fälle beträgt der mittlere b*-Fehler "
            f"{mean_k_nonclamped:+.3f} (Schwelle: +1,0).\n"
        )

    # Reichweitengrenze (only if there are clamped cases)
    if clamped_cast_count > 0:
        # Find which K-illuminants have clamped cases
        clamped_ill_names = [n for n in k_ills if per_ill_clamped_cast[n] > 0]
        lines.append(
            "### Reichweitengrenze (±1-Limit in `_to_values`)\n"
        )
        lines.append(
            f"{clamped_cast_k}/{n_k} K-Kombinationen erreichen das ±1-Limit "
            f"(max. ±20 b*). Betroffene Lichtfarben: {', '.join(clamped_ill_names)}. "
            f"Der Cast ist größer als 20 b*; die Korrektur kann ihn nicht vollständig "
            f"entfernen. Das ist eine Reichweitengrenze des Korrekturwegs, kein Schätzfehler.\n"
        )

    # Erklärung: warum der Eindruck „zu warm" entstehen kann (nur wenn kein Bias)
    if mean_k_nonclamped <= 1.0:
        lines.append(
            '### Warum der Eindruck „zu warm" entstehen kann\n'
        )
        lines.append(
            f"1. **Pool-Bilder:** {n_too_warm}/{len(files)} Bilder haben einen natürlichen "
            f"Warm-Farbstich (temp < −0,1). Die WB kühlt sie, aber das Ergebnis kann "
            f"immer noch warm wirken, wenn der ursprüngliche Cast stark war.\n"
        )
        if clamped_cast_count > 0 and clamped_ill_names:
            lines.append(
                f"2. **Reichweitengrenze:** Bei sehr warmem Licht ({', '.join(clamped_ill_names)}) "
                f"ist der Cast > 20 b*; die ±1-Klemmung verhindert eine vollständige Korrektur, "
                f"der Restfehler ist systematisch warm.\n"
            )
        lines.append(
            "3. **Korrekturweg:** Der Lab-b*-Shift korrigiert nur b* und a* linear. "
            "Für komplexe Farbstiche (magenta/cyan) ist dieser Weg ungenau "
            "(ΔE76 = " + f"{corr_results[best3[0]]['our_path_mean']:.1f}" +
            " vs. " + f"{corr_results[best3[0]]['von_kries_mean']:.1f}" + " für von Kries).\n"
        )

    # Einordnung der Metriken (keine Entscheidung, keine offene Frage)
    lines.append("### Einordnung der Metriken\n")
    lines.append(
        "- Der b*-Fehler misst die Differenz zwischen Auto-WB(Bild mit Cast) und "
        "Auto-WB(Original), nicht das absolute b* nach der Korrektur. Einen absoluten "
        "Sollwert gibt es bei Fotos ohne bekannte Lichtfarbe nicht.\n"
    )
    lines.append(
        "- Die Pool-Bilder sind nicht nachweislich cast-frei; |temp| < 0,1 ist dort "
        "ein Hinweis, kein Fehlermaß.\n"
    )
    lines.append(
        "- Die Klemmung bei ±1 ist bestehendes Verhalten und nicht Gegenstand dieses "
        "Pakets; `whitebalance.py` bleibt unverändert.\n"
    )
    lines.append(
        "- Bestehende Fehler anderer Pakete in der Gesamtsuite (nicht WB): "
        "test_parade_glow, test_plugin_k25c, test_plugin_k27d.\n"
    )

    # Schätzfehler vs. Reichweitengrenze getrennt
    lines.append(
        "### Schätzfehler vs. Reichweitengrenze (getrennt)\n"
    )
    lines.append(
        f"- **Schätzfehler von `ours`:** Winkelfehler Median {ours_median:.2f}°. "
        f"Der Schätzer ist ein Faktor für den Gesamtfehler.\n"
    )
    if clamped_cast_count > 0:
        lines.append(
            f"- **Reichweitengrenze:** {clamped_cast_count}/{n_total} Kombinationen "
            f"geklemmt (max. ±20 b*). Unabhängig vom Schätzfehler.\n"
        )

    dt = time.perf_counter() - t0
    lines.append(f"\n---\nLaufzeit: {dt:.1f} s\n")

    report_path = _REPORTS / "autowb_check.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Fertig in {dt:.1f} s")


if __name__ == "__main__":
    main()
