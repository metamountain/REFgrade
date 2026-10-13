"""LUTs vermessen und in Kategorien-Unterordner einsortieren.

Aufruf:
    python tools/sort_luts.py <lut-ordner> [--apply]

Ohne --apply: Tabelle (Datei → Ordner → Grund).
Mit --apply: Dateien in Unterordner verschieben, sort_luts.log schreiben.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.lut import read_cube, apply as lut_apply
from refgrade.analysis import analyze_image, compare
from refgrade.color import rgb_to_lab

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RefGrade")

CATEGORIES = ("Camera", "Film", "Warm", "Cool", "Moody", "Natural", "B&W", "Creative", "RefGrade")

_CAMERA_NAME_HINTS = ("log", "rec709", "s-log", "vlog", "logc")
_FILM_NAME_HINTS = ("kodak", "fuji", "film", "bleach", "portra", "velvia", "ektar")


def _gray_ramp(device: torch.device) -> torch.Tensor:
    """Graurampe 1x1x256x3, Werte 0..1."""
    vals = torch.linspace(0.0, 1.0, 256, device=device)
    return vals.view(1, 1, 256, 1).expand(1, 1, 256, 3).contiguous()


def _test_image(device: torch.device) -> torch.Tensor:
    """Synthetisches Testbild 64x64 mit Farb- und Helligkeitsvariation."""
    h, w = 64, 64
    x = torch.linspace(0, 1, w, device=device)
    y = torch.linspace(0, 1, h, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    r = (0.5 + 0.4 * torch.sin(2 * math.pi * xx)).clamp(0, 1)
    g = (0.5 + 0.4 * torch.sin(2 * math.pi * yy + 2)).clamp(0, 1)
    b = (0.5 + 0.4 * torch.cos(2 * math.pi * (xx + yy))).clamp(0, 1)
    img = torch.stack([r, g, b], dim=-1).unsqueeze(0)
    return img


def _read_title(path: Path) -> str:
    """TITLE-Zeile aus der .cube-Datei lesen."""
    try:
        with open(path, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.upper().startswith("TITLE"):
                    parts = line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
    except OSError:
        pass
    return ""


def _classify_one(
    lut: dict,
    filename: str,
    title: str,
    device: torch.device,
) -> tuple[str, str]:
    """Ein LUT vermessen → (Kategorie, Grund)."""
    name_lower = filename.lower()

    # Testbilder
    ramp = _gray_ramp(device)
    test_img = _test_image(device)

    # LUT anwenden
    ramp_result = lut_apply(ramp, lut).clamp(0.0, 1.0)
    test_result = lut_apply(test_img, lut).clamp(0.0, 1.0)

    # Analysen
    a_orig = analyze_image(test_img)
    a_res = analyze_image(test_result)
    sliders = compare(a_orig, a_res)["sliders"]

    # Sättigung: Chroma-Median-Verhältnis
    sat_src = max(a_orig["color"]["saturation"]["chroma_median"], 1e-6)
    sat_res = a_res["color"]["saturation"]["chroma_median"]
    sat_ratio = sat_res / sat_src

    # Graurampe: Log-Erkennung
    a_ramp_orig = analyze_image(ramp)
    a_ramp_res = analyze_image(ramp_result)
    ramp_sliders = compare(a_ramp_orig, a_ramp_res)["sliders"]

    # Farbtonverschiebung: pro Pixel Hue (atan2(b,a)) vergleichen,
    # zirkuläre Differenz → mittlerer absoluter Wert.
    # Translation (Temperatur) ändert den Hue kaum; Rotation (Hue-Shift) ändert ihn überall.
    lab_o = rgb_to_lab(test_img)
    lab_r = rgb_to_lab(test_result)
    hue_o = torch.atan2(lab_o[..., 2], lab_o[..., 1])
    hue_r = torch.atan2(lab_r[..., 2], lab_r[..., 1])
    diff = hue_r - hue_o
    circular_diff = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
    hue_shift_abs = (circular_diff * (180.0 / math.pi)).mean().item()

    # Zonen-Tönung (Teal & Orange Erkennung)
    sh_hue = a_res["color"]["zones"]["shadows"]["hue"]
    hi_hue = a_res["color"]["zones"]["highlights"]["hue"]
    # Gegenläufig: Schatten kühl (180-270°) + Lichter warm (0-90°) oder umgekehrt
    shadows_cool = 180 <= sh_hue <= 270
    highlights_warm = 0 <= hi_hue <= 90 or hi_hue >= 315
    shadows_warm = 0 <= sh_hue <= 90 or sh_hue >= 315
    highlights_cool = 180 <= hi_hue <= 270
    counter_toning = (shadows_cool and highlights_warm) or (shadows_warm and highlights_cool)

    # Temperatur
    temperature = sliders.get("temperature", 0.0)

    # Helligkeit und Kontrast
    l_delta = a_res["tone"]["median_l"] - a_orig["tone"]["median_l"]
    contrast_src = max(a_orig["tone"]["contrast"], 1e-6)
    contrast_res = a_res["tone"]["contrast"]
    contrast_ratio = contrast_res / contrast_src

    # Regeln in Reihenfolge
    # 1. B&W: mittlere Sättigung < 15 % der Eingangssättigung
    if sat_ratio < 0.15:
        return "B&W", f"Sättigung {sat_ratio * 100:.0f} % der Original"

    # 2. Camera: Log-Eingang oder Name
    is_log = (ramp_sliders.get("exposure", 0) >= 1.5
              and ramp_sliders.get("contrast", 0) >= 1.5)
    name_has_camera = any(h in name_lower for h in _CAMERA_NAME_HINTS)
    if is_log or name_has_camera:
        reason = "Log-Eingang" if is_log else f"Name enthält '{[h for h in _CAMERA_NAME_HINTS if h in name_lower][0]}'"
        return "Camera", reason

    # 3. Film: Name
    film_hint = [h for h in _FILM_NAME_HINTS if h in name_lower]
    if film_hint:
        return "Film", f"Name enthält '{film_hint[0]}'"

    # 4. Creative: Hue-Shift > 20° oder gegenläufige Tönung
    if hue_shift_abs > 20 or counter_toning:
        reason = f"Hue-Shift {hue_shift_abs:.0f}°" if hue_shift_abs > 20 else "Teal & Orange"
        return "Creative", reason

    # 5. Warm / Cool
    if temperature > 0.15:
        return "Warm", f"Temperatur {temperature:+.2f}"
    if temperature < -0.15:
        return "Cool", f"Temperatur {temperature:+.2f}"

    # 6. Moody: Helligkeit sinkt > 5 L* oder Sättigung sinkt > 20 %, bei steigendem Kontrast
    moody_bright = l_delta < -5.0
    moody_sat = sat_ratio < 0.8
    if (moody_bright or moody_sat) and contrast_ratio > 1.0:
        reason = f"ΔL {l_delta:+.1f}" if moody_bright else f"Sättigung {sat_ratio * 100:.0f} %"
        return "Moody", f"{reason}, Kontrast {contrast_ratio:.2f}×"

    # 7. Natural
    # 8. RefGrade (Titel)
    if title.startswith("RefGrade"):
        return "RefGrade", "Titel beginnt mit 'RefGrade'"

    return "Natural", "keine spezielle Eigenschaft"


def sort_luts(lut_dir: Path, apply: bool = False) -> list[tuple[str, str, str, str]]:
    """Alle .cube-Dateien in `lut_dir` vermessen und klassifizieren.

    Returns:
        Liste von (Dateiname, Kategorie, Grund, Zielpfad).
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    log.info("Gerät: %s", device)

    cube_files = sorted(lut_dir.glob("*.cube"), key=lambda p: p.name.lower())
    cube_files += sorted(lut_dir.glob("*.CUBE"), key=lambda p: p.name.lower())
    log.info("LUTs gefunden: %d", len(cube_files))

    results: list[tuple[str, str, str, str]] = []
    for f in cube_files:
        title = _read_title(f)
        try:
            lut = read_cube(str(f))
            cat, reason = _classify_one(lut, f.stem, title, device)
            target = str(f.parent / cat / f.name) if apply else cat
            results.append((f.name, cat, reason, target))
            log.info("%-30s → %-10s (%s)", f.name, cat, reason)
        except Exception as e:
            log.warning("Fehler bei %s: %s", f.name, e)
            results.append((f.name, "Natural", f"Fehler: {e}", str(f)))

    if apply:
        _apply_moves(lut_dir, results)

    return results


def _apply_moves(lut_dir: Path, results: list[tuple[str, str, str, str]]) -> None:
    """Dateien verschieben und sort_luts.log schreiben."""
    log_lines = []
    for fname, cat, reason, _ in results:
        src = lut_dir / fname
        if not src.exists():
            continue
        dest_dir = lut_dir / cat
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / fname
        if dest.exists():
            # Name-Kollision: _2, _3, …
            stem, suffix = src.stem, src.suffix
            i = 2
            while dest.exists():
                dest = dest_dir / f"{stem}_{i}{suffix}"
                i += 1
        src.rename(dest)
        log_lines.append(f"{fname} → {cat}/{dest.name} ({reason})")
        log.info("Verschoben: %s → %s", fname, dest)

    log_path = lut_dir / "sort_luts.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log.info("Log geschrieben: %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="LUTs in Kategorien einsortieren")
    parser.add_argument("lut_dir", help="Pfad zum LUT-Ordner")
    parser.add_argument("--apply", action="store_true", help="Dateien tatsächlich verschieben")
    args = parser.parse_args()

    lut_dir = Path(args.lut_dir)
    if not lut_dir.is_dir():
        log.error("Ordner existiert nicht: %s", lut_dir)
        sys.exit(1)

    results = sort_luts(lut_dir, apply=args.apply)

    # Tabelle ausgeben
    if not args.apply:
        print(f"\n{'Datei':<30} {'Ordner':<12} Grund")
        print("-" * 70)
        for fname, cat, reason, _ in results:
            print(f"{fname:<30} {cat:<12} {reason}")
        print(f"\n{len(results)} LUTs. Mit --apply verschieben.")


if __name__ == "__main__":
    main()
