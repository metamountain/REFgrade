"""LUTs vermessen und katalogisieren.

Aufruf:
    python tools/classify_luts.py [ordner]

Standard-Ordner: die LUT-Bibliothek des Projekts (RefGrade/luts).
Nur lesen. Ausgabe: luts_catalog.json im Paketordner.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.lut import read_cube, apply as lut_apply
from refgrade.analysis import analyze_image, compare
from refgrade.metrics import structure, banding
from refgrade.color import rgb_to_lab, luma

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RefGrade")

_DEFAULT_LUT_DIR = str(Path(__file__).resolve().parents[2] / "luts")
_PKG_DIR = Path(__file__).resolve().parent.parent
_OUT_PATH = _PKG_DIR / "luts_catalog.json"


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _gray_ramp(device: torch.device) -> torch.Tensor:
    """Graurampe 0..1, 1x256x1x3."""
    vals = torch.linspace(0.0, 1.0, 256, device=device)
    return vals.view(1, 1, 256, 1).expand(1, 1, 256, 3).contiguous()


def _green_patch(device: torch.device) -> torch.Tensor:
    """Synthetisches Grün-Testbild: 64x64, Hue 120°, Sättigung 0.5, Luma 0.6 (außer Schattenzone)."""
    # HSV: H=120°, S=0.5, V=0.6 → RGB
    h, s, v = 120.0, 0.5, 0.6
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    # Sektor 2 (120°): R=0, G=C, B=X
    r, g, b = m, c + m, x + m
    arr = np.full((64, 64, 3), [r, g, b], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _load_skimage_images(device: torch.device) -> dict[str, torch.Tensor]:
    """skimage.data Testbilder laden → {name: [1,H,W,3] float32 0..1}."""
    from skimage.data import astronaut, coffee, chelsea, rocket

    imgs: dict[str, torch.Tensor] = {}
    for name, fn in [("astronaut", astronaut), ("coffee", coffee),
                     ("chelsea", chelsea), ("rocket", rocket)]:
        arr = fn()
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        imgs[name] = t
    return imgs


def _green_sat_change(orig: torch.Tensor, result: torch.Tensor) -> float:
    """Sättigungsänderung im Grün-Band (Hue 90–150°)."""
    def _green_sat(img: torch.Tensor) -> float:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        mx = img.amax(dim=-1)
        mn = img.amin(dim=-1)
        delta = mx - mn
        S = delta / mx.clamp_min(1e-6)
        # Hue
        h = torch.zeros_like(mx)
        h = torch.where((mx == r) & (delta > 1e-6),
                        (((g - b) / delta.clamp_min(1e-6)) % 6.0) * 60.0,
                        torch.where((mx == g) & (delta > 1e-6),
                                    ((b - r) / delta.clamp_min(1e-6) + 2.0) * 60.0,
                                    ((r - g) / delta.clamp_min(1e-6) + 4.0) * 60.0))
        h = h % 360.0
        green_mask = (h >= 90.0) & (h <= 150.0) & (delta > 0.05)
        if green_mask.sum() < 10:
            return 0.0
        return float(S[green_mask].mean().item())

    return _green_sat(result) - _green_sat(orig)


def _shadow_hue(orig: torch.Tensor, result: torch.Tensor) -> float:
    """Schattentönung (Hue) im Ergebnis, aus Lab a/b der Schattenzone."""
    lab = rgb_to_lab(result)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    # Schatten: L < 30
    mask = L < 30.0
    if mask.sum() < 10:
        return 0.0
    mean_a = float(a[mask].mean().item())
    mean_b = float(b[mask].mean().item())
    import math
    hue = (math.degrees(math.atan2(mean_b, mean_a)) + 360.0) % 360.0
    return hue


def _classify(
    ramp_metrics: dict,
    avg_sliders: dict[str, float],
    green_sat: float,
    shadow_hue_val: float,
) -> list[str]:
    """Kategorien aus den Messwerten ableiten. Mehrfachkategorien erlaubt."""
    cats: list[str] = []

    # log_input: Graurampe stark aufgehellt (Median-L +20) und Kontrast massiv
    if (ramp_metrics.get("exposure", 0) >= 1.5
            and ramp_metrics.get("contrast", 0) >= 1.5):
        cats.append("log_input")

    # bleach_bypass: Sättigung ≤ −0.3 und Kontrast ≥ +0.15
    sat_change = avg_sliders.get("saturation", 1.0) - 1.0
    con_change = avg_sliders.get("contrast", 1.0) - 1.0
    if sat_change <= -0.3 and con_change >= 0.15:
        cats.append("bleach_bypass")

    # warm / cold
    temp = avg_sliders.get("temperature", 0.0)
    if temp >= 0.15:
        cats.append("warm")
    elif temp <= -0.15:
        cats.append("cold")

    # nature: Grün-Band-Sättigung ≥ +0.1
    if green_sat >= 0.1:
        cats.append("nature")

    # deep_blacks: Schwarzpunkt gesenkt (compare lift_m ≥ +0.01) und Schattentönung 15–60°
    # compare lift_m = (bp_src - bp_ref)/100: positiv wenn Ergebnis dunklere Blacks hat
    lift_m = avg_sliders.get("lift_m", 0.0)
    if lift_m >= 0.01 and 15.0 <= shadow_hue_val <= 60.0:
        cats.append("deep_blacks")

    if not cats:
        cats.append("other")
    return cats


def _read_lut_header(path: Path) -> dict[str, str]:
    """TITLE und Copyright-Kommentar aus der .cube-Datei lesen."""
    title = ""
    copyright_ = ""
    try:
        with open(path, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                line = line.strip()
                upper = line.upper()
                if upper.startswith("TITLE") and not title:
                    parts = line.split('"')
                    if len(parts) >= 2:
                        title = parts[1]
                elif line.startswith("#") and not copyright_:
                    if "copyright" in line.lower() or "license" in line.lower():
                        copyright_ = line.lstrip("# ").strip()
    except OSError:
        pass
    return {"title": title, "copyright": copyright_}


def classify_lut(lut: dict, device: torch.device) -> dict:
    """Ein LUT vermessen: Testset anwenden, analysieren, kategorisieren."""
    # Testbilder
    ramp = _gray_ramp(device)
    green = _green_patch(device)
    skimage_imgs = _load_skimage_images(device)

    ramp_metrics: dict[str, float] = {}
    slider_sums: dict[str, float] = {}
    n_images = 0
    struct_vals: list[float] = []
    band_vals: list[float] = []

    # Graurampe: Log-Erkennung + neutrale Schattentönung
    ramp_result = lut_apply(ramp, lut).clamp(0.0, 1.0)
    a_ramp_orig = analyze_image(ramp)
    a_ramp_res = analyze_image(ramp_result)
    ramp_metrics = compare(a_ramp_orig, a_ramp_res)["sliders"]
    ramp_shadow_hue = _shadow_hue(ramp, ramp_result)

    # Grün-Patch: Sättigungsänderung im Grün-Band
    green_result = lut_apply(green, lut).clamp(0.0, 1.0)
    green_sat_change = _green_sat_change(green, green_result)

    # Skimage-Bilder: Regler + Metriken
    for name, img in skimage_imgs.items():
        result = lut_apply(img, lut).clamp(0.0, 1.0)
        a_orig = analyze_image(img)
        a_res = analyze_image(result)
        sliders = compare(a_orig, a_res)["sliders"]
        for k, v in sliders.items():
            slider_sums[k] = slider_sums.get(k, 0.0) + v
        n_images += 1
        struct_vals.append(structure(img, result))
        band_vals.append(banding(img, result))

    # Durchschnitt über Skimage-Bilder
    avg_sliders: dict[str, float] = {}
    if n_images > 0:
        for k in slider_sums:
            avg_sliders[k] = slider_sums[k] / n_images

    # Kategorien
    categories = _classify(ramp_metrics, avg_sliders, green_sat_change,
                           ramp_shadow_hue)

    return {
        "ramp": {k: round(v, 4) for k, v in ramp_metrics.items()},
        "avg_sliders": {k: round(v, 4) for k, v in avg_sliders.items()},
        "green_sat_change": round(green_sat_change, 4),
        "shadow_hue": round(ramp_shadow_hue, 1),
        "structure_mean": round(float(np.mean(struct_vals)), 4) if struct_vals else 0.0,
        "banding_mean": round(float(np.mean(band_vals)), 4) if band_vals else 0.0,
        "categories": categories,
    }


def main() -> None:
    lut_dir = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_LUT_DIR
    dir_path = Path(lut_dir)
    if not dir_path.is_dir():
        log.error("Ordner existiert nicht: %s", lut_dir)
        sys.exit(1)

    device = _get_device()
    log.info("Gerät: %s", device)
    log.info("LUT-Ordner: %s", lut_dir)

    cube_files = sorted(dir_path.glob("*.cube")) + sorted(dir_path.glob("*.CUBE"))
    log.info("LUTs gefunden: %d", len(cube_files))

    catalog: dict[str, dict] = {}
    for f in cube_files:
        name = f.stem
        log.info("Vermesse: %s", name)
        try:
            lut = read_cube(str(f))
            header = _read_lut_header(f)
            metrics = classify_lut(lut, device)
            catalog[name] = {
                "file": f.name,
                "title": header["title"],
                "copyright": header["copyright"],
                **metrics,
            }
        except Exception as e:
            log.warning("Fehler bei %s: %s", name, e)
            catalog[name] = {"file": f.name, "error": str(e)}

    # JSON schreiben
    _OUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    log.info("Katalog geschrieben: %s", _OUT_PATH)

    # Zusammenfassung
    cat_counts: dict[str, int] = {}
    for entry in catalog.values():
        for c in entry.get("categories", ["error"]):
            cat_counts[c] = cat_counts.get(c, 0) + 1
    log.info("Kategorien: %s", cat_counts)


if __name__ == "__main__":
    main()
