"""Shootout: Bildraster, HTML-Bericht, Blindtest.

Aufruf:
    python.exe tools/shootout.py --cases <ordner> --candidates <liste> [--blind] [--crop x,y,w,h]
    python.exe tools/shootout.py --score <ordner>
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.methods import get as get_method, list_all
from refgrade.methods import classic  # noqa: F401
from refgrade.presets import PRESETS
from refgrade.metrics import delta_e00, structure, banding
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from refgrade.lut import read_cube, apply as lut_apply
from refgrade.color import rgb_to_lab

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RefGrade")

# --- Konstanten ---
_CELL_SIZE = 512
_JPEG_QUALITY = 95
_MAX_COLS = 8
_BLIND_SEED = 42
_POOL_DIR = Path(__file__).resolve().parents[2] / "bench" / "pool"
_SHOOTOUT_DIR = Path(__file__).resolve().parents[1] / "bench" / "shootout"
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# --- Datenstrukturen ---

@dataclass
class Case:
    name: str
    image: torch.Tensor
    reference: torch.Tensor | None = None
    truth: torch.Tensor | None = None
    image_weights: torch.Tensor | None = None
    reference_weights: torch.Tensor | None = None
    apply_mask: torch.Tensor | None = None


@dataclass
class Candidate:
    label: str
    kind: str  # "method", "preset", "lut"
    value: str  # method name, preset name, or lut path/dir
    fn: Any = None  # fitted callable or lut dict


@dataclass
class CellResult:
    image: torch.Tensor
    label: str
    delta_e00_mean: float | None = None
    structure_val: float | None = None
    banding_val: float | None = None


# --- Laden ---

def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    """Bild laden → [1,H,W,3] float32 0..1."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _load_mask(path: str, device: torch.device) -> torch.Tensor:
    """Maske laden → [1,H,W] float32 0..1."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _find_image(dir_path: Path) -> Path | None:
    """Erstes Bild im Ordner finden."""
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in _VALID_EXTS:
            return f
    return None


def load_cases(cases_dir: str, device: torch.device) -> list[Case]:
    """Fälle aus Ordner laden + synthetische T1b/T4."""
    cases: list[Case] = []
    root = Path(cases_dir)
    if not root.is_dir():
        log.warning("Cases-Ordner existiert nicht: %s", cases_dir)
        return cases

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        img_path = _find_image(case_dir)
        if img_path is None:
            continue

        name = case_dir.name
        image = _load_image_tensor(str(img_path), device)

        # Optional: reference
        ref_dir = case_dir / "reference"
        ref_path = None
        if ref_dir.is_dir():
            ref_path = _find_image(ref_dir)
        else:
            # reference.* direkt im Ordner
            for f in sorted(case_dir.iterdir()):
                if f.name.startswith("reference") and f.suffix.lower() in _VALID_EXTS:
                    ref_path = f
                    break

        reference = _load_image_tensor(str(ref_path), device) if ref_path else None

        # Masken
        image_weights = None
        ref_weights = None
        apply_mask = None

        for f in case_dir.iterdir():
            if f.name == "image_ignore.png":
                m = _load_mask(str(f), device)
                image_weights = 1.0 - m
            elif f.name == "reference_ignore.png":
                m = _load_mask(str(f), device)
                ref_weights = 1.0 - m
            elif f.name == "apply.png":
                apply_mask = _load_mask(str(f), device)

        # Wahrheit: truth.* im Ordner
        truth_path = None
        for f in sorted(case_dir.iterdir()):
            if f.name.startswith("truth") and f.suffix.lower() in _VALID_EXTS:
                truth_path = f
                break
        truth = _load_image_tensor(str(truth_path), device) if truth_path else None

        cases.append(Case(
            name=name,
            image=image,
            reference=reference,
            truth=truth,
            image_weights=image_weights,
            reference_weights=ref_weights,
            apply_mask=apply_mask,
        ))

    # Synthetische Fälle T1b und T4
    cases.extend(_synthetic_cases(device))
    return cases


def _random_slider_values(seed: int) -> dict[str, float]:
    """Seed-feste Zufalls-Reglerwerte."""
    rng = np.random.default_rng(seed)
    values = {}
    for s in SLIDERS:
        if s.core:
            continue
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
            val = float(rng.uniform(s.min, s.max))
        values[s.name] = round(val, 3)
    return values


def _synthetic_cases(device: torch.device) -> list[Case]:
    """Synthetische Fälle T1b und T4 (nur dort gibt es eine Wahrheit)."""
    cases: list[Case] = []
    if not os.path.isdir(_POOL_DIR):
        return cases

    files = sorted(
        f for f in os.listdir(_POOL_DIR)
        if os.path.splitext(f)[1].lower() in _VALID_EXTS
    )
    if not files:
        return cases

    # T1b: Vorlage nur Ausschnitt 40%, Fehler auf ganzem Bild
    pool_path = os.path.join(_POOL_DIR, files[0])
    img = _load_image_tensor(pool_path, device)
    slider_vals = _random_slider_values(0)
    truth = grade(img, slider_vals).clamp(0.0, 1.0)

    # Ausschnitt 40% oben-links
    h, w = img.shape[1], img.shape[2]
    ch, cw = h * 40 // 100, w * 40 // 100
    ref_crop = truth[:, :ch, :cw, :]

    cases.append(Case(
        name="T1b_synthetic",
        image=img,
        reference=ref_crop,
        truth=truth,
    ))

    # T4: dominante Fremdfläche
    if len(files) > 1:
        pool_path2 = os.path.join(_POOL_DIR, files[1])
        img2 = _load_image_tensor(pool_path2, device)
        slider_vals2 = _random_slider_values(1)
        truth2 = grade(img2, slider_vals2).clamp(0.0, 1.0)

        # Baum in linker Hälfte
        h2, w2 = img2.shape[1], img2.shape[2]
        tree_mask = torch.zeros(1, h2, w2, device=device)
        tree_mask[:, :, :w2 // 2] = 1.0
        torch.manual_seed(0)
        tree = torch.zeros(1, h2, w2, 3, device=device)
        tree[..., 1] = 0.55 + 0.25 * torch.rand(1, h2, w2, device=device)
        tree[..., 0] = 0.10 + 0.10 * torch.rand(1, h2, w2, device=device)
        tree[..., 2] = 0.05 + 0.08 * torch.rand(1, h2, w2, device=device)

        src = img2.clone()
        src[tree_mask > 0.5] = tree[tree_mask > 0.5]
        ref2 = grade(img2, slider_vals2).clamp(0.0, 1.0)

        cases.append(Case(
            name="T4_synthetic",
            image=src,
            reference=ref2,
            truth=truth2,
            image_weights=1.0 - tree_mask,
        ))

    return cases


def parse_candidates(spec: str) -> list[dict[str, str]]:
    """Kandidatenliste parsen: 'method:reinhard,preset:warm_light,lut:file.cube'."""
    items = [s.strip() for s in spec.split(",") if s.strip()]
    return [{"spec": item} for item in items]


def fit_candidates(
    candidates: list[dict[str, str]],
    case: Case,
    device: torch.device,
) -> list[Candidate]:
    """Kandidaten für einen Fall fitten."""
    results: list[Candidate] = []
    for item in candidates:
        spec = item["spec"]
        if ":" not in spec:
            log.warning("Ungültiger Kandidat (kein ':'): %s", spec)
            continue

        kind, value = spec.split(":", 1)
        kind = kind.strip()
        value = value.strip()

        try:
            if kind == "method":
                fit_fn = get_method(value)
                fn = fit_fn(
                    case.image,
                    case.reference if case.reference is not None else case.image,
                    case.image_weights,
                    case.reference_weights,
                )
                results.append(Candidate(label=value, kind=kind, value=value, fn=fn))

            elif kind == "preset":
                if value not in PRESETS:
                    log.warning("Unbekanntes Preset: %s", value)
                    continue
                vals = PRESETS[value]
                fn = lambda img, v=vals: grade(img, v)
                results.append(Candidate(label=value, kind=kind, value=value, fn=fn))

            elif kind == "lut":
                if value.startswith("category="):
                    cat_name = value[9:]
                    catalog_path = Path(__file__).resolve().parent.parent / "luts_catalog.json"
                    if not catalog_path.exists():
                        log.warning("luts_catalog.json fehlt — classify_luts.py ausführen")
                        continue
                    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                    lut_dir = str(Path(__file__).resolve().parents[2] / "luts")
                    for lut_name, entry in catalog.items():
                        if cat_name not in entry.get("categories", []):
                            continue
                        if "log_input" in entry.get("categories", []):
                            continue
                        cube_file = entry.get("file", lut_name + ".cube")
                        lut_path = os.path.join(lut_dir, cube_file)
                        if not os.path.isfile(lut_path):
                            continue
                        lut = read_cube(lut_path)
                        label = lut_name
                        fn = lambda img, l=lut: lut_apply(img, l)
                        results.append(Candidate(label=label, kind=kind, value=value, fn=fn))
                elif value.startswith("dir="):
                    lut_dir = value[4:]
                    cube_files = sorted(
                        f for f in os.listdir(lut_dir)
                        if f.endswith(".cube")
                    )
                    if not cube_files:
                        log.warning("Keine .cube in %s", lut_dir)
                        continue
                    lut_path = os.path.join(lut_dir, cube_files[0])
                    lut = read_cube(lut_path)
                    label = os.path.basename(lut_path)
                    fn = lambda img, l=lut: lut_apply(img, l)
                    results.append(Candidate(label=label, kind=kind, value=value, fn=fn))
                else:
                    lut = read_cube(value)
                    label = os.path.basename(value)
                    fn = lambda img, l=lut: lut_apply(img, l)
                    results.append(Candidate(label=label, kind=kind, value=value, fn=fn))

            else:
                log.warning("Unbekannte Kandidaten-Art: %s", kind)
        except Exception as e:
            log.warning("Kandidat %s fehlgeschlagen: %s", spec, e)

    return results


# --- Grid-Erzeugung ---

def _tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    """[1,H,W,3] float32 → HxWx3 uint8 BGR."""
    arr = img[0].clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resize_cell(img: np.ndarray, size: int = _CELL_SIZE) -> np.ndarray:
    """Zelle auf feste Größe (lange Seite = size)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Beschriftung unter der Zelle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    h, w = img.shape[:2]
    label_h = th + pad * 2
    canvas = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    canvas[:h, :] = img
    canvas[h:, :] = 255
    x = (w - tw) // 2
    y = h + pad + th
    cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def _crop_region(img: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """1:1-Ausschnitt (x, y, w, h)."""
    x, y, cw, ch = crop
    h, w = img.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    cw = min(cw, w - x)
    ch = min(ch, h - y)
    return img[y:y + ch, x:x + cw].copy()


def _make_cell(
    img: torch.Tensor,
    label: str,
    truth: torch.Tensor | None = None,
    original: torch.Tensor | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Eine Zelle erzeugen: Bild + optional Crop + Label."""
    bgr = _tensor_to_bgr(img)
    bgr = _resize_cell(bgr)

    # Kennzahlen
    metrics_parts = []
    if truth is not None:
        de_mean, _ = delta_e00(img, truth)
        metrics_parts.append(f"ΔE {de_mean:.2f}")
    if original is not None:
        s = structure(original, img)
        b = banding(original, img)
        metrics_parts.append(f"S {s:.3f}")
        metrics_parts.append(f"B {b:.3f}")

    label_text = label
    if metrics_parts:
        label_text += f" | {' '.join(metrics_parts)}"

    bgr = _add_label(bgr, label_text)

    # Crop-Zeile
    if crop is not None:
        crop_img = _crop_region(_tensor_to_bgr(img), crop)
        crop_img = _resize_cell(crop_img)
        crop_img = _add_label(crop_img, "1:1")
        # Breite angleichen
        if crop_img.shape[1] != bgr.shape[1]:
            crop_img = cv2.resize(crop_img, (bgr.shape[1], int(crop_img.shape[0] * bgr.shape[1] / crop_img.shape[1])))
        bgr = np.vstack([bgr, crop_img])

    return bgr


def make_case_grid(
    case: Case,
    candidates: list[Candidate],
    blind: bool = False,
    seed: int = _BLIND_SEED,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Raster für einen Fall: Original | Vorlage | (Wahrheit) | Kandidaten.

    Returns:
        (grid_image, column_labels)
    """
    cells: list[np.ndarray] = []
    labels: list[str] = []

    # Original
    cells.append(_make_cell(case.image, "Original", crop=crop))
    labels.append("Original")

    # Vorlage
    if case.reference is not None:
        cells.append(_make_cell(case.reference, "Vorlage", crop=crop))
        labels.append("Vorlage")

    # Wahrheit (nur wenn vorhanden)
    if case.truth is not None:
        cells.append(_make_cell(case.truth, "Wahrheit", crop=crop))
        labels.append("Wahrheit")

    # Kandidaten
    cand_cells = []
    cand_labels = []
    for cand in candidates:
        out = cand.fn(case.image)
        out = out.clamp(0.0, 1.0)
        if case.apply_mask is not None:
            out = case.apply_mask * out + (1.0 - case.apply_mask) * case.image
        cell = _make_cell(
            out, cand.label,
            truth=case.truth,
            original=case.image,
            crop=crop,
        )
        cand_cells.append(cell)
        cand_labels.append(cand.label)

    # Blind: mischen
    if blind:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(cand_cells))
        cand_cells = [cand_cells[i] for i in indices]
        cand_labels = [cand_labels[i] for i in indices]
        # Buchstaben
        letters = [chr(ord('A') + i) for i in range(len(cand_cells))]
        # Beschriftungen ersetzen: nur Buchstabe, keine Kennzahlen
        new_cells = []
        for i, c in enumerate(cand_cells):
            # Neu beschriften (Label-Zeile ist die unterste)
            # Wir bauen die Zelle neu ohne Kennzahlen
            # (einfacher: wir erstellen die Zelle neu)
            pass
        # Für Blind: Zellen ohne Kennzahlen neu erzeugen
        cand_cells = []
        for i, orig_idx in enumerate(indices):
            cand = candidates[orig_idx]
            out = cand.fn(case.image)
            out = out.clamp(0.0, 1.0)
            if case.apply_mask is not None:
                out = case.apply_mask * out + (1.0 - case.apply_mask) * case.image
            cell = _make_cell(out, chr(ord('A') + i), crop=crop)
            cand_cells.append(cell)
        cand_labels = [chr(ord('A') + i) for i in range(len(cand_cells))]

    cells.extend(cand_cells)
    labels.extend(cand_labels)

    # Spalten begrenzen: max 8 pro Seite
    # (wenn mehr, wird das Raster in mehrere Seiten aufgeteilt — hier nur die erste)
    if len(cells) > _MAX_COLS:
        cells = cells[:_MAX_COLS]
        labels = labels[:_MAX_COLS]

    # Breite angleichen
    max_h = max(c.shape[0] for c in cells)
    max_w = max(c.shape[1] for c in cells)
    padded = []
    for c in cells:
        if c.shape[0] < max_h or c.shape[1] < max_w:
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            canvas[:c.shape[0], :c.shape[1]] = c
            c = canvas
        padded.append(c)

    grid = np.hstack(padded)
    return grid, labels


def make_overall_grid(
    case_grids: list[np.ndarray],
    case_names: list[str],
) -> np.ndarray:
    """Gesamtraster: Zeilen = Fälle, Spalten = Kandidaten."""
    # Alle Raster auf gleiche Breite bringen
    max_w = max(g.shape[1] for g in case_grids)
    padded = []
    for g in case_grids:
        if g.shape[1] < max_w:
            canvas = np.full((g.shape[0], max_w, 3), 255, dtype=np.uint8)
            canvas[:, :g.shape[1]] = g
            g = canvas
        padded.append(g)
    return np.vstack(padded)


# --- HTML-Bericht ---

def _tensor_to_jpeg_bytes(img: torch.Tensor, size: int = _CELL_SIZE, quality: int = _JPEG_QUALITY) -> bytes:
    """Tensor → JPEG-Bytes (für Base64 im HTML)."""
    import base64
    bgr = _tensor_to_bgr(img)
    bgr = _resize_cell(bgr, size)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _generate_html(
    cases: list[Case],
    candidates: list[dict[str, str]],
    case_results: list[dict[str, Any]],
    blind: bool,
    seed: int,
    out_dir: Path,
) -> None:
    """Statischer HTML-Bericht schreiben."""
    import base64

    # Bilder als Base64
    img_data: dict[str, str] = {}
    for cr in case_results:
        for key in cr["cells"]:
            img = cr["cells"][key]
            img_data[f"{cr['case']}_{key}"] = _tensor_to_jpeg_bytes(img)

    # HTML
    html_parts: list[str] = []
    html_parts.append("""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>RefGrade Shootout</title>
<style>
body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }
h1 { font-size: 1.4em; }
.case-section { margin-bottom: 30px; }
.case-title { font-size: 1.1em; margin-bottom: 8px; color: #aaa; }
.grid { display: flex; flex-wrap: wrap; gap: 4px; }
.cell { text-align: center; }
.cell img { max-width: 256px; cursor: pointer; border: 1px solid #333; }
.cell .label { font-size: 0.75em; color: #ccc; margin-top: 2px; }
.cell img:hover { border-color: #666; }
#fullscreen { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.95); z-index: 1000; justify-content: center; align-items: center; }
#fullscreen img { max-width: 95vw; max-height: 95vh; }
#fullscreen.active { display: flex; }
table { border-collapse: collapse; width: 100%; margin-top: 20px; }
th, td { border: 1px solid #444; padding: 6px 10px; text-align: right; font-size: 0.85em; }
th { background: #2a2a2a; cursor: pointer; }
th:hover { background: #3a3a3a; }
td:first-child, th:first-child { text-align: left; }
.contradiction { color: #f44; font-weight: bold; }
</style>
</head>
<body>
<h1>RefGrade Shootout</h1>
<p>Seed: """ + str(seed) + """ | Blind: """ + ("ja" if blind else "nein") + """</p>
""")

    # Pro Fall: Raster
    for cr in case_results:
        html_parts.append(f'<div class="case-section">')
        html_parts.append(f'<div class="case-title">{cr["case"]}</div>')
        html_parts.append('<div class="grid">')
        for key, label in cr["cell_labels"]:
            b64 = img_data.get(f'{cr["case"]}_{key}', "")
            html_parts.append(
                f'<div class="cell">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'onclick="showFull(this)" '
                f'data-full="data:image/jpeg;base64,{b64}">'
                f'<div class="label">{label}</div>'
                f'</div>'
            )
        html_parts.append('</div></div>')

    # Kennzahlen-Tabelle
    html_parts.append("""
<h2>Kennzahlen</h2>
<table id="metrics">
<thead><tr>
<th onclick="sortTable(0)">Fall</th>
<th onclick="sortTable(1)">Kandidat</th>
<th onclick="sortTable(2)">ΔE00</th>
<th onclick="sortTable(3)">Structure</th>
<th onclick="sortTable(4)">Banding</th>
</tr></thead>
<tbody>
""")
    for cr in case_results:
        for row in cr["metrics"]:
            cls = ' class="contradiction"' if row.get("contradiction") else ""
            html_parts.append(
                f'<tr{cls}>'
                f'<td>{cr["case"]}</td>'
                f'<td>{row["candidate"]}</td>'
                f'<td>{row["delta_e00"]:.3f}</td>'
                f'<td>{row["structure"]:.4f}</td>'
                f'<td>{row["banding"]:.4f}</td>'
                f'</tr>'
            )
    html_parts.append("</tbody></table>")

    # Fullscreen + Sortier-JS
    html_parts.append("""
<div id="fullscreen" onclick="this.classList.remove('active')">
<img id="fullscreen-img" src="">
</div>
<script>
function showFull(el) {
  var fs = document.getElementById('fullscreen');
  document.getElementById('fullscreen-img').src = el.dataset.full;
  fs.classList.add('active');
}
function sortTable(col) {
  var tbl = document.getElementById('metrics');
  var rows = Array.from(tbl.tBodies[0].rows);
  var asc = tbl.dataset.sortCol == col && tbl.dataset.sortDir == '1';
  tbl.dataset.sortCol = col;
  tbl.dataset.sortDir = asc ? '0' : '1';
  rows.sort(function(a, b) {
    var av = a.cells[col].textContent.trim();
    var bv = b.cells[col].textContent.trim();
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  rows.forEach(function(r) { tbl.tBodies[0].appendChild(r); });
}
</script>
</body>
</html>""")

    (out_dir / "index.html").write_text("\n".join(html_parts), encoding="utf-8")


# --- Hauptlogik ---

def run_shootout(
    cases_dir: str,
    candidates_spec: str,
    blind: bool = False,
    crop: tuple[int, int, int, int] | None = None,
    seed: int = _BLIND_SEED,
) -> Path:
    """Shootout ausführen, Raster + Bericht schreiben. Returns: Output-Verzeichnis."""
    device = _get_device()
    log.info("Gerät: %s", device)

    # Fälle laden
    cases = load_cases(cases_dir, device)
    if not cases:
        log.error("Keine Fälle gefunden.")
        sys.exit(1)
    log.info("Fälle: %d", len(cases))

    # Kandidaten parsen
    candidates_spec_list = parse_candidates(candidates_spec)
    log.info("Kandidaten: %s", [c["spec"] for c in candidates_spec_list])

    # Output-Verzeichnis
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(_SHOOTOUT_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(exist_ok=True)

    # Pro Fall: Raster erzeugen
    case_results: list[dict[str, Any]] = []
    case_grids: list[np.ndarray] = []
    case_names: list[str] = []
    key_entries: dict[str, list[str]] = {}

    for case in cases:
        log.info("Fall: %s", case.name)
        fitted = fit_candidates(candidates_spec_list, case, device)
        if not fitted:
            log.warning("Keine Kandidaten für %s", case.name)
            continue

        grid, labels = make_case_grid(case, fitted, blind=blind, seed=seed, crop=crop)

        # Raster speichern
        grid_path = out_dir / "cases" / f"{case.name}.jpg"
        cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

        # Einzelbilder für HTML
        cells: dict[str, torch.Tensor] = {}
        cell_labels: list[tuple[str, str]] = []
        metrics_rows: list[dict[str, Any]] = []

        # Original
        cells["orig"] = case.image
        cell_labels.append(("orig", "Original"))
        if case.reference is not None:
            cells["ref"] = case.reference
            cell_labels.append(("ref", "Vorlage"))
        if case.truth is not None:
            cells["truth"] = case.truth
            cell_labels.append(("truth", "Wahrheit"))

        # Kandidaten
        for i, cand in enumerate(fitted):
            out_img = cand.fn(case.image).clamp(0.0, 1.0)
            if case.apply_mask is not None:
                out_img = case.apply_mask * out_img + (1.0 - case.apply_mask) * case.image

            key = f"cand_{i}"
            cells[key] = out_img

            if blind:
                letter = chr(ord('A') + i)
                cell_labels.append((key, letter))
            else:
                cell_labels.append((key, cand.label))

            # Kennzahlen
            row: dict[str, Any] = {"candidate": cand.label}
            if case.truth is not None:
                de_mean, _ = delta_e00(out_img, case.truth)
                row["delta_e00"] = de_mean
            else:
                row["delta_e00"] = 0.0
            row["structure"] = structure(case.image, out_img)
            row["banding"] = banding(case.image, out_img)
            metrics_rows.append(row)

        case_results.append({
            "case": case.name,
            "cells": cells,
            "cell_labels": cell_labels,
            "metrics": metrics_rows,
        })
        case_grids.append(grid)
        case_names.append(case.name)

        # Blind-Key: gemischte Reihenfolge (Index 0 = A, 1 = B, ...)
        if blind:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(fitted))
            key_entries[case.name] = [fitted[i].label for i in perm]

    # Gesamtraster
    if case_grids:
        overall = make_overall_grid(case_grids, case_names)
        cv2.imwrite(str(out_dir / "overall.jpg"), overall, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

    # results.json
    results_json: dict[str, Any] = {
        "timestamp": timestamp,
        "device": str(device),
        "blind": blind,
        "seed": seed,
        "cases": [],
    }
    for cr in case_results:
        case_entry: dict[str, Any] = {"name": cr["case"], "metrics": []}
        for row in cr["metrics"]:
            case_entry["metrics"].append({
                "candidate": row["candidate"],
                "delta_e00": round(row["delta_e00"], 4),
                "structure": round(row["structure"], 4),
                "banding": round(row["banding"], 4),
            })
        results_json["cases"].append(case_entry)

    (out_dir / "results.json").write_text(
        json.dumps(results_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Blind-Key
    if blind:
        (out_dir / "key.json").write_text(
            json.dumps(key_entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # HTML-Bericht
    _generate_html(cases, candidates_spec_list, case_results, blind, seed, out_dir)

    log.info("Fertig: %s", out_dir)
    return out_dir


def run_score(score_dir: str) -> Path:
    """Stimmen zählen, LEADERBOARD.md schreiben."""
    score_path = Path(score_dir)
    votes_path = score_path / "votes.csv"
    if not votes_path.exists():
        log.error("votes.csv nicht gefunden in %s", score_dir)
        sys.exit(1)

    # votes.csv lesen: fall,buchstabe,rang
    votes: list[dict[str, str]] = []
    with open(votes_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            votes.append(row)

    # key.json finden
    key_path = score_path / "key.json"
    if not key_path.exists():
        log.error("key.json nicht gefunden in %s")
        sys.exit(1)
    key = json.loads(key_path.read_text(encoding="utf-8"))

    # results.json für Kennzahlen
    results_path = score_path / "results.json"
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    # Stimmen zählen: Pro Kandidat → Summe der Ränge (niedriger = besser)
    candidate_scores: dict[str, list[int]] = {}
    for v in votes:
        case_name = v["fall"].strip()
        letter = v["buchstabe"].strip().upper()
        rank = int(v["rang"].strip())

        # Buchstabe → Kandidaten-Name
        if case_name in key:
            idx = ord(letter) - ord('A')
            if idx < len(key[case_name]):
                cand_name = key[case_name][idx]
                if cand_name not in candidate_scores:
                    candidate_scores[cand_name] = []
                candidate_scores[cand_name].append(rank)

    # Mittelrang pro Kandidat
    avg_ranks: dict[str, float] = {}
    for name, ranks in candidate_scores.items():
        avg_ranks[name] = sum(ranks) / len(ranks)

    # Sortieren nach Mittelrang
    ranked = sorted(avg_ranks.items(), key=lambda x: x[1])

    # Kennzahlen aus results.json
    metrics_by_cand: dict[str, dict[str, float]] = {}
    for case_data in results.get("cases", []):
        for m in case_data.get("metrics", []):
            cand = m["candidate"]
            if cand not in metrics_by_cand:
                metrics_by_cand[cand] = {"delta_e00": [], "structure": [], "banding": []}
            metrics_by_cand[cand]["delta_e00"].append(m["delta_e00"])
            metrics_by_cand[cand]["structure"].append(m["structure"])
            metrics_by_cand[cand]["banding"].append(m["banding"])

    # Mittelwerte
    metrics_avg: dict[str, dict[str, float]] = {}
    for cand, vals in metrics_by_cand.items():
        metrics_avg[cand] = {
            "delta_e00": sum(vals["delta_e00"]) / len(vals["delta_e00"]) if vals["delta_e00"] else 0,
            "structure": sum(vals["structure"]) / len(vals["structure"]) if vals["structure"] else 0,
            "banding": sum(vals["banding"]) / len(vals["banding"]) if vals["banding"] else 0,
        }

    # Widersprüche: gewinnt Stimmen (Top 2), aber schlechte Kennzahlen (ΔE00 > Median oder Structure < Median)
    de_values = [m["delta_e00"] for m in metrics_avg.values()]
    struct_values = [m["structure"] for m in metrics_avg.values()]
    de_median = float(np.median(de_values)) if de_values else 0
    struct_median = float(np.median(struct_values)) if struct_values else 0

    contradictions: set[str] = set()
    top_n = min(2, len(ranked))
    for i, (name, _) in enumerate(ranked):
        if i >= top_n:
            break
        m = metrics_avg.get(name, {})
        if m.get("delta_e00", 0) > de_median or m.get("structure", 1) < struct_median:
            contradictions.add(name)

    # LEADERBOARD.md schreiben
    lb_path = Path(_SHOOTOUT_DIR) / "LEADERBOARD.md"
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Shootout Leaderboard", "",
             f"Stimmen: {len(votes)} | Fälle: {len(set(v['fall'] for v in votes))}", ""]
    lines.append("| Rang | Kandidat | Mittelrang | ΔE00 | Structure | Banding | ⚠️ |")
    lines.append("|------|----------|-----------|------|-----------|---------|-----|")
    for i, (name, avg) in enumerate(ranked):
        m = metrics_avg.get(name, {})
        warn = "⚠️" if name in contradictions else ""
        lines.append(
            f"| {i + 1} | {name} | {avg:.2f} "
            f"| {m.get('delta_e00', 0):.3f} "
            f"| {m.get('structure', 0):.4f} "
            f"| {m.get('banding', 0):.4f} "
            f"| {warn} |"
        )

    if contradictions:
        lines.append("")
        lines.append("**Widersprüche:** " + ", ".join(sorted(contradictions)) +
                     " — gewinnen Stimmen, aber schlechte Kennzahlen.")

    lb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Leaderboard: %s", lb_path)
    return lb_path


# --- CLI ---

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Shootout: Bildraster, Bericht, Blindtest")
    parser.add_argument("--cases", type=str, help="Ordner mit Fällen")
    parser.add_argument("--candidates", type=str, help="Kandidatenliste (method:reinhard,preset:warm_light,...)")
    parser.add_argument("--blind", action="store_true", help="Blindtest-Modus")
    parser.add_argument("--crop", type=str, help="1:1-Ausschnitt: x,y,w,h")
    parser.add_argument("--seed", type=int, default=_BLIND_SEED, help="Seed für Blind-Mischung")
    parser.add_argument("--score", type=str, help="Ordner mit votes.csv → LEADERBOARD.md")
    args = parser.parse_args()

    if args.score:
        run_score(args.score)
        return

    if not args.cases or not args.candidates:
        parser.error("--cases und --candidates erforderlich (oder --score)")

    crop = None
    if args.crop:
        parts = args.crop.split(",")
        if len(parts) != 4:
            parser.error("--crop erwartet x,y,w,h")
        crop = tuple(int(p.strip()) for p in parts)

    run_shootout(
        cases_dir=args.cases,
        candidates_spec=args.candidates,
        blind=args.blind,
        crop=crop,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
