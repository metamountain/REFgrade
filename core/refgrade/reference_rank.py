"""Vorlagen-Bibliothek: Merkmale, Ähnlichkeit, Cache, Ranking."""

from __future__ import annotations

import json
import math
import os
import hashlib
import numpy as np
import torch
from PIL import Image

from .color import rgb_to_lab, luma
from .stats import sample
from .analysis import analyze_image

# Gewichte für die Distanzberechnung (oben im Modul, Konstante)
FEATURE_WEIGHTS: dict[str, float] = {
    "tone_median_l": 2.0,
    "tone_contrast": 1.5,
    "zone_shadows_hue": 1.0,
    "zone_shadows_strength": 1.0,
    "zone_mids_hue": 0.8,
    "zone_mids_strength": 0.8,
    "zone_highlights_hue": 1.0,
    "zone_highlights_strength": 1.0,
    "chroma_median": 1.5,
    "content_sky": 0.5,
    "content_green": 0.5,
    "content_neutral": 0.3,
    "content_skin": 0.5,
    "layout": 1.0,
}

_CACHE_DIRNAME = ".refgrade_cache"
_NUM_LUMINANCE_GRID = 8


def _load_image_tensor(path: str, max_size: int = 1024) -> torch.Tensor:
    """Bild laden, auf max_size verkleinern, als [1,H,W,3] float32 0..1."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,3]


def _content_fractions(lab: torch.Tensor) -> dict[str, float]:
    """Grobe Inhaltsverteilung über feste Lab-Bereiche."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    # Himmel-blau: L > 60, b < -10, a < 5
    sky = ((L > 60) & (b < -10) & (a < 5)).float().mean().item()
    # Grün: a in [-40, -5], L in [20, 80]
    green = ((a < -5) & (a > -40) & (L > 20) & (L < 80)).float().mean().item()
    # Neutral/Grau: Chroma < 8
    chroma = torch.sqrt(a**2 + b**2)
    neutral = (chroma < 8).float().mean().item()
    # Hauttöne: a in [10, 35], b in [10, 35], L in [40, 85]
    skin = ((a > 10) & (a < 35) & (b > 10) & (b < 35) & (L > 40) & (L < 85)).float().mean().item()

    return {
        "content_sky": round(sky, 4),
        "content_green": round(green, 4),
        "content_neutral": round(neutral, 4),
        "content_skin": round(skin, 4),
    }


def _luminance_layout(img: torch.Tensor) -> list[float]:
    """8×8-Luminanz-Layout (normalisiert 0..1)."""
    l = luma(img).squeeze(0, -1)  # [H,W]
    # Auf 8×8 herunterrechnen
    import torch.nn.functional as F
    l_small = F.interpolate(l.unsqueeze(0).unsqueeze(0), size=(8, 8),
                            mode="area").squeeze()
    # Normalisieren
    l_min = l_small.min().item()
    l_max = l_small.max().item()
    if l_max - l_min < 1e-6:
        return [0.0] * 64
    l_norm = ((l_small - l_min) / (l_max - l_min)).flatten().tolist()
    return [round(v, 3) for v in l_norm]


def _cache_path(reference_dir: str, filename: str) -> str:
    """Pfad zur Feature-Cache-Datei."""
    cache_dir = os.path.join(reference_dir, _CACHE_DIRNAME)
    return os.path.join(cache_dir, f"{filename}.json")


def _file_key(path: str) -> str:
    """Cache-Schlüssel: Pfad + mtime."""
    mtime = os.path.getmtime(path)
    return f"{path}|{mtime:.3f}"


def features(img: torch.Tensor, cache_path: str | None = None, file_key: str | None = None) -> dict:
    """Merkmale aus einem Bild berechnen (mit Cache).

    Args:
        img: [1,H,W,3] sRGB 0..1.
        cache_path: optional, Pfad für JSON-Cache.
        file_key: optional, Cache-Schlüssel (Pfad + mtime).

    Returns:
        dict mit JSON-Typen.
    """
    # Cache prüfen
    if cache_path and file_key:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("_key") == file_key:
                return {k: v for k, v in cached.items() if k != "_key"}

    # Analyse
    analysis = analyze_image(img)
    lab = rgb_to_lab(img)

    # Ton
    tone = analysis["tone"]
    # Zonen
    zones = analysis["color"]["zones"]
    # Sättigung
    sat = analysis["color"]["saturation"]
    # Inhalt
    content = _content_fractions(lab)
    # Layout
    layout = _luminance_layout(img)

    feat = {
        "tone_median_l": tone["median_l"],
        "tone_contrast": tone["contrast"],
        "zone_shadows_hue": zones["shadows"]["hue"],
        "zone_shadows_strength": zones["shadows"]["strength"],
        "zone_mids_hue": zones["mids"]["hue"],
        "zone_mids_strength": zones["mids"]["strength"],
        "zone_highlights_hue": zones["highlights"]["hue"],
        "zone_highlights_strength": zones["highlights"]["strength"],
        "chroma_median": sat["chroma_median"],
        **content,
        "layout": layout,
    }

    # Cache schreiben
    if cache_path and file_key:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out = {"_key": file_key, **feat}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f)

    return feat


def _feature_distance(f1: dict, f2: dict) -> float:
    """Gewichtete Distanz zwischen zwei Feature-Vektoren."""
    dist = 0.0
    for key, weight in FEATURE_WEIGHTS.items():
        if key == "layout":
            # L2 auf dem 8×8-Layout
            l1 = np.array(f1["layout"])
            l2 = np.array(f2["layout"])
            d = float(np.sqrt(np.mean((l1 - l2)**2)))
        else:
            v1 = f1.get(key, 0.0)
            v2 = f2.get(key, 0.0)
            # Hue: zirkuläre Distanz
            if key.endswith("_hue"):
                d = min(abs(v1 - v2), 360.0 - abs(v1 - v2)) / 180.0
            else:
                d = abs(v1 - v2)
        dist += weight * d
    return dist


def rank(
    image: torch.Tensor,
    reference_dir: str,
    k: int = 8,
) -> list[tuple[str, float]]:
    """Passendste Vorlagen aus einem Ordner vorschlagen.

    Args:
        image: [1,H,W,3] sRGB 0..1.
        reference_dir: Ordner mit Vorlagen-Bildern.
        k: Anzahl der Top-Ergebnisse.

    Returns:
        Liste von (Pfad, Score) sortiert nach absteigender Ähnlichkeit.
    """
    # Features des Bildes
    img_feat = features(image)

    # Alle Bilder im Ordner
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    files = sorted(
        f for f in os.listdir(reference_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    )

    scored: list[tuple[str, float]] = []
    for fname in files:
        fpath = os.path.join(reference_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ref_img = _load_image_tensor(fpath)
        cpath = _cache_path(reference_dir, fname)
        fkey = _file_key(fpath)
        ref_feat = features(ref_img, cache_path=cpath, file_key=fkey)
        d = _feature_distance(img_feat, ref_feat)
        # Ähnlichkeit = 1 / (1 + Distanz)
        score = 1.0 / (1.0 + d)
        scored.append((fpath, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:k]
