"""Presets prüfen: Bench-Werte (structure, banding) + Blind-Shootout vorbereiten.

Aufruf:
    python.exe tools/check_presets.py [--shootout]

Ohne --shootout: nur Kennzahlen berichten.
Mit --shootout: zusätzlich Blind-Shootout ausführen.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.presets import PRESETS
from refgrade.grade import grade
from refgrade.metrics import structure, banding
from refgrade.sliders import SLIDERS

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("RefGrade")

_PKG_DIR = Path(__file__).resolve().parent.parent
_SHOOTOUT_DIR = _PKG_DIR / "bench" / "shootout"
_CASES_DIR = _SHOOTOUT_DIR / "cases"
_POOL_DIR = Path(__file__).resolve().parents[2] / "bench" / "pool"
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _load_test_images(device: torch.device) -> dict[str, torch.Tensor]:
    """Testbilder laden: aus cases/ wenn vorhanden, sonst synthetisch aus pool."""
    imgs: dict[str, torch.Tensor] = {}

    # Versuche cases/
    if _CASES_DIR.is_dir():
        for case_dir in sorted(_CASES_DIR.iterdir()):
            if not case_dir.is_dir():
                continue
            for f in sorted(case_dir.iterdir()):
                if f.suffix.lower() in _VALID_EXTS and not f.name.startswith("reference") \
                        and not f.name.startswith("truth"):
                    from PIL import Image
                    arr = np.array(Image.open(str(f)).convert("RGB"), dtype=np.float32) / 255.0
                    imgs[case_dir.name] = torch.from_numpy(arr).unsqueeze(0).to(device)
                    break

    # Fallback: pool
    if not imgs and os.path.isdir(_POOL_DIR):
        files = sorted(
            f for f in os.listdir(_POOL_DIR)
            if os.path.splitext(f)[1].lower() in _VALID_EXTS
        )
        from PIL import Image
        for f in files[:4]:
            arr = np.array(
                Image.open(os.path.join(_POOL_DIR, f)).convert("RGB"),
                dtype=np.float32,
            ) / 255.0
            imgs[f] = torch.from_numpy(arr).unsqueeze(0).to(device)

    return imgs


def check_presets() -> dict[str, dict[str, float]]:
    """Pro Preset: structure und banding über alle Testbilder."""
    device = _get_device()
    log.info("Gerät: %s", device)

    images = _load_test_images(device)
    if not images:
        log.error("Keine Testbilder gefunden.")
        sys.exit(1)
    log.info("Testbilder: %d", len(images))

    results: dict[str, dict[str, float]] = {}
    for preset_name, vals in PRESETS.items():
        if preset_name == "none":
            continue
        struct_vals: list[float] = []
        band_vals: list[float] = []
        for img_name, img in images.items():
            out = grade(img, vals).clamp(0.0, 1.0)
            struct_vals.append(structure(img, out))
            band_vals.append(banding(img, out))
        results[preset_name] = {
            "structure_mean": round(float(np.mean(struct_vals)), 4),
            "banding_mean": round(float(np.mean(band_vals)), 4),
        }
        log.info(
            "%-24s  S=%.4f  B=%.4f",
            preset_name,
            results[preset_name]["structure_mean"],
            results[preset_name]["banding_mean"],
        )

    return results


def run_blind_shootout() -> None:
    """Blind-Shootout mit allen Presets ausführen."""
    preset_names = [n for n in PRESETS if n != "none"]
    candidates = ",".join(f"preset:{n}" for n in preset_names)

    cases_dir = str(_CASES_DIR) if _CASES_DIR.is_dir() else str(_SHOOTOUT_DIR)
    cmd = [
        sys.executable,
        str(_PKG_DIR / "tools" / "shootout.py"),
        "--cases", cases_dir,
        "--candidates", candidates,
        "--blind",
    ]
    log.info("Shootout: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Presets prüfen")
    parser.add_argument("--shootout", action="store_true",
                        help="Blind-Shootout ausführen")
    args = parser.parse_args()

    check_presets()

    if args.shootout:
        run_blind_shootout()


if __name__ == "__main__":
    main()
