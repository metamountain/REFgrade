"""Rohdaten-I/O: Krita-Pixel-Format (B,G,R,A) und Bilddateien."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps


def read_raw(path: str, width: int, height: int, depth: str) -> tuple[torch.Tensor, np.ndarray]:
    """Rohdaten lesen: B,G,R,A → RGB-Tensor [1,H,W,3] float32 0..1 + Alpha [H,W].

    Args:
        path: Pfad zur Rohdatendatei.
        width: Bildbreite in Pixeln.
        height: Bildhöhe in Pixeln.
        depth: "U8" (1 Byte/kanal) oder "U16" (2 Bytes little-endian/kanal).

    Returns:
        (rgb [1,H,W,3] float32 CPU, alpha [H,W] im Original-dtype).

    Raises:
        ValueError: bei falscher Dateilänge.
    """
    if depth == "U8":
        dtype = np.uint8
        max_val = 255.0
        itemsize = 1
    elif depth == "U16":
        dtype = np.uint16
        max_val = 65535.0
        itemsize = 2
    else:
        raise ValueError(f"Unbekannte Tiefe: {depth!r}")

    expected = width * height * 4 * itemsize
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected:
        raise ValueError(f"Dateilänge: erwartet {expected}, gefunden {len(data)}")

    arr = np.frombuffer(data, dtype=dtype).reshape(height, width, 4)
    rgb = arr[..., [2, 1, 0]].astype(np.float32) / max_val
    alpha = arr[..., 3].copy()
    tensor = torch.from_numpy(rgb).unsqueeze(0)
    return tensor, alpha


def write_raw(path: str, rgb: torch.Tensor, alpha: np.ndarray, depth: str) -> None:
    """Rohdaten schreiben: RGB-Tensor + Alpha → B,G,R,A Bytes.

    Args:
        path: Zielpfad.
        rgb: [1,H,W,3] float32 0..1.
        alpha: [H,W] im Zielformat (uint8 oder uint16).
        depth: "U8" oder "U16".
    """
    if depth == "U8":
        dtype = np.uint8
        max_val = 255.0
    elif depth == "U16":
        dtype = np.uint16
        max_val = 65535.0
    else:
        raise ValueError(f"Unbekannte Tiefe: {depth!r}")

    rgb_np = rgb.squeeze(0).numpy()
    rgb_int = np.round(np.clip(rgb_np, 0.0, 1.0) * max_val).astype(dtype)
    alpha = alpha.astype(dtype)

    h, w = rgb_int.shape[:2]
    out = np.empty((h, w, 4), dtype=dtype)
    out[..., 0] = rgb_int[..., 2]
    out[..., 1] = rgb_int[..., 1]
    out[..., 2] = rgb_int[..., 0]
    out[..., 3] = alpha

    with open(path, "wb") as f:
        f.write(out.tobytes())


def read_mask(path: str, width: int, height: int) -> torch.Tensor:
    """Maske lesen: 1 Byte/Pixel → [H,W] float32 0..1.

    Args:
        path: Pfad zur Maskendatei.
        width: Breite in Pixeln.
        height: Höhe in Pixeln.

    Returns:
        [H,W] float32 0..1.

    Raises:
        ValueError: bei falscher Dateilänge.
    """
    expected = width * height
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected:
        raise ValueError(f"Maskenlänge: erwartet {expected}, gefunden {len(data)}")
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
    return torch.from_numpy(arr.astype(np.float32) / 255.0)


def load_image(path: str, max_side: int) -> torch.Tensor:
    """Bilddatei laden: EXIF-Rotation, RGB, LANCZOS ≤ max_side (nie vergrößern).

    Args:
        path: Pfad zur Bilddatei.
        max_side: Maximale längste Seite in Pixeln.

    Returns:
        [1,H,W,3] float32 0..1.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def save_png(rgb: torch.Tensor, path: str) -> None:
    """PNG speichern: 8 Bit, RGB.

    Args:
        rgb: [1,H,W,3] float32 0..1.
        path: Zielpfad.
    """
    arr = np.clip(rgb.squeeze(0).numpy(), 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    # compress_level 1: a temp file read once by the plugin; level 6 (default) cost ~100 ms
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", compress_level=1)
