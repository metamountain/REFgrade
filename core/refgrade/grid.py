"""Labeled Grid: Bild-Batch mit Beschriftungen zu einem Raster."""

from __future__ import annotations

import cv2
import numpy as np
import torch

_CELL_SIZE = 512
_LABEL_PAD = 4
_LABEL_SCALE = 0.5
_LABEL_THICKNESS = 1


def _tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    """[H,W,3] float32 → HxWx3 uint8 BGR."""
    arr = img.clamp(0.0, 1.0).cpu().numpy()
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
    """Beschriftung unter der Zelle (schwarzer Text auf Weiß)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, _LABEL_SCALE, _LABEL_THICKNESS)
    h, w = img.shape[:2]
    label_h = th + _LABEL_PAD * 2
    canvas = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    canvas[:h, :] = img
    canvas[h:, :] = 255
    x = (w - tw) // 2
    y = h + _LABEL_PAD + th
    cv2.putText(canvas, text, (x, y), font, _LABEL_SCALE, (0, 0, 0), _LABEL_THICKNESS, cv2.LINE_AA)
    return canvas


def labeled_grid(images: torch.Tensor, labels: list[str], columns: int) -> torch.Tensor:
    """Bild-Batch mit Labels zu einem Raster.

    Args:
        images: [B,H,W,3] float32, 0..1.
        labels: eine Beschriftung pro Bild (len == B).
        columns: Spaltenanzahl.

    Returns:
        [1,H',W',3] float32, 0..1.
    """
    B = images.shape[0]
    if len(labels) != B:
        raise ValueError(f"Erwartet {B} Labels, bekommen {len(labels)}")
    if columns < 1:
        columns = 1

    cells: list[np.ndarray] = []
    for i in range(B):
        bgr = _tensor_to_bgr(images[i])
        bgr = _resize_cell(bgr)
        bgr = _add_label(bgr, labels[i])
        cells.append(bgr)

    # Zeilen aufbauen
    rows: list[np.ndarray] = []
    for r in range(0, B, columns):
        row_cells = cells[r:r + columns]
        max_h = max(c.shape[0] for c in row_cells)
        max_w = max(c.shape[1] for c in row_cells)
        padded = []
        for c in row_cells:
            if c.shape[0] < max_h or c.shape[1] < max_w:
                canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
                canvas[:c.shape[0], :c.shape[1]] = c
                c = canvas
            padded.append(c)
        rows.append(np.hstack(padded))

    grid_bgr = np.vstack(rows)
    grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
    grid_tensor = torch.from_numpy(grid_rgb).float().unsqueeze(0) / 255.0
    return grid_tensor
