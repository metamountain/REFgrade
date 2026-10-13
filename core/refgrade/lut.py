"""LUT: build, apply, mix, read_cube, write_cube."""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-6


def _default_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def build(fn: callable, size: int = 33, title: str = "", device: torch.device | None = None) -> dict:
    """Identitätsgitter durch dieselbe Funktion wie das Bild.

    Args:
        fn: Callable [B,H,W,3] → [B,H,W,3] (sRGB, 0..1).
        size: LUT-Größe (Kantenlänge).
        title: Titel für den Cube-Export.
        device: Ziel-Gerät (Default: CUDA wenn verfügbar, sonst CPU).

    Returns:
        dict(size, table: Tensor[N,N,N,3] Index [b][g][r], title).
    """
    if device is None:
        device = _default_device()
    r_vals = torch.linspace(0.0, 1.0, size, device=device)
    g_vals = torch.linspace(0.0, 1.0, size, device=device)
    b_vals = torch.linspace(0.0, 1.0, size, device=device)

    # Gitter: [N, N, N, 3] = [B=N, H=N, W=N, C=3]
    # Pixel (h, w) in Batch b: (r_vals[w], g_vals[h], b_vals[b])
    r_3d = r_vals[None, None, :].expand(size, size, size)
    g_3d = g_vals[None, :, None].expand(size, size, size)
    b_3d = b_vals[:, None, None].expand(size, size, size)
    grid = torch.stack([r_3d, g_3d, b_3d], dim=-1)  # [N,N,N,3]

    # Durch fn schicken: [B=N, H=N, W=N, C=3] → [N,N,N,3]
    table = fn(grid).clamp(0.0, 1.0)  # [N,N,N,3] Index [b][g][r]

    return {"size": size, "table": table, "title": title}


def apply(img: torch.Tensor, lut: dict) -> torch.Tensor:
    """LUT anwenden: trilinear per grid_sample (5D, align_corners=True).

    Args:
        img: [B,H,W,3] sRGB, 0..1.
        lut: dict(size, table [N,N,N,3] Index [b][g][r], title).

    Returns:
        [B,H,W,3] sRGB, 0..1.
    """
    B, H, W, _ = img.shape
    N = lut["size"]
    table = lut["table"].to(img.device)  # [N,N,N,3] Index [b][g][r]

    # LUT zu 5D: [1, 3, N, N, N] → d=b, h=g, w=r
    lut_5d = table.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,N,N,N]
    lut_5d = lut_5d.expand(B, -1, -1, -1, -1)  # [B,3,N,N,N]

    # Grid: [B, H, W, 1, 3], Werte in [-1, 1]
    # grid_sample 3D: Grid (N, H_out, W_out, D_out, 3); Koordinaten (x,y,z) = (W,H,D) = (R,G,B)
    grid = torch.stack([
        img[..., 0] * 2.0 - 1.0,  # red → W
        img[..., 1] * 2.0 - 1.0,  # green → H
        img[..., 2] * 2.0 - 1.0,  # blue → D
    ], dim=-1)  # [B,H,W,3]
    grid = grid.unsqueeze(3)  # [B,H,W,1,3] → D_out=1, H_out=H, W_out=W

    out = F.grid_sample(
        lut_5d, grid, mode="bilinear", align_corners=True
    )  # [B,3,H,W,1]
    out = out.squeeze(-1).permute(0, 2, 3, 1)  # [B,H,W,3]
    return out


def mix(lut_a: dict, lut_b: dict, t: float) -> dict:
    """Lineare Mischung zweier LUT-Tabellen."""
    if lut_a["size"] != lut_b["size"]:
        raise ValueError(f"LUT-Größen unterscheiden sich: {lut_a['size']} vs {lut_b['size']}")
    table = lut_a["table"] * (1.0 - t) + lut_b["table"] * t
    return {"size": lut_a["size"], "table": table, "title": f"mix({lut_a['title']}, {lut_b['title']}, {t:.2f})"}


def write_cube(lut: dict, path: str) -> None:
    """Adobe Cube 1.0 schreiben. R läuft am schnellsten."""
    N = lut["size"]
    table = lut["table"]  # [N,N,N,3] Index [b][g][r]
    title = lut.get("title", "")

    lines: list[str] = []
    lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {N}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    lines.append("# RefGrade input: sRGB/Rec.709 display")
    lines.append("# R runs fastest")

    # Daten: b (slowest) → g → r (fastest)
    for bi in range(N):
        for gi in range(N):
            for ri in range(N):
                r, g, b = table[bi, gi, ri].tolist()
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_cube(path: str) -> dict:
    """Adobe Cube 1.0 lesen. Kopfzeilen in beliebiger Reihenfolge.

    Raises:
        ValueError: bei LUT_1D_SIZE oder ungültiger Datei.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".cube",):
        raise ValueError(f"Ungültige Endung: {ext} (erwartet .cube)")

    size: int | None = None
    title = ""
    data: list[tuple[float, float, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                parts = line.split('"')
                if len(parts) >= 2:
                    title = parts[1]
                continue
            if upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
                continue
            if upper.startswith("LUT_1D_SIZE"):
                raise ValueError("LUT_1D_SIZE wird nicht unterstützt")
            if upper.startswith("DOMAIN_MIN") or upper.startswith("DOMAIN_MAX"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if size is None:
        raise ValueError("LUT_3D_SIZE fehlt in der Datei")

    expected = size ** 3
    if len(data) != expected:
        raise ValueError(f"Erwartet {expected} Datenzeilen, gefunden {len(data)}")

    # table[b][g][r][c], R läuft am schnellsten
    table = torch.zeros(size, size, size, 3, dtype=torch.float32)
    idx = 0
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                table[bi, gi, ri, 0] = data[idx][0]  # R
                table[bi, gi, ri, 1] = data[idx][1]  # G
                table[bi, gi, ri, 2] = data[idx][2]  # B
                idx += 1

    return {"size": size, "table": table, "title": title}
