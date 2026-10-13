"""LUT aus Vorher/Nachher: trilineare Gitterverteilung + regularisierte kleinste Quadrate."""

from __future__ import annotations

import logging
import math
import numpy as np
import torch
import torch.nn.functional as F

from .color import luma

logger = logging.getLogger("RefGrade")

_EPS = 1e-6
_MAX_PIXELS = 2_000_000
_SEED = 0
_CONJ_GRAD_MAX_ITER = 1000
_CONJ_GRAD_TOL = 1e-10
_SHIFT_SIDE = 768


def _subsample_pairs(
    before: torch.Tensor,
    after: torch.Tensor,
    weights: torch.Tensor | None = None,
    n: int = _MAX_PIXELS,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixelpaare subsampeln: (before [n,3], after [n,3]) als numpy float64."""
    B, H, W, C = before.shape
    total = B * H * W
    b_flat = before.reshape(total, C).cpu().numpy().astype(np.float64)
    a_flat = after.reshape(total, C).cpu().numpy().astype(np.float64)

    if total <= n:
        return b_flat, a_flat

    rng = np.random.default_rng(seed)
    if weights is not None:
        w_flat = weights.reshape(total).cpu().numpy().astype(np.float64)
        w_flat = np.clip(w_flat, _EPS, None)
        probs = w_flat / w_flat.sum()
        indices = rng.choice(total, size=n, replace=False, p=probs)
    else:
        indices = rng.choice(total, size=n, replace=False)
    return b_flat[indices], a_flat[indices]


def _trilinear_weights(pos: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Trilineare Gewichte und Ecken-Indizes für Positionen [n,3] in [0,1]^3.

    Returns:
        (corners [n,8,3] int, weights [n,8] float)
    """
    s = size - 1
    scaled = pos * s
    lo = np.floor(scaled).astype(int).clip(0, s)
    frac = scaled - lo
    hi = (lo + 1).clip(0, s)

    corners = np.zeros((len(pos), 8, 3), dtype=int)
    weights = np.zeros((len(pos), 8), dtype=np.float64)

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                idx = di * 4 + dj * 2 + dk
                corners[:, idx, 0] = hi[:, 0] if di else lo[:, 0]
                corners[:, idx, 1] = hi[:, 1] if dj else lo[:, 1]
                corners[:, idx, 2] = hi[:, 2] if dk else lo[:, 2]

                w = np.ones(len(pos))
                w *= np.where(di, frac[:, 0], 1.0 - frac[:, 0])
                w *= np.where(dj, frac[:, 1], 1.0 - frac[:, 1])
                w *= np.where(dk, frac[:, 2], 1.0 - frac[:, 2])
                weights[:, idx] = w

    return corners, weights


def _identity_table(size: int) -> np.ndarray:
    """Identitäts-LUT als [size, size, size, 3] numpy (Index [r][g][b])."""
    r = np.linspace(0, 1, size)
    g = np.linspace(0, 1, size)
    b = np.linspace(0, 1, size)
    rr, gg, bb = np.meshgrid(r, g, b, indexing='ij')
    return np.stack([rr, gg, bb], axis=-1)


def _laplacian_op(table: np.ndarray, size: int) -> np.ndarray:
    """Laplace-Operator mit linearer Extrapolation an den Rändern.

    Für lineare Funktionen ist das Ergebnis überall null.
    Der Operator ist NICHT symmetrisch; für die Regularisierung
    wird LᵀL verwendet (siehe _laplacian_T_op).
    """
    out = np.zeros_like(table)
    idx = np.arange(size)
    for axis in range(3):
        for delta in (-1, 1):
            nidx_raw = idx + delta
            nidx_clipped = np.clip(nidx_raw, 0, size - 1)
            neighbor = np.take(table, nidx_clipped, axis=axis)
            if delta == -1:
                mask = nidx_raw < 0
                if mask.any():
                    t0 = np.take(table, np.zeros(size, dtype=int), axis=axis)
                    t1 = np.take(table, np.ones(size, dtype=int), axis=axis)
                    extrap = 2.0 * t0 - t1
                    sl_mask = [slice(None)] * table.ndim
                    sl_mask[axis] = mask
                    neighbor[tuple(sl_mask)] = extrap[tuple(sl_mask)]
            else:
                mask = nidx_raw >= size
                if mask.any():
                    t_last = np.take(table, np.full(size, size - 1, dtype=int), axis=axis)
                    t_prev = np.take(table, np.full(size, size - 2, dtype=int), axis=axis)
                    extrap = 2.0 * t_last - t_prev
                    sl_mask = [slice(None)] * table.ndim
                    sl_mask[axis] = mask
                    neighbor[tuple(sl_mask)] = extrap[tuple(sl_mask)]
            out += neighbor
    out -= 6.0 * table
    return out


def _laplacian_T_op(table: np.ndarray, size: int) -> np.ndarray:
    """Transponierter Laplace-Operator (Lᵀ).

    Lᵀ[i] = sum_j L[j,i] * table[j]
    Pro Achse:
      i=0:     table[1]
      i=1:     -2*table[1] + table[2]
      2<=i<N-2: table[i-1] - 2*table[i] + table[i+1]
      i=N-2:   table[N-3] - 2*table[N-2]
      i=N-1:   table[N-2]
    """
    out = np.zeros_like(table)
    for axis in range(3):
        # i=0: Lᵀ contribution = table[1]
        sl0 = [slice(None)] * table.ndim
        sl1 = [slice(None)] * table.ndim
        sl0[axis] = 0
        sl1[axis] = 1
        out[tuple(sl0)] += table[tuple(sl1)]

        # i=1: Lᵀ contribution = -2*table[1] + table[2]
        sl1a = [slice(None)] * table.ndim
        sl2 = [slice(None)] * table.ndim
        sl1a[axis] = 1
        sl2[axis] = 2
        out[tuple(sl1a)] += -2.0 * table[tuple(sl1a)] + table[tuple(sl2)]

        # Interior (2 <= i <= size-3)
        for i in range(2, size - 2):
            sl_i = [slice(None)] * table.ndim
            sl_im1 = [slice(None)] * table.ndim
            sl_ip1 = [slice(None)] * table.ndim
            sl_i[axis] = i
            sl_im1[axis] = i - 1
            sl_ip1[axis] = i + 1
            out[tuple(sl_i)] += (
                table[tuple(sl_im1)] - 2.0 * table[tuple(sl_i)] + table[tuple(sl_ip1)]
            )

        # i=size-2: Lᵀ contribution = table[size-3] - 2*table[size-2]
        sl_nm2 = [slice(None)] * table.ndim
        sl_nm3 = [slice(None)] * table.ndim
        sl_nm2[axis] = size - 2
        sl_nm3[axis] = size - 3
        out[tuple(sl_nm2)] += table[tuple(sl_nm3)] - 2.0 * table[tuple(sl_nm2)]

        # i=size-1: Lᵀ contribution = table[size-2]
        sl_last = [slice(None)] * table.ndim
        sl_prev = [slice(None)] * table.ndim
        sl_last[axis] = size - 1
        sl_prev[axis] = size - 2
        out[tuple(sl_last)] += table[tuple(sl_prev)]

    return out


def _apply_A(
    x: np.ndarray,
    flat_corners: np.ndarray,
    flat_weights: np.ndarray,
    n_pixels: int,
    size: int,
) -> np.ndarray:
    """A·x: trilineare Interpolation. x [n_cells,3] → [n_pixels,3]."""
    flat_idx = (
        flat_corners[:, 0] * size * size
        + flat_corners[:, 1] * size
        + flat_corners[:, 2]
    )
    gathered = x.reshape(-1, 3)[flat_idx] * flat_weights[:, None]
    return gathered.reshape(n_pixels, 8, 3).sum(axis=1)


def _apply_At(
    y: np.ndarray,
    flat_corners: np.ndarray,
    flat_weights: np.ndarray,
    n_cells: int,
    size: int,
) -> np.ndarray:
    """Aᵀ·y: Scatter. y [n_pixels,3] → [n_cells,3]."""
    y_8 = np.repeat(y, 8, axis=0)
    weighted = y_8 * flat_weights[:, None]
    flat_idx = (
        flat_corners[:, 0] * size * size
        + flat_corners[:, 1] * size
        + flat_corners[:, 2]
    )
    result = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(result, flat_idx, weighted)
    return result


def _conjugate_gradient(
    rhs: np.ndarray,
    apply_op,
    x0: np.ndarray,
    max_iter: int = _CONJ_GRAD_MAX_ITER,
    tol: float = _CONJ_GRAD_TOL,
    preconditioner: np.ndarray | None = None,
) -> np.ndarray:
    """Präconditionierte konjugierte Gradienten für SPD-Systeme."""
    x = x0.copy()
    r = rhs - apply_op(x)
    if preconditioner is not None:
        z = r / preconditioner
    else:
        z = r.copy()
    p = z.copy()
    rz_old = np.dot(r.ravel(), z.ravel())

    for _ in range(max_iter):
        if np.sqrt(np.dot(r.ravel(), r.ravel())) < tol:
            break
        Ap = apply_op(p)
        pAp = np.dot(p.ravel(), Ap.ravel())
        if abs(pAp) < _EPS:
            break
        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap
        if preconditioner is not None:
            z = r / preconditioner
        else:
            z = r.copy()
        rz_new = np.dot(r.ravel(), z.ravel())
        if np.sqrt(np.dot(r.ravel(), r.ravel())) < tol:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    return x


def fit_pair(
    before: torch.Tensor,
    after: torch.Tensor,
    size: int = 33,
    smoothness: float = 0.1,
    weights: torch.Tensor | None = None,
) -> dict:
    """LUT aus Vorher/Nachher-Paaren.

    Args:
        before: [B,H,W,3] sRGB 0..1.
        after: [B,H,W,3] sRGB 0..1.
        size: LUT-Größe (Kantenlänge).
        smoothness: Glättungsgewicht (Laplace).
        weights: [B,H,W] 0..1 oder None.

    Returns:
        dict mit:
            size, table (Tensor [N,N,N,3] Index [b][g][r]), title,
            residual (float), empty_fraction (float)
    """
    b_pairs, a_pairs = _subsample_pairs(before, after, weights)
    n_pixels = len(b_pairs)

    corners, tri_w = _trilinear_weights(b_pairs, size)
    flat_corners = corners.reshape(-1, 3)
    flat_weights = tri_w.reshape(-1)

    n_cells = size ** 3

    flat_idx_all = (
        flat_corners[:, 0] * size * size
        + flat_corners[:, 1] * size
        + flat_corners[:, 2]
    )

    # Belegte Zellen: mindestens ein Nicht-Null-Gewicht
    nz = flat_weights > _EPS
    occupied = np.zeros(n_cells, dtype=bool)
    if nz.any():
        occupied[np.unique(flat_idx_all[nz])] = True

    # RHS: Aᵀ · after
    rhs = _apply_At(a_pairs, flat_corners, flat_weights, n_cells, size).reshape(-1)

    # Start: Identität
    x0 = _identity_table(size).reshape(-1).copy()

    def matvec(x_flat: np.ndarray) -> np.ndarray:
        x = x_flat.reshape(n_cells, 3)
        y = _apply_A(x, flat_corners, flat_weights, n_pixels, size)
        AAt_x = _apply_At(y, flat_corners, flat_weights, n_cells, size).reshape(-1)
        x_4d = x.reshape(size, size, size, 3)
        Lx = _laplacian_op(x_4d, size)
        LTLx = _laplacian_T_op(Lx, size)
        return AAt_x + smoothness * LTLx.reshape(-1)

    # Jacobi-Präkonditionierer mit exakter LᵀL-Diagonale
    diag_AAt = np.zeros(n_cells, dtype=np.float64)
    np.add.at(diag_AAt, flat_idx_all, flat_weights ** 2)
    d1 = np.full(size, 6.0)
    d1[0] = 1.0
    d1[-1] = 1.0
    if size > 2:
        d1[1] = 5.0
        d1[-2] = 5.0
    lap_diag = (d1[:, None, None] + d1[None, :, None] + d1[None, None, :]).reshape(-1)
    prec = np.repeat(diag_AAt + smoothness * lap_diag, 3)
    prec = np.where(prec > _EPS, prec, 1.0)

    x = _conjugate_gradient(rhs, matvec, x0, preconditioner=prec)
    table = x.reshape(size, size, size, 3)
    table = np.clip(table, 0.0, 1.0)

    # Restfehler: mittlerer Pixel-Fehler
    pred = _apply_A(table, flat_corners, flat_weights, n_pixels, size)
    residual = float(np.sqrt(np.mean(np.sum((pred - a_pairs) ** 2, axis=1))))

    empty_fraction = float((~occupied).sum() / n_cells)

    if residual > 0.05:
        logger.warning(
            "fit_pair: großer Restfehler (%.4f) — kein globales Farbmapping", residual
        )

    # Tensor: [b][g][r][3] — LUT-Konvention
    table_tensor = torch.from_numpy(table.copy()).permute(2, 1, 0, 3).contiguous()

    return {
        "size": size,
        "table": table_tensor,
        "title": "fit_pair",
        "residual": residual,
        "empty_fraction": empty_fraction,
    }


def _sobel_magnitude(lum: torch.Tensor) -> torch.Tensor:
    """Sobel-Betrag eines [H,W] Luminanzbilds → [H,W]."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=lum.device)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                      device=lum.device)
    x = lum[None, None]
    gx = F.conv2d(x, kx[None, None], padding=1)
    gy = F.conv2d(x, ky[None, None], padding=1)
    return torch.sqrt(gx * gx + gy * gy + _EPS)[0, 0]


def pair_shift(before: torch.Tensor, after: torch.Tensor,
               side: int = _SHIFT_SIDE) -> float:
    """Verschiebung zwischen Vorher und Nachher per Phasenkorrelation der Kantenbilder.

    Args:
        before: [B,H,W,3] sRGB 0..1.
        after: [B,H,W,3] sRGB 0..1.

    Returns:
        Betrag der Verschiebung in Pixeln bei `side` px langer Seite.
    """
    def _prep(t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[1], t.shape[2]
        scale = side / max(h, w)
        if scale < 1.0:
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            t = F.interpolate(t.permute(0, 3, 1, 2), size=(new_h, new_w),
                              mode="bilinear", align_corners=False)
            t = t.permute(0, 2, 3, 1)
        return t

    b = _prep(before)
    a = _prep(after)
    if a.shape[1:3] != b.shape[1:3]:
        a = F.interpolate(a.permute(0, 3, 1, 2), size=b.shape[1:3],
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

    eb = _sobel_magnitude(luma(b)[0, :, :, 0])
    ea = _sobel_magnitude(luma(a)[0, :, :, 0])

    eb = eb - eb.mean()
    ea = ea - ea.mean()
    r = torch.fft.fft2(eb) * torch.conj(torch.fft.fft2(ea))
    r = r / (r.abs() + _EPS)
    corr = torch.fft.ifft2(r).real
    h, w = corr.shape
    py, px = divmod(int(torch.argmax(corr).item()), w)
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    return math.hypot(px, py)
