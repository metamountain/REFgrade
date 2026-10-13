"""Fit-Strategien für den PAIR-Bench.

Jede Strategie: fit(before: Tensor[P,3], after: Tensor[P,3], N: int) -> Tensor[N,N,N,3]
Ausgabe-Achsen: [r][g][b] (R am schnellsten). Der Bench permutiert zu [b][g][r].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

_EPS = 1e-6


# --- Splat-Kern ---

def _splat_core(before: torch.Tensor, after: torch.Tensor, N: int,
                weights: torch.Tensor | None = None,
                lam: float = 5.0, jacobi_iters: int = 300) -> torch.Tensor:
    """Trilineares Splat + Jacobi-Diffusion. Returns [N,N,N,3] in [r][g][b]."""
    x = before.double()
    y = after.double()
    if weights is not None:
        w = weights.double()
    else:
        w = torch.ones(x.shape[0], dtype=torch.float64)

    g = x * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    f = g - i0

    num = torch.zeros(N**3, 3, dtype=torch.float64)
    den = torch.zeros(N**3, dtype=torch.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                tw = (f[:, 0] if dx else 1 - f[:, 0]) * \
                     (f[:, 1] if dy else 1 - f[:, 1]) * \
                     (f[:, 2] if dz else 1 - f[:, 2])
                tw = tw * w
                idx = ((i0[:, 0] + dx) * N + (i0[:, 1] + dy)) * N + (i0[:, 2] + dz)
                num.index_add_(0, idx, tw[:, None] * y)
                den.index_add_(0, idx, tw)

    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)

    r = torch.where(den[:, None] > 0, num / den.clamp_min(1e-12)[:, None] - grid,
                    torch.zeros_like(grid))
    conf = den / (den + lam)
    r = r.reshape(N, N, N, 3)
    c = conf.reshape(N, N, N, 1)
    target = r.clone()

    for _ in range(jacobi_iters):
        p = F.pad(r.permute(3, 0, 1, 2)[None], (1,) * 6, mode="replicate")[0].permute(1, 2, 3, 0)
        nb = (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] +
              p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1] +
              p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]) / 6
        r = c * target + (1 - c) * nb

    return (grid.reshape(N, N, N, 3) + r).clamp(0, 1)


def _blur_lut(lut: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gauß-Blur auf der LUT (3D, separable Box-Approximation)."""
    if sigma < 0.5:
        return lut
    # sigma in Gitter-Einheiten → Kernel-Größe
    ksize = int(2 * 3 * sigma) | 1
    ksize = max(ksize, 3)
    # 3 Box-Blurs ≈ Gauß
    lut_5d = lut.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,N,N,N]
    for _ in range(3):
        lut_5d = F.avg_pool3d(
            F.pad(lut_5d, (ksize // 2,) * 6, mode="replicate"), ksize, 1
        )
    return lut_5d.squeeze(0).permute(1, 2, 3, 0)


# --- Least Squares ---

def _fit_ls(before: torch.Tensor, after: torch.Tensor, N: int,
            mu: float = 1e-7, iters: int = 800) -> torch.Tensor:
    """Least Squares + Glättung, vorkonditioniertes CG."""
    x = before.double()
    y = after.double()
    g = x * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    f = g - i0

    idx_list, w_list = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w_list.append((f[:, 0] if dx else 1 - f[:, 0]) *
                              (f[:, 1] if dy else 1 - f[:, 1]) *
                              (f[:, 2] if dz else 1 - f[:, 2]))
                idx_list.append(((i0[:, 0] + dx) * N + (i0[:, 1] + dy)) * N + (i0[:, 2] + dz))
    idx = torch.stack(idx_list, 1)
    w = torch.stack(w_list, 1)
    t = y - x
    P = x.shape[0]

    def Bt(v):
        o = torch.zeros(N**3, 3, dtype=torch.float64)
        for k in range(8):
            o.index_add_(0, idx[:, k], w[:, k:k+1] * v)
        return o

    def B(r):
        return sum(w[:, k:k+1] * r[idx[:, k]] for k in range(8))

    def lap(r_4d):
        out = torch.zeros_like(r_4d)
        for ax in range(3):
            d = torch.diff(r_4d, dim=ax)
            sl_lo = [slice(None)] * 4
            sl_hi = [slice(None)] * 4
            sl_lo[ax] = slice(0, N - 1)
            sl_hi[ax] = slice(1, N)
            out[tuple(sl_lo)] -= d
            out[tuple(sl_hi)] += d
        return out

    eps = 1e-9
    def A(r):
        return (Bt(B(r)) / P + mu * lap(r.reshape(N, N, N, 3)).reshape(-1, 3) + eps * r)

    b = Bt(t) / P
    r = torch.zeros_like(b)

    dg = torch.zeros(N**3, dtype=torch.float64)
    for k in range(8):
        dg.index_add_(0, idx[:, k], w[:, k] ** 2)
    Minv = 1.0 / (dg / P + 6 * mu + eps)[:, None]

    res = b - A(r)
    z = Minv * res
    p = z.clone()
    rs = (res * z).sum()

    for _ in range(iters):
        Ap = A(p)
        denom = (p * Ap).sum()
        if denom.abs() < 1e-30:
            break
        al = rs / denom
        r = r + al * p
        res = res - al * Ap
        if res.norm() < 1e-9 * b.norm():
            break
        z = Minv * res
        rn = (res * z).sum()
        p = z + rn / rs * p
        rs = rn

    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    )
    return (grid + r.reshape(N, N, N, 3)).clamp(0, 1)


# --- PCHIP ---

def _pchip_slopes(xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson PCHIP-Slopes."""
    n = len(xk)
    d = np.diff(yk) / np.diff(xk)
    m = np.zeros(n)
    m[0] = d[0]
    m[-1] = d[-1]
    for i in range(1, n - 1):
        if d[i-1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2.0 * (xk[i+1] - xk[i])
            w2 = (xk[i] - xk[i-1])
            denom = w1 / d[i] + w2 / d[i-1]
            m[i] = (w1 + w2) / denom if abs(denom) > 1e-15 else 0.0
    return m


def _pchip_eval(xk: np.ndarray, yk: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """PCHIP-Interpolation an xq."""
    m = _pchip_slopes(xk, yk)
    idx = np.searchsorted(xk, xq) - 1
    idx = np.clip(idx, 0, len(xk) - 2)
    h = xk[idx+1] - xk[idx]
    t = (xq - xk[idx]) / h
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*yk[idx] + h10*h*m[idx] + h01*yk[idx+1] + h11*h*m[idx+1]


def _fit_pchip_channel(x_vals: np.ndarray, y_vals: np.ndarray, n_knots: int = 9) -> np.ndarray:
    """Fitte 9 PCHIP-Stützstellenwerte (Binning + Monotonie)."""
    xk = np.linspace(0.0, 1.0, n_knots)
    yk = np.percentile(y_vals, np.linspace(0, 100, n_knots))
    for _ in range(3):
        bins = np.digitize(x_vals, xk) - 1
        bins = np.clip(bins, 0, n_knots - 1)
        for i in range(n_knots):
            mask = bins == i
            if mask.any():
                yk[i] = y_vals[mask].mean()
        for i in range(1, n_knots):
            if yk[i] < yk[i-1]:
                yk[i] = yk[i-1]
    return yk


def _pchip_model(x: np.ndarray, yk_list: list[np.ndarray]) -> np.ndarray:
    """Per-Kanal PCHIP anwenden. x: [P,3], yk_list: 3 arrays of 9 values."""
    xk = np.linspace(0.0, 1.0, 9)
    out = np.zeros_like(x)
    for c in range(3):
        out[:, c] = _pchip_eval(xk, yk_list[c], x[:, c])
    return out


# --- Strategien ---

def strat_splat(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Trilineares Splat + Jacobi (λ=5, 300 Schritte)."""
    return _splat_core(before, after, N, lam=5.0, jacobi_iters=300)


def strat_splat_blur07(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    lut = _splat_core(before, after, N, lam=5.0, jacobi_iters=300)
    return _blur_lut(lut, 0.7)


def strat_splat_blur10(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    lut = _splat_core(before, after, N, lam=5.0, jacobi_iters=300)
    return _blur_lut(lut, 1.0)


def strat_splat_blur15(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    lut = _splat_core(before, after, N, lam=5.0, jacobi_iters=300)
    return _blur_lut(lut, 1.5)


def strat_splat_blur25(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    lut = _splat_core(before, after, N, lam=5.0, jacobi_iters=300)
    return _blur_lut(lut, 2.5)


def strat_ls(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Least Squares + Glättung, CG (μ=1e-7)."""
    return _fit_ls(before, after, N, mu=1e-7, iters=800)


def strat_size17(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """splat_blur1.0 mit Gittergröße 17."""
    lut = _splat_core(before, after, 17, lam=5.0, jacobi_iters=200)
    return _blur_lut(lut, 1.0)


def strat_size65(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """splat_blur1.0 mit Gittergröße 65."""
    lut = _splat_core(before, after, 65, lam=5.0, jacobi_iters=200)
    return _blur_lut(lut, 1.0)


def _to_space(x: torch.Tensor, space: str) -> torch.Tensor:
    if space == "linear":
        from refgrade.color import srgb_to_linear
        return srgb_to_linear(x)
    elif space == "log":
        return torch.log2(x.clamp_min(_EPS) + _EPS) / 10.0
    return x


def _from_space(x: torch.Tensor, space: str) -> torch.Tensor:
    if space == "linear":
        from refgrade.color import linear_to_srgb
        return linear_to_srgb(x)
    elif space == "log":
        return (2.0 ** (x * 10.0) - _EPS).clamp(0.0, 1.0)
    return x


def _trilinear_sample(lut: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Trilineare Interpolation: lut [N,N,N,3], coords [P,3] in 0..1."""
    N = lut.shape[0]
    g = coords * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    f = g - i0
    out = torch.zeros(coords.shape[0], 3, dtype=lut.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) *
                     (f[:, 1] if dy else 1 - f[:, 1]) *
                     (f[:, 2] if dz else 1 - f[:, 2]))[:, None]
                out += w * lut[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    return out


def strat_space_linear(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Fit in linearem Licht. 33³ sRGB-Gitter abtasten."""
    b_lin = _to_space(before, "linear")
    a_lin = _to_space(after, "linear")
    lut_lin = _splat_core(b_lin, a_lin, N, lam=5.0, jacobi_iters=300)
    lut_lin = _blur_lut(lut_lin, 1.0)
    # sRGB-Gitter abtasten
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    grid_lin = _to_space(grid, "linear")
    out = _trilinear_sample(lut_lin, grid_lin)
    return _from_space(out, "linear").clamp(0, 1).reshape(N, N, N, 3)


def strat_space_log(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Fit in log2-Kodierung."""
    b_log = _to_space(before, "log")
    a_log = _to_space(after, "log")
    lut_log = _splat_core(b_log, a_log, N, lam=5.0, jacobi_iters=300)
    lut_log = _blur_lut(lut_log, 1.0)
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    grid_log = _to_space(grid, "log")
    out = _trilinear_sample(lut_log, grid_log)
    return _from_space(out, "log").clamp(0, 1).reshape(N, N, N, 3)


def _fit_hybrid_model(before: torch.Tensor, after: torch.Tensor) -> tuple[list, np.ndarray, np.ndarray]:
    """Fit per-Kanal PCHIP + 3×3 Matrix + Offset. Returns (yk_list, M, offset)."""
    x_np = before.numpy()
    y_np = after.numpy()

    yk = [_fit_pchip_channel(x_np[:, c], y_np[:, c]) for c in range(3)]
    pchip_out = _pchip_model(x_np, yk)
    resid = y_np - pchip_out

    A = np.hstack([x_np, np.ones((x_np.shape[0], 1))])
    M = np.zeros((3, 3))
    offset = np.zeros(3)
    for c in range(3):
        coef, _, _, _ = np.linalg.lstsq(A, resid[:, c], rcond=None)
        M[c, :] = coef[:3]
        offset[c] = coef[3]

    return yk, M, offset


def strat_hybrid(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Globales parametrisches Modell (PCHIP + Matrix) + Residual-Splat."""
    yk, M, offset = _fit_hybrid_model(before, after)

    x_np = before.numpy()
    y_np = after.numpy()
    pchip_out = _pchip_model(x_np, yk)
    model_full = pchip_out + x_np @ M.T + offset
    resid2 = y_np - model_full

    # Residual-Splat: before → before + resid2
    resid2_t = torch.from_numpy(resid2).float()
    after_approx = before + resid2_t
    lut_full = _splat_core(before, after_approx, N, lam=5.0, jacobi_iters=200)
    lut_full = _blur_lut(lut_full, 1.0)

    # Residual am Gitter
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    resid_grid = (lut_full.reshape(-1, 3) - grid).numpy()

    # Finale LUT: model_full(grid) + resid_grid
    grid_np = grid.numpy()
    pchip_grid = _pchip_model(grid_np, yk)
    model_grid = pchip_grid + grid_np @ M.T + offset
    final_grid = model_grid + resid_grid
    return torch.from_numpy(final_grid).float().reshape(N, N, N, 3).clamp(0, 1)


def strat_hybrid_decay(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Hybrid mit exponentiellem Decay des Residuals (exp(-d/3))."""
    yk, M, offset = _fit_hybrid_model(before, after)

    x_np = before.numpy()
    y_np = after.numpy()
    pchip_out = _pchip_model(x_np, yk)
    model_full = pchip_out + x_np @ M.T + offset
    resid2 = y_np - model_full

    resid2_t = torch.from_numpy(resid2).float()
    after_approx = before + resid2_t
    lut_full = _splat_core(before, after_approx, N, lam=5.0, jacobi_iters=200)

    # Residual am Gitter
    grid = torch.stack(
        torch.meshgrid(*[torch.linspace(0, 1, N, dtype=torch.float64)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    resid_grid = (lut_full.reshape(-1, 3) - grid).numpy()

    # Distanz zur nächsten besetzten Zelle (BFS, vereinfacht)
    g = before.double() * (N - 1)
    i0 = g.floor().clamp(0, N - 2).long()
    occ = np.zeros((N, N, N), dtype=np.float64)
    for i in range(before.shape[0]):
        occ[i0[i, 0].item(), i0[i, 1].item(), i0[i, 2].item()] = 1.0

    # Einfache Distanz-Approximation: iteratives Minimum
    dist = np.full((N, N, N), 100.0, dtype=np.float64)
    dist[occ > 0] = 0.0
    for _ in range(N):
        for ax in range(3):
            d = np.diff(dist, axis=ax)
            dist_lo = np.concatenate([dist.take([0], axis=ax), dist.take(range(N-1), axis=ax) + np.minimum(d, 1.0)], axis=ax)
            dist_hi = np.concatenate([dist.take(range(1, N), axis=ax) + np.minimum(-d, 1.0), dist.take([N-1], axis=ax)], axis=ax)
            dist = np.minimum(dist, np.minimum(dist_lo, dist_hi))

    decay = np.exp(-dist / 3.0)
    resid_decayed = resid_grid * decay.reshape(N, N, N, 1)

    grid_np = grid.numpy()
    pchip_grid = _pchip_model(grid_np, yk)
    model_grid = pchip_grid + grid_np @ M.T + offset
    final_grid = model_grid + resid_decayed
    return torch.from_numpy(final_grid).float().reshape(N, N, N, 3).clamp(0, 1)


def strat_density(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Unterabtastung auf 200k, gewichtet mit 1/Dichte, dann splat_blur1.0."""
    P = before.shape[0]
    if P > 200_000:
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(P, generator=gen)[:200_000]
        before = before[idx]
        after = after[idx]
        P = 200_000

    # Dichte: 10³-Histogramm
    b_np = before.numpy()
    hist = np.zeros((10, 10, 10), dtype=np.float64)
    ri = np.clip((b_np[:, 0] * 10).astype(int), 0, 9)
    gi = np.clip((b_np[:, 1] * 10).astype(int), 0, 9)
    bi = np.clip((b_np[:, 2] * 10).astype(int), 0, 9)
    np.add.at(hist, (ri, gi, bi), 1.0)

    weights = 1.0 / np.maximum(hist[ri, gi, bi], 1.0)
    lut = _splat_core(before, after, N, weights=torch.from_numpy(weights).float(),
                      lam=5.0, jacobi_iters=300)
    return _blur_lut(lut, 1.0)


def strat_mono_gray(before: torch.Tensor, after: torch.Tensor, N: int = 33) -> torch.Tensor:
    """Hybrid + Monotonie entlang der Grauachse."""
    lut = strat_hybrid(before, after, N)

    # Monotonie auf der Diagonalen (r=g=b)
    diag = torch.arange(N, dtype=torch.float64) / (N - 1)
    diag_idx = torch.arange(N)
    lum = 0.2126 * lut[diag_idx, diag_idx, diag_idx, 0] + \
          0.7152 * lut[diag_idx, diag_idx, diag_idx, 1] + \
          0.0722 * lut[diag_idx, diag_idx, diag_idx, 2]

    lum_mono = torch.cummax(lum, 0)[0]

    for i in range(N):
        if lum[i] < lum_mono[i] - 1e-6:
            scale = lum_mono[i] / lum[i].clamp_min(1e-6)
            lut[i, i, i] = lut[i, i, i] * scale

    return lut.clamp(0, 1)


# --- Registry ---

STRATEGIES: dict[str, callable] = {
    "splat": strat_splat,
    "splat_blur0.7": strat_splat_blur07,
    "splat_blur1.0": strat_splat_blur10,
    "splat_blur1.5": strat_splat_blur15,
    "splat_blur2.5": strat_splat_blur25,
    "ls": strat_ls,
    "size17": strat_size17,
    "size65": strat_size65,
    "space_linear": strat_space_linear,
    "space_log": strat_space_log,
    "hybrid": strat_hybrid,
    "hybrid_decay": strat_hybrid_decay,
    "density": strat_density,
    "mono_gray": strat_mono_gray,
}
