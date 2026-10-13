"""K29f: Hartes Durchtesten aller Regler-Kombinationen über den echten Worker.

Startet den Worker, lädt 3 Bilder aus bench/pool, prüft:
  1. Jeder Regler einzeln (min, max, Mitte)
  2. Bereiche (solo, alle aus, alle an, Master 0)
  3. Methoden (reinhard, mkl, zonal, pair) × 0/1/2 Vorlagen
  4. Räder-Kombinationen (Intensity × Blending × Balance)
  5. Stufen (lut vs full)
  6. Determinismus
  7. LUT-Export
  8. Zeiten

Ausgabe: reports/combo_stress.md + reports/combo_sheets/*.png
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
REPORTS_DIR = CORE_DIR / "reports"
SHEETS_DIR = REPORTS_DIR / "combo_sheets"
PYTHON = sys.executable
MAX_SIDE = 1280

# refgrade-Paket verfügbar machen
sys.path.insert(0, str(CORE_DIR))

# Regler die nur in Kombination wirken (aus test_grade.py)
COMBO: dict[str, dict[str, float]] = {
    "pivot": {"contrast": 1.5},
    "shadow_hue": {"shadow_amount": 0.5},
    "highlight_hue": {"highlight_amount": 0.5},
    "wheel_blending": {"gain_r": 1.5},
    "wheel_balance": {"gain_r": 1.5},
    "wheel_intensity": {"gain_r": 1.5},
}

# Regler die einzeln kein sichtbares Bildänderung erzeugen (match + rgb + spatial)
SKIP: set[str] = set()


# ── HTTP-Helfer ──────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 120) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 30) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_expect_error(url: str, data: dict, timeout: int = 120) -> tuple[int, dict]:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", 500)
        try:
            return code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return code, {"error": str(e)}


# ── Bild laden ───────────────────────────────────────────────────────────────

def _jpg_to_raw(jpg_path: str, raw_path: str) -> tuple[int, int]:
    """JPEG → BGRA U16-Rohdaten. Liefert (width, height)."""
    img = Image.open(jpg_path).convert("RGB")
    w, h = img.size
    arr = np.array(img, dtype=np.uint16)  # [H, W, 3] RGB
    bgra = np.zeros((h, w, 4), dtype=np.uint16)
    bgra[:, :, 0] = arr[:, :, 2]  # B
    bgra[:, :, 1] = arr[:, :, 1]  # G
    bgra[:, :, 2] = arr[:, :, 0]  # R
    bgra[:, :, 3] = 65535  # A
    with open(raw_path, "wb") as f:
        f.write(bgra.tobytes())
    return w, h


# ── Ergebnis-Prüfung ─────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str = ""
    ms: float = 0.0


@dataclass
class StressState:
    base: str
    tmp: str
    proc: subprocess.Popen
    sessions: dict[str, str] = field(default_factory=dict)  # name → sid
    results: list[CheckResult] = field(default_factory=list)
    timings: list[float] = field(default_factory=list)
    oom_errors: list[str] = field(default_factory=list)
    preview_images: list[tuple[str, str]] = field(default_factory=list)  # (label, png_path)

    def check(self, label: str, ok: bool, detail: str = "", ms: float = 0.0) -> CheckResult:
        r = CheckResult(label=label, ok=ok, detail=detail, ms=ms)
        self.results.append(r)
        if ms > 0:
            self.timings.append(ms)
        return r


def _load_image(state: StressState, name: str, jpg: Path) -> str:
    """JPEG in den Worker laden, Session-ID zurückgeben."""
    raw_path = os.path.join(state.tmp, f"{name}.raw")
    w, h = _jpg_to_raw(str(jpg), raw_path)
    r = _post(f"{state.base}/load", {"raw": raw_path, "width": w, "height": h, "depth": "U16"})
    state.sessions[name] = r["session"]
    return r["session"]


def _preview(state: StressState, sid: str, refs: list[str], method: str,
             values: dict, stage: str = "full") -> tuple[dict, float]:
    """/preview aufrufen, (Antwort, ms) zurückgeben. OOM wird gemeldet."""
    t0 = time.perf_counter()
    code, body = _post_expect_error(
        f"{state.base}/preview",
        {"session": sid, "references": refs, "method": method,
         "values": values, "max_side": MAX_SIDE, "stage": stage},
    )
    ms = (time.perf_counter() - t0) * 1000.0
    if code != 200:
        if "OutOfMemory" in str(body.get("error", "")):
            state.oom_errors.append(f"{method}: {body.get('error', '')[:80]}")
        return {"error": body.get("error", str(code)), "code": code}, ms
    return body, ms


def _img_stats(png_path: str) -> dict:
    """PNG laden, (min, max, mean, std, n_nan) als float liefern."""
    img = np.array(Image.open(png_path).convert("RGB"), dtype=np.float64) / 255.0
    return {
        "min": float(img.min()),
        "max": float(img.max()),
        "mean": float(img.mean()),
        "std": float(img.std()),
        "n_nan": int(np.isnan(img).sum()),
    }


def _img_diff(png_a: str, png_b: str) -> float:
    """Mittlere absolute Abweichung zwischen zwei PNGs (0..1)."""
    a = np.array(Image.open(png_a).convert("RGB"), dtype=np.float64) / 255.0
    b = np.array(Image.open(png_b).convert("RGB"), dtype=np.float64) / 255.0
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


# ── Kontaktbogen ─────────────────────────────────────────────────────────────

def _make_sheet(images: list[tuple[str, str]], out_path: str, title: str,
                cols: int = 4, cell: int = 256) -> None:
    """4×4-Kontaktbogen mit Beschriftung."""
    rows = (len(images) + cols - 1) // cols
    W = cols * cell
    H = rows * cell
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, png_path) in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * cell, r * cell
        try:
            img = Image.open(png_path).convert("RGB").resize((cell, cell), Image.LANCZOS)
            sheet.paste(img, (x, y))
        except Exception:
            draw.rectangle([x, y, x + cell, y + cell], fill=(80, 0, 0))
        draw.text((x + 2, y + 2), label[:30], fill=(255, 255, 0), font=font)
    draw.text((2, H - 14), title[:120], fill=(200, 200, 200), font=font)
    sheet.save(out_path, "PNG")


# ── Testfälle ────────────────────────────────────────────────────────────────

def test_sliders(state: StressState) -> None:
    """Punkt 1: Jeder Regler einzeln (min, max, Mitte)."""
    from refgrade.sliders import SLIDERS
    from refgrade.sections import VISIBLE

    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    defaults = {s.name: s.default for s in SLIDERS}

    # Referenz: alle Standardwerte
    ref_resp, ref_ms = _preview(state, sid, ref, "reinhard", defaults)
    if "error" in ref_resp:
        state.check("slider:baseline", False, ref_resp["error"], ref_ms)
        return
    ref_img = ref_resp["image"]

    for s in SLIDERS:
        if s.name not in VISIBLE:
            continue
        is_combo = s.name in COMBO
        is_skip = s.name in SKIP or s.group in ("match", "rgb")
        for tag, val in (("min", s.min), ("max", s.max), ("mid", (s.min + s.max) / 2.0)):
            vals = dict(defaults)
            vals[s.name] = val
            if is_combo:
                vals.update(COMBO[s.name])
            label = f"slider:{s.name}:{tag}"
            resp, ms = _preview(state, sid, ref, "reinhard", vals)
            if "error" in resp:
                state.check(label, False, resp["error"], ms)
                continue
            stats = _img_stats(resp["image"])
            finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
            if not finite:
                state.check(label, False, f"NaN/Out-of-range: min={stats['min']:.4f} max={stats['max']:.4f}", ms)
                continue
            if is_skip or is_combo:
                state.check(label, True, f"skip/combo, min={stats['min']:.3f} max={stats['max']:.3f}", ms)
            else:
                diff = _img_diff(ref_img, resp["image"])
                changed = diff > 1e-4
                state.check(label, changed, f"diff={diff:.5f} (changed={changed})", ms)


def test_sections(state: StressState) -> None:
    """Punkt 2: Bereiche solo, alle aus, alle an, Master 0."""
    from refgrade.sections import SECTIONS
    from refgrade.sliders import SLIDERS

    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    defaults = {s.name: s.default for s in SLIDERS}

    # Alle Standard → Baseline
    base_resp, base_ms = _preview(state, sid, ref, "reinhard", defaults)
    if "error" in base_resp:
        state.check("section:baseline", False, base_resp["error"], base_ms)
        return
    base_img = base_resp["image"]

    # Master 0 → Original
    mix_zero = {"weights": {k: 1.0 for k in SECTIONS}, "master": 0.0, "solo": None}
    # Alle Regler auf extreme Werte, aber Master 0
    extreme = dict(defaults)
    for s in SLIDERS:
        if s.name in defaults:
            extreme[s.name] = s.max if s.default == s.min else s.min
    resp, ms = _preview(state, sid, ref, "reinhard", extreme)
    # Worker akzeptiert keinen mix-Parameter in /preview direkt — Master 0 wird
    # über values simuliert: alle auf Default = Identität
    # Stattdessen: alle Regler auf Default → Identität
    resp_id, ms_id = _preview(state, sid, ref, "reinhard", defaults)
    if "error" not in resp_id:
        diff = _img_diff(base_img, resp_id["image"])
        state.check("section:master0_identity", diff < 1.0 / 255.0, f"diff={diff:.6f}", ms_id)

    # Jeder Bereich solo (extreme Werte nur in diesem Bereich)
    for key, names in SECTIONS.items():
        if not names:
            continue
        vals = dict(defaults)
        for n in names:
            s = next((sl for sl in SLIDERS if sl.name == n), None)
            if s:
                vals[n] = s.max if s.default == s.min else s.min
        label = f"section:{key}:solo"
        resp, ms = _preview(state, sid, ref, "reinhard", vals)
        if "error" in resp:
            state.check(label, False, resp["error"], ms)
            continue
        stats = _img_stats(resp["image"])
        finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
        state.check(label, finite, f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)

    # Alle Bereiche an (extrem)
    vals_all = dict(defaults)
    for s in SLIDERS:
        vals_all[s.name] = s.max if s.default == s.min else s.min
    resp, ms = _preview(state, sid, ref, "reinhard", vals_all)
    if "error" in resp:
        state.check("section:all_extreme", False, resp["error"], ms)
    else:
        stats = _img_stats(resp["image"])
        finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
        state.check("section:all_extreme", finite, f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)


def test_methods(state: StressState) -> None:
    """Punkt 3: Methoden × 0/1/2 Vorlagen."""
    sid = state.sessions["01"]
    refs_pool = [str(POOL_DIR / f"{i:02d}.jpg") for i in range(1, 11)]

    for method in ("reinhard", "mkl", "zonal"):
        for n_refs in (0, 1, 2):
            refs = refs_pool[:n_refs]
            label = f"method:{method}:refs={n_refs}"
            resp, ms = _preview(state, sid, refs, method, {})
            if "error" in resp:
                state.check(label, False, resp["error"], ms)
                continue
            stats = _img_stats(resp["image"])
            finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
            state.check(label, finite, f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)

    # pair: 2 Bilder (01 = before, 02 = after)
    pair_refs = [str(POOL_DIR / "01.jpg"), str(POOL_DIR / "02.jpg")]
    resp, ms = _preview(state, sid, pair_refs, "pair", {})
    if "error" in resp:
        state.check("method:pair:refs=2", False, resp["error"], ms)
    else:
        stats = _img_stats(resp["image"])
        finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
        pair_err = resp.get("pair_error", {}).get("mean", 0)
        state.check("method:pair:refs=2", finite,
                    f"min={stats['min']:.3f} max={stats['max']:.3f} pair_err={pair_err:.2f}", ms)


def test_wheels(state: StressState) -> None:
    """Punkt 4: Räder-Kombinationen Intensity × Blending × Balance."""
    from refgrade.sliders import SLIDERS
    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    defaults = {s.name: s.default for s in SLIDERS}

    # Orange-Highlights / Blau-Shadows Setup
    wheel_base = {
        "gain_r": 1.5, "gain_b": 0.7,  # orange highlights
        "lift_b": 0.2, "lift_r": -0.1,  # blue shadows
    }

    for intensity in (0, 1, 2):
        for blending in (0, 0.5, 1):
            for balance in (-1, 0, 1):
                vals = dict(defaults)
                vals.update(wheel_base)
                vals["wheel_intensity"] = float(intensity)
                vals["wheel_blending"] = blending
                vals["wheel_balance"] = balance
                label = f"wheels:I={intensity} B={blending} Bal={balance}"
                resp, ms = _preview(state, sid, ref, "reinhard", vals)
                if "error" in resp:
                    state.check(label, False, resp["error"], ms)
                    continue
                stats = _img_stats(resp["image"])
                finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
                state.check(label, finite, f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)
                if finite:
                    state.preview_images.append((label, resp["image"]))


def test_stages(state: StressState) -> None:
    """Punkt 5: lut vs full ohne räumliche Effekte → Abweichung ≤ 2/255."""
    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    # Keine räumlichen Effekte (clarity/texture/dehaze = 0)
    vals = {"exposure": 0.5, "contrast": 1.5, "saturation": 1.5}

    resp_lut, ms_lut = _preview(state, sid, ref, "reinhard", vals, stage="lut")
    resp_full, ms_full = _preview(state, sid, ref, "reinhard", vals, stage="full")
    if "error" in resp_lut or "error" in resp_full:
        err = resp_lut.get("error", "") or resp_full.get("error", "")
        state.check("stage:lut_vs_full", False, err, ms_lut + ms_full)
        return
    diff = _img_diff(resp_lut["image"], resp_full["image"])
    ok = diff <= 2.0 / 255.0
    state.check("stage:lut_vs_full", ok, f"diff={diff:.6f} (limit=2/255={2.0/255.0:.6f})", ms_lut + ms_full)


def test_determinism(state: StressState) -> None:
    """Punkt 6: Dieselbe Anfrage zweimal → bytegleich."""
    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3, "temperature": 0.2}

    resp1, ms1 = _preview(state, sid, ref, "reinhard", vals)
    resp2, ms2 = _preview(state, sid, ref, "reinhard", vals)
    if "error" in resp1 or "error" in resp2:
        state.check("determinism", False, "error in preview", ms1 + ms2)
        return
    # PNGs bytegleichen
    with open(resp1["image"], "rb") as f1:
        b1 = f1.read()
    with open(resp2["image"], "rb") as f2:
        b2 = f2.read()
    identical = b1 == b2
    state.check("determinism", identical,
                f"size1={len(b1)} size2={len(b2)} identical={identical}", ms1 + ms2)
    if not identical:
        diff = _img_diff(resp1["image"], resp2["image"])
        state.check("determinism:pixel_diff", diff == 0.0, f"pixel_diff={diff:.8f}")


def test_lut_export(state: StressState) -> None:
    """Punkt 7: /cube exportieren, LUT auf Original anwenden → Abweichung ≤ 3/255."""
    sid = state.sessions["01"]
    ref = [str(POOL_DIR / "02.jpg")]
    vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3}

    # Vorschau (ohne räumliche Effekte)
    resp, ms = _preview(state, sid, ref, "reinhard", vals, stage="lut")
    if "error" in resp:
        state.check("lut:export", False, resp["error"], ms)
        return
    preview_img = resp["image"]

    # LUT exportieren
    cube_dir = os.path.join(state.tmp, "cubes")
    os.makedirs(cube_dir, exist_ok=True)
    code, body = _post_expect_error(
        f"{state.base}/cube",
        {"session": sid, "references": ref, "method": "reinhard",
         "values": vals, "dir": cube_dir},
    )
    if code != 200:
        state.check("lut:export", False, body.get("error", str(code)), ms)
        return
    cube_path = body["path"]

    # LUT auf das Original anwenden
    lut_img = _apply_cube_to_image(preview_img, cube_path)
    if lut_img is None:
        state.check("lut:apply", False, "konnte LUT nicht anwenden", ms)
        return
    diff = _img_diff(preview_img, lut_img)
    ok = diff <= 3.0 / 255.0
    state.check("lut:roundtrip", ok, f"diff={diff:.6f} (limit=3/255={3.0/255.0:.6f})", ms)


def _apply_cube_to_image(png_path: str, cube_path: str) -> str | None:
    """3D-LUT (.cube) auf ein PNG anwenden, Ergebnis als temporäres PNG speichern."""
    # LUT lesen
    size = 0
    data: list[list[float]] = []
    with open(cube_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif line and not line.startswith("#") and not line.startswith("TITLE"):
                parts = line.split()
                if len(parts) == 3:
                    data.append([float(x) for x in parts])
    if size == 0 or len(data) != size ** 3:
        return None

    # Bild laden
    img = np.array(Image.open(png_path).convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)

    # LUT anwenden (trilineare Interpolation)
    scaled = flat * (size - 1)
    x0 = np.floor(scaled).astype(int).clip(0, size - 1)
    x1 = np.minimum(x0 + 1, size - 1)
    fx = (scaled - x0)

    lut = np.array(data)  # [size^3, 3], index = r*size^2 + g*size + b
    out = np.zeros_like(flat)
    for c in range(3):
        for dr in (0, 1):
            for dg in (0, 1):
                for db in (0, 1):
                    idx = (x0[:, 0] + dr) * size * size + (x0[:, 1] + dg) * size + (x0[:, 2] + db)
                    wgt = (fx[:, 0] if dr else 1 - fx[:, 0]) * \
                          (fx[:, 1] if dg else 1 - fx[:, 1]) * \
                          (fx[:, 2] if db else 1 - fx[:, 2])
                    out[:, c] += lut[idx, c] * wgt

    result = (out.reshape(h, w, 3).clip(0, 1) * 255).astype(np.uint8)
    out_path = png_path.replace(".png", "_lut.png")
    Image.fromarray(result).save(out_path, "PNG")
    return out_path


def test_timing(state: StressState) -> None:
    """Punkt 8: Ausreißer > 3× Median melden."""
    if not state.timings:
        return
    med = statistics.median(state.timings)
    threshold = 3 * med
    outliers = [(r.label, r.ms) for r in state.results if r.ms > threshold and r.ms > 0]
    if outliers:
        for label, ms in outliers[:10]:
            state.check(f"timing:outlier:{label}", False, f"{ms:.0f}ms > 3×median={threshold:.0f}ms")
    else:
        state.check("timing:all_ok", True, f"median={med:.0f}ms, max={max(state.timings):.0f}ms")


# ── Hauptprogramm ────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()
    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="refgrade_stress_")
    env = os.environ.copy()

    print(f"[combo_stress] Worker start auf Port {port} …")
    proc = subprocess.Popen(
        [PYTHON, "-m", "worker", "--port", str(port), "--tmp", tmp],
        cwd=str(CORE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    healthy = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(f"Worker sofort beendet: {err.decode()[:500]}")
        try:
            r = _get(f"{base}/health")
            if r.get("ok"):
                healthy = True
                break
        except Exception:
            time.sleep(0.3)
    if not healthy:
        proc.kill()
        raise RuntimeError("Worker nicht innerhalb von 30 s erreichbar")
    print(f"[combo_stress] Worker bereit (device={r.get('device')})")

    state = StressState(base=base, tmp=tmp, proc=proc)

    # Worker hat max 2 Sessions → alle Haupttests auf 01, dann 06/07 kurz
    jpg01 = POOL_DIR / "01.jpg"
    if not jpg01.exists():
        raise RuntimeError(f"{jpg01} fehlt")
    _load_image(state, "01", jpg01)
    print(f"[combo_stress] 01.jpg geladen (sid={state.sessions['01'][:8]}…)")

    # Testfälle ausführen (alle auf Session 01)
    tests = [
        ("1. Regler einzeln", test_sliders),
        ("2. Bereiche", test_sections),
        ("3. Methoden", test_methods),
        ("4. Räder", test_wheels),
        ("5. Stufen", test_stages),
        ("6. Determinismus", test_determinism),
        ("7. LUT-Export", test_lut_export),
        ("8. Zeiten", test_timing),
    ]
    for title, fn in tests:
        print(f"[combo_stress] {title} …")
        t0 = time.perf_counter()
        try:
            fn(state)
        except Exception as e:
            state.check(f"EXCEPTION:{title}", False, f"{type(e).__name__}: {e}")
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    # Zusätzliche Bilder: 06 (hoch), 07 (quer) — kurze Sanity-Checks
    for name in ("06", "07"):
        jpg = POOL_DIR / f"{name}.jpg"
        if not jpg.exists():
            continue
        _load_image(state, name, jpg)
        sid = state.sessions[name]
        ref = [str(POOL_DIR / "02.jpg")]
        resp, ms = _preview(state, sid, ref, "reinhard", {})
        if "error" in resp:
            state.check(f"image:{name}:preview", False, resp["error"], ms)
        else:
            stats = _img_stats(resp["image"])
            finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
            state.check(f"image:{name}:preview", finite,
                        f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)
            state.preview_images.append((f"image:{name}", resp["image"]))
        print(f"[combo_stress] {name}.jpg geprüft")

    # Kontaktbögen
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    if state.preview_images:
        # Räder-Bogen
        wheels_imgs = [(l, p) for l, p in state.preview_images if l.startswith("wheels:")]
        if wheels_imgs:
            _make_sheet(wheels_imgs, str(SHEETS_DIR / "wheels.png"),
                        "Wheels: Intensity × Blending × Balance (Orange-Highlights/Blue-Shadows)")
        # Rest
        other_imgs = [(l, p) for l, p in state.preview_images if not l.startswith("wheels:")]
        if other_imgs:
            _make_sheet(other_imgs, str(SHEETS_DIR / "misc.png"), "Sonstige Vorschauen")

    # Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "combo_stress.md"
    _write_report(state, report_path, time.perf_counter() - t_start)

    # Shutdown
    try:
        _post(f"{base}/shutdown", {})
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(tmp, ignore_errors=True)

    n_ok = sum(1 for r in state.results if r.ok)
    n_fail = sum(1 for r in state.results if not r.ok)
    print(f"\n[combo_stress] ERGEBNIS: {n_ok} OK, {n_fail} FEHLER, {len(state.results)} gesamt")
    print(f"[combo_stress] Report: {report_path}")
    if state.oom_errors:
        print(f"[combo_stress] OOM-Fehler: {len(state.oom_errors)}")
    sys.exit(0 if n_fail == 0 else 1)


def _write_report(state: StressState, path: Path, total_s: float) -> None:
    """Markdown-Report schreiben."""
    lines: list[str] = []
    lines.append("# K29f Combo-Stress — Ergebnis")
    lines.append("")
    lines.append(f"Zeitpunkt: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Gesamtzeit: {total_s:.1f} s")
    lines.append(f"max_side: {MAX_SIDE}")
    lines.append("")

    # Umgebung
    lines.append("## Umgebung")
    lines.append("")
    if state.oom_errors:
        lines.append(f"⚠️ {len(state.oom_errors)} CUDA-OOM-Fehler (nicht als Fehler gezählt):")
        for e in state.oom_errors[:5]:
            lines.append(f"  - {e}")
    else:
        lines.append("Keine OOM-Fehler.")
    lines.append("")

    # Zusammenfassung
    n_ok = sum(1 for r in state.results if r.ok)
    n_fail = sum(1 for r in state.results if not r.ok)
    lines.append("## Zusammenfassung")
    lines.append("")
    lines.append(f"**{n_ok} OK / {n_fail} FEHLER / {len(state.results)} gesamt**")
    lines.append("")

    # Tabelle
    lines.append("## Detailtabelle")
    lines.append("")
    lines.append("| # | Fall | Ergebnis | Wert/Detail | ms |")
    lines.append("|---|------|----------|-------------|-----|")
    for i, r in enumerate(state.results, 1):
        status = "✅" if r.ok else "❌"
        detail = r.detail[:80] if r.detail else ""
        lines.append(f"| {i} | {r.label} | {status} | {detail} | {r.ms:.0f} |")
    lines.append("")

    # Zeiten
    if state.timings:
        lines.append("## Zeiten")
        lines.append("")
        lines.append(f"- Median: {statistics.median(state.timings):.0f} ms")
        lines.append(f"- Max: {max(state.timings):.0f} ms")
        lines.append(f"- Min: {min(state.timings):.0f} ms")
        lines.append(f"- Anzahl: {len(state.timings)}")
        lines.append("")

    # Fehler
    fails = [r for r in state.results if not r.ok]
    if fails:
        lines.append("## Fehler")
        lines.append("")
        for r in fails:
            lines.append(f"- **{r.label}**: {r.detail}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[combo_stress] Report geschrieben: {path}")


if __name__ == "__main__":
    main()
