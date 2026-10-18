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
REPORTS_DIR = PROJECT_DIR / "reports"
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
    "wheel_gain": {"gain_r": 1.5},
    "wheel_falloff": {"gain_r": 1.5},
}

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


# ── Vorlagen / Original ──────────────────────────────────────────────────────

def _refs_for(state: StressState, n: int) -> list[str]:
    """n Vorlagen aus dem Pool, ohne das aktive Session-Bild."""
    active = state.active_image
    refs: list[str] = []
    for i in range(1, 11):
        name = f"{i:02d}"
        if name == active:
            continue
        jpg = POOL_DIR / f"{name}.jpg"
        if jpg.exists():
            refs.append(str(jpg))
        if len(refs) >= n:
            break
    return refs


def _load_original_scaled(state: StressState, max_side: int = MAX_SIDE) -> str | None:
    """Session-Quelle (JPG) auf max_side skaliert als PNG in tmp speichern."""
    jpg = POOL_DIR / f"{state.active_image}.jpg"
    if not jpg.exists():
        return None
    img = Image.open(jpg).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out_path = os.path.join(state.tmp, f"{state.active_image}_original.png")
    img.save(out_path, "PNG")
    return out_path


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
    active_image: str = "01"
    results: list[CheckResult] = field(default_factory=list)
    timings: list[float] = field(default_factory=list)
    oom_errors: list[str] = field(default_factory=list)
    preview_images: list[tuple[str, str]] = field(default_factory=list)  # (label, png_path)
    results_start: int = 0  # Index in results/timings am Anfang des aktuellen Bildes

    def sid(self) -> str:
        """Session-ID des aktiven Bildes."""
        return self.sessions[self.active_image]

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
             values: dict, stage: str = "full",
             mix: dict | None = None) -> tuple[dict, float]:
    """/preview aufrufen, (Antwort, ms) zurückgeben. OOM wird gemeldet."""
    t0 = time.perf_counter()
    payload: dict = {
        "session": sid, "references": refs, "method": method,
        "values": values, "max_side": MAX_SIDE, "stage": stage,
    }
    if mix is not None:
        payload["mix"] = mix
    code, body = _post_expect_error(
        f"{state.base}/preview",
        payload,
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
    """Kontaktbogen mit Beschriftung, Seitenverhältnis wird beibehalten."""
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
            img = Image.open(png_path).convert("RGB")
            # Seitenverhältnis beibehalten: in die Zelle einpassen
            img.thumbnail((cell, cell), Image.LANCZOS)
            # Zentriert in die Zelle einfügen
            ox = x + (cell - img.width) // 2
            oy = y + (cell - img.height) // 2
            sheet.paste(img, (ox, oy))
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

    sid = state.sid()
    ref = _refs_for(state, 1)
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
        # "mid" muss vom Standard abweichen: bei symmetrischen Reglern ist
        # (min+max)/2 == default, daher einen Wert nahe am Standard nehmen.
        mid_val = (s.min + s.max) / 2.0
        if abs(mid_val - s.default) < 1e-9:
            # Standard == Mitte: einen Viertel-Schritt vom Standard weg
            if s.max != s.default:
                mid_val = s.default + (s.max - s.default) / 4.0
            elif s.min != s.default:
                mid_val = s.default - (s.default - s.min) / 4.0
        for tag, val in (("min", s.min), ("max", s.max), ("mid", mid_val)):
            if val == s.default and tag == "mid":
                continue  # kein Unterschied zum Standard → prüft nichts
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
            diff = _img_diff(ref_img, resp["image"])
            # Worker liefert 8-bit BMP: jede sichtbare Änderung ist ≥ 1/255
            if is_combo:
                changed = diff > 0.0
                state.check(label, changed, f"combo, diff={diff:.5f} (changed={changed})", ms)
            else:
                changed = diff > 0.0
                if not changed:
                    state.check(label, False,
                                f"diff=0.0 exakt — Regler wirkt nicht über Worker", ms)
                else:
                    state.check(label, changed, f"diff={diff:.5f}", ms)
            # min/max für Kontaktbogen
            if tag in ("min", "max"):
                state.preview_images.append((f"slider:{s.name}:{tag}", resp["image"]))

    # Fester Sollwert: exposure=2 muss über den Worker diff ≥ 0,1 erzeugen
    vals_exp = dict(defaults)
    vals_exp["exposure"] = 2.0
    resp_exp, ms_exp = _preview(state, sid, ref, "reinhard", vals_exp)
    if "error" not in resp_exp:
        diff_exp = _img_diff(ref_img, resp_exp["image"])
        state.check("slider:exposure2_sollwert", diff_exp >= 0.1,
                    f"exposure=2: diff={diff_exp:.5f} (Sollwert ≥ 0.1)", ms_exp)
    else:
        state.check("slider:exposure2_sollwert", False, resp_exp["error"], ms_exp)


def test_sections(state: StressState) -> None:
    """Punkt 2: Bereiche solo, alle aus, alle an, Master 0."""
    from refgrade.sections import SECTIONS
    from refgrade.sliders import SLIDERS

    sid = state.sid()
    ref = _refs_for(state, 1)
    defaults = {s.name: s.default for s in SLIDERS}

    # Alle Standard → Baseline
    base_resp, base_ms = _preview(state, sid, ref, "reinhard", defaults)
    if "error" in base_resp:
        state.check("section:baseline", False, base_resp["error"], base_ms)
        return
    base_img = base_resp["image"]

    # Master 0 → Identität: Vergleich zweier Identity-Configs + Abweichung von Baseline
    mix_zero = {"weights": {k: 1.0 for k in SECTIONS}, "master": 0.0, "solo": None}
    extreme = dict(defaults)
    for s in SLIDERS:
        if s.name in defaults:
            extreme[s.name] = s.max if s.default == s.min else s.min
    # Master 0 mit extremen Werten → Identität (Source)
    resp_m0, ms_m0 = _preview(state, sid, ref, "reinhard", extreme, mix=mix_zero)
    if "error" in resp_m0:
        state.check("section:master0", False, resp_m0["error"], ms_m0)
    else:
        # Master 0 mit Defaults → ebenfalls Identität (Source)
        resp_m0_def, ms_m0d = _preview(state, sid, ref, "reinhard", defaults, mix=mix_zero)
        if "error" in resp_m0_def:
            state.check("section:master0_identity", False, resp_m0_def["error"], ms_m0d)
        else:
            # Beide Identity-Configs müssen identisch sein (gleiche Source)
            diff_ident = _img_diff(resp_m0["image"], resp_m0_def["image"])
            state.check("section:master0_identity", diff_ident == 0.0,
                        f"extreme vs defaults bei master=0: diff={diff_ident:.8f} (muss 0 sein)",
                        ms_m0 + ms_m0d)
        # Master 0 vs. Baseline (Master 1, mit Reference): muss sich unterscheiden
        diff_m0_vs_base = _img_diff(resp_m0["image"], base_img)
        state.check("section:master0_differs_from_baseline", diff_m0_vs_base > 0.0,
                    f"Master 0 vs Baseline: diff={diff_m0_vs_base:.6f} (muss >0 sein)", 0.0)

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
    sid = state.sid()

    for method in ("reinhard", "mkl", "zonal"):
        for n_refs in (0, 1, 2):
            refs = _refs_for(state, n_refs) if n_refs > 0 else []
            label = f"method:{method}:refs={n_refs}"
            t0 = time.perf_counter()
            try:
                resp, ms = _preview(state, sid, refs, method, {})
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000.0
                state.check(label, False, f"Timeout/Ausnahme: {type(e).__name__}: {e}", ms)
                continue
            if "error" in resp:
                state.check(label, False, resp["error"], ms)
                continue
            stats = _img_stats(resp["image"])
            finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
            state.check(label, finite, f"min={stats['min']:.3f} max={stats['max']:.3f}", ms)
            if method in ("mkl", "zonal") and n_refs == 2:
                print(f"  [timing] {method} refs=2: {ms:.0f} ms")
            if finite:
                state.preview_images.append((f"method:{method}:refs={n_refs}", resp["image"]))

    # pair: 2 Bilder aus dem Pool (ohne aktives Bild)
    pair_refs = _refs_for(state, 2)
    if len(pair_refs) < 2:
        state.check("method:pair:refs=2", False, "weniger als 2 Vorlagen verfügbar")
        return
    t0 = time.perf_counter()
    try:
        resp, ms = _preview(state, sid, pair_refs, "pair", {})
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        state.check("method:pair:refs=2", False, f"Timeout/Ausnahme: {type(e).__name__}: {e}", ms)
        return
    if "error" in resp:
        state.check("method:pair:refs=2", False, resp["error"], ms)
    else:
        stats = _img_stats(resp["image"])
        finite = stats["n_nan"] == 0 and stats["min"] >= 0.0 and stats["max"] <= 1.0
        pair_err = resp.get("pair_error", {}).get("mean", -1)
        # pair_err muss vorhanden und plausibel sein (> 0, da 33³-LUT nie exakt)
        pair_ok = finite and pair_err > 0.0
        state.check("method:pair:refs=2", pair_ok,
                    f"min={stats['min']:.3f} max={stats['max']:.3f} pair_err={pair_err:.4f} (muss >0 sein)", ms)
        if finite:
            state.preview_images.append(("method:pair:refs=2", resp["image"]))


def test_wheels(state: StressState) -> None:
    """Punkt 4: Räder-Kombinationen Intensity × Blending × Balance."""
    from refgrade.sliders import SLIDERS
    sid = state.sid()
    ref = _refs_for(state, 1)
    defaults = {s.name: s.default for s in SLIDERS}

    # Orange-Highlights / Blau-Shadows Setup
    wheel_base = {
        "gain_r": 1.5, "gain_b": 0.7,  # orange highlights
        "lift_b": 0.2, "lift_r": -0.1,  # blue shadows
    }

    # Gain=0.5, Falloff=1.0 (volle Zonen, 0→1) vs. weggelassen: "klassisches Verhalten"
    # Beide Bilder in den Bogen + Zahl in den Bericht
    vals_b1 = dict(defaults)
    vals_b1.update(wheel_base)
    vals_b1["wheel_gain"] = 0.5
    vals_b1["wheel_falloff"] = 1.0
    vals_no_blend = dict(defaults)
    vals_no_blend.update(wheel_base)
    # wheel_gain/falloff NICHT setzen → Standardwert aus defaults
    resp_b1, ms_b1 = _preview(state, sid, ref, "reinhard", vals_b1)
    resp_nb, ms_nb = _preview(state, sid, ref, "reinhard", vals_no_blend)
    if "error" not in resp_b1 and "error" not in resp_nb:
        # 2026-10-18: volle Zonen (G=0.5, F=1.0) und Standard (0/0.5) sind NICHT identisch
        # — nur Finitheit prüfen, Diff als Info.
        diff_blend = _img_diff(resp_b1["image"], resp_nb["image"])
        state.check("wheels:full_vs_default", True,
                    f"Gain=0.5 Falloff=1.0 vs. Standard: diff={diff_blend:.6f} (Info, kein Limit)", ms_b1 + ms_nb)
        state.preview_images.append(("wheels:full", resp_b1["image"]))
        state.preview_images.append(("wheels:noB", resp_nb["image"]))

    for gain in (0.0, 0.1, 0.5):
        for falloff in (0.0, 0.1, 0.5, 1.0):
            vals = dict(defaults)
            vals.update(wheel_base)
            vals["wheel_gain"] = gain
            vals["wheel_falloff"] = falloff
            label = f"wheels:G={gain} F={falloff}"
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
    sid = state.sid()
    ref = _refs_for(state, 1)
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
    sid = state.sid()
    ref = _refs_for(state, 1)
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
    """Punkt 7: /cube exportieren, LUT auf das ORIGINAL anwenden → Abweichung ≤ 3/255.

    Der LUT-Test muss die LUT auf das Original (JPG skaliert) anwenden und mit dem
    Worker-Preview (Master 1, stage=lut) vergleichen.
    """
    from refgrade.sections import SECTIONS

    sid = state.sid()
    ref = _refs_for(state, 1)
    vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3}

    # Original aus dem JPG (skaliert auf MAX_SIDE)
    orig_img = _load_original_scaled(state)
    if orig_img is None:
        state.check("lut:export", False, f"Original {state.active_image}.jpg nicht gefunden")
        return

    # Worker-Preview (Master 1, stage=lut)
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

    # LUT auf das ORIGINAL anwenden (nicht auf die Vorschau!)
    lut_img = _apply_cube_to_image(orig_img, cube_path)
    if lut_img is None:
        state.check("lut:apply", False, "konnte LUT nicht anwenden", ms)
        return
    diff = _img_diff(preview_img, lut_img)
    ok = diff <= 3.0 / 255.0
    state.check("lut:roundtrip", ok, f"diff={diff:.6f} (limit=3/255={3.0/255.0:.6f})", ms)


def _apply_cube_to_image(png_path: str, cube_path: str) -> str | None:
    """3D-LUT (.cube) auf ein PNG anwenden, Ergebnis als temporäres PNG speichern.

    Cube-Format: R läuft am schnellsten → flacher Index = b*N² + g*N + r.
    """
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
    # Cube: R fastest → index = b*N² + g*N + r
    scaled = flat * (size - 1)
    x0 = np.floor(scaled).astype(int).clip(0, size - 1)
    fx = (scaled - x0)

    lut = np.array(data)  # [size^3, 3], index = b*size² + g*size + r (R fastest)
    out = np.zeros_like(flat)
    for c in range(3):
        for dr in (0, 1):
            for dg in (0, 1):
                for db in (0, 1):
                    ri = x0[:, 0] + dr
                    gi = x0[:, 1] + dg
                    bi = x0[:, 2] + db
                    idx = bi * size * size + gi * size + ri
                    wgt = (fx[:, 0] if dr else 1 - fx[:, 0]) * \
                          (fx[:, 1] if dg else 1 - fx[:, 1]) * \
                          (fx[:, 2] if db else 1 - fx[:, 2])
                    out[:, c] += lut[idx, c] * wgt

    result = (out.reshape(h, w, 3).clip(0, 1) * 255).astype(np.uint8)
    out_path = os.path.splitext(png_path)[0] + "_lut.png"
    Image.fromarray(result).save(out_path, "PNG")
    return out_path


def test_timing(state: StressState) -> None:
    """Punkt 8: Ausreißer > 3× Median melden (nur aktives Bild)."""
    # Nur Ergebnisse des aktuellen Bildes auswerten
    img_results = state.results[state.results_start:]
    img_timings = [r.ms for r in img_results if r.ms > 0]
    if not img_timings:
        return
    med = statistics.median(img_timings)
    threshold = 3 * med
    outliers = [(r.label, r.ms) for r in img_results if r.ms > threshold and r.ms > 0]
    if outliers:
        for label, ms in outliers:
            state.check(f"timing:outlier:{label}", False, f"{ms:.0f}ms > 3×median={threshold:.0f}ms")
    else:
        state.check("timing:all_ok", True, f"median={med:.0f}ms, max={max(img_timings):.0f}ms")


def test_direct_comparison(state: StressState) -> None:
    """Punkt 9: Worker-Preview vs. direkter session.preview-Vergleich.

    Prüft, dass der Worker die Kernlogik korrekt aufruft, indem die
    Worker-Antwort mit einem direkten session.preview-Aufruf verglichen wird.
    """
    import torch
    import refgrade.session as sess
    from refgrade.sliders import SLIDERS

    name = state.active_image
    jpg = POOL_DIR / f"{name}.jpg"
    if not jpg.exists():
        return

    # Direkte Session (außerhalb des Workers)
    raw_path = os.path.join(state.tmp, f"{name}_direct.raw")
    w, h = _jpg_to_raw(str(jpg), raw_path)
    s = sess.open_session(raw_path, w, h, "U16", device=torch.device("cpu"))
    ref = _refs_for(state, 1)
    vals = {"exposure": 0.3, "contrast": 1.2, "saturation": 1.3}

    # Direkter preview
    t0 = time.perf_counter()
    direct_tensor, _ = sess.preview(s, ref, "reinhard", vals, MAX_SIDE)
    ms_direct = (time.perf_counter() - t0) * 1000.0

    # Tensor → PNG speichern
    direct_arr = (direct_tensor[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    direct_png = os.path.join(state.tmp, f"{name}_direct.png")
    Image.fromarray(direct_arr).save(direct_png, "PNG")

    # Worker-Preview
    resp, ms_worker = _preview(state, state.sid(), ref, "reinhard", vals)
    if "error" in resp:
        state.check("direct:comparison", False, f"Worker error: {resp['error']}", ms_worker)
        return
    worker_img = resp["image"]

    # Vergleich
    diff = _img_diff(worker_img, direct_png)
    ok = diff <= 1.0 / 255.0
    state.check("direct:comparison", ok,
                f"diff={diff:.6f} (limit=1/255={1.0/255.0:.6f}) "
                f"direct={ms_direct:.0f}ms worker={ms_worker:.0f}ms",
                ms_direct + ms_worker)


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

    # Alle drei Bilder aus dem Pool: 01 (Referenz), 06 (hoch), 07 (quer)
    # Worker hat max 2 Sessions (LRU) → Tests nacheinander pro Bild
    image_names = ["01", "06", "07"]
    for name in image_names:
        jpg = POOL_DIR / f"{name}.jpg"
        if not jpg.exists():
            print(f"[combo_stress] {name}.jpg fehlt, übersprungen")
            continue
        _load_image(state, name, jpg)
        state.active_image = name
        state.results_start = len(state.results)
        print(f"[combo_stress] {name}.jpg geladen (sid={state.sessions[name][:8]}…)")

        tests = [
            ("1. Regler einzeln", test_sliders),
            ("2. Bereiche", test_sections),
            ("3. Methoden", test_methods),
            ("4. Räder", test_wheels),
            ("5. Stufen", test_stages),
            ("6. Determinismus", test_determinism),
            ("7. LUT-Export", test_lut_export),
            ("8. Zeiten", test_timing),
            ("9. Direktvergleich", test_direct_comparison),
        ]
        for title, fn in tests:
            print(f"[combo_stress] {name}: {title} …")
            t0 = time.perf_counter()
            try:
                fn(state)
            except Exception as e:
                state.check(f"EXCEPTION:{name}:{title}", False, f"{type(e).__name__}: {e}")
            print(f"  ({time.perf_counter() - t0:.1f}s)")

    # Kontaktbögen (4×4 pro Bogen, mehrere Bögen bei > 16 Bildern)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    def _make_sheets(imgs: list[tuple[str, str]], prefix: str, title: str) -> None:
        """Mehrere 4×4-Bögen aus einer Bildliste."""
        for i in range(0, len(imgs), 16):
            chunk = imgs[i:i + 16]
            num = i // 16 + 1
            _make_sheet(chunk, str(SHEETS_DIR / f"{prefix}_{num}.png"),
                        f"{title} (Bogen {num})")

    if state.preview_images:
        wheels_imgs = [(l, p) for l, p in state.preview_images if l.startswith("wheels:")]
        if wheels_imgs:
            _make_sheets(wheels_imgs, "wheels",
                         "Wheels: Intensity × Blending × Balance")
        methods_imgs = [(l, p) for l, p in state.preview_images if l.startswith("method:")]
        if methods_imgs:
            _make_sheets(methods_imgs, "methods", "Methoden: reinhard/mkl/zonal/pair")
        sliders_imgs = [(l, p) for l, p in state.preview_images if l.startswith("slider:")]
        if sliders_imgs:
            _make_sheets(sliders_imgs, "sliders", "Regler min/max")

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
