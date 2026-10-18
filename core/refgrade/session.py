"""Sitzungslogik: Session, LUT-Bau, Vorschau, Apply, Fit, Rank, Cube."""

from __future__ import annotations

import math

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import rawio
from .device import pick_device, synchronize
from .clarity import apply as clarity_apply
from .dehaze import apply as dehaze_apply
from .texture import apply as texture_apply
from .recover import apply as reconstruct_apply
from .color import rgb_to_lab, lab_to_rgb
from .curves import is_identity as curves_is_identity, validate as curves_validate
from .fit_pair import pair_shift as _pair_shift
from .fit_sliders import fit_sliders as _fit_sliders
from .grade import grade
from .lut import build as lut_build, write_cube as lut_write_cube
from .pairlut import fit as pair_fit, split_pair as pair_split, as_lut as pair_as_lut
from .protect import protect
from .reference_rank import rank as _rank
from .sections import FITTABLE, SECTIONS, VISIBLE, chain_order, chain_step, section_values
from .sliders import SLIDERS
from .transfer import mkl as _mkl, reinhard as _reinhard
from .zonal import analyze as _zonal_analyze, apply_params as _zonal_apply

logger = logging.getLogger("RefGrade")

PREVIEW_SIDE = 1536
REF_SIDE = 768
LUT_SIZE = 33
HIST_BINS = 64
RGB_HIST_BINS = 128
APPLY_CHUNK = 4_000_000

_VALID_METHODS = ("reinhard", "mkl", "zonal", "pair")
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*]')


@dataclass
class Session:
    """Zustand einer REF-Grade-Sitzung."""
    source_full: torch.Tensor
    alpha: np.ndarray
    depth: str
    source: torch.Tensor
    ignore: torch.Tensor | None
    refs: dict = field(default_factory=dict)
    luts: dict = field(default_factory=dict)
    cube_cache: dict = field(default_factory=dict)
    # Caches for things a slider/curve drag does not change, so it only rebuilds the LUT.
    method_fns: dict = field(default_factory=dict)
    previews: dict = field(default_factory=dict)
    hists: dict = field(default_factory=dict)
    pair_luts: dict = field(default_factory=dict)
    pair_shifts: dict = field(default_factory=dict)
    # Präfix-Cache der Kette am LUT-Gitter: Schlüssel = alle Glieder bis hierher. Ändert sich ein
    # spätes Glied (z. B. die Räder), werden die früheren aus dem Cache genommen (LRU, begrenzt).
    section_grid_cache: dict = field(default_factory=dict)


def open_session(
    raw: str,
    width: int,
    height: int,
    depth: str,
    ignore_mask: str | None = None,
    device: torch.device | None = None,
) -> Session:
    """Session aus Rohdaten öffnen.

    Args:
        raw: Pfad zur Rohdatendatei.
        width: Breite in Pixeln.
        height: Höhe in Pixeln.
        depth: "U8" oder "U16".
        ignore_mask: optional Pfad zur Maskendatei (1 = ignorieren).
        device: Zielgerät für die Vorschau.

    Returns:
        Session-Instanz.
    """
    if device is None:
        device = pick_device()

    source_full, alpha = rawio.read_raw(raw, width, height, depth)

    # Vorschau: längste Seite ≤ PREVIEW_SIDE, nie vergrößern
    h, w = height, width
    scale = min(1.0, PREVIEW_SIDE / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        source = F.interpolate(
            source_full.permute(0, 3, 1, 2), size=(new_h, new_w),
            mode="area",
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    else:
        source = source_full

    # Ignore-Maske: 1-m (Gewicht), auf Vorschaugröße
    ignore = None
    if ignore_mask is not None:
        mask = rawio.read_mask(ignore_mask, width, height)
        if scale < 1.0:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                mode="area",
            ).squeeze()
        ignore = (1.0 - mask).clamp(0.0, 1.0)

    source = source.to(device)
    if ignore is not None:
        ignore = ignore.to(device)

    return Session(
        source_full=source_full,
        alpha=alpha,
        depth=depth,
        source=source,
        ignore=ignore,
    )


def values_with_defaults(values: dict) -> dict:
    """Fehlende Regler mit SLIDERS-Standardwerten ergänzen.

    Raises:
        KeyError: bei unbekanntem Reglernamen.
    """
    defaults = {s.name: s.default for s in SLIDERS}
    for name in values:
        if name not in defaults:
            raise KeyError(f"Unbekannter Regler: {name!r}")
    result = dict(defaults)
    result.update(values)
    return result


def is_identity(values: dict, mix: dict | None = None) -> bool:
    """True wenn der Mix die Quelle unverändert zurückgibt.

    Identität gilt bei:
    - master == 0 (ohne SOLO)
    - allen w_i == 0 (ohne SOLO)
    - jedem aktiven Bereich auf Standard UND (reference w=0 ODER Stärken 0)
    - lut: Gewicht > 0 aber kein Pfad → übersprungen → Identität
    - curves: Gewicht > 0 aber keine Kurven oder Identitätskurven → übersprungen
    """
    if mix is None:
        mix = {"weights": {k: 1.0 for k in SECTIONS}, "master": 1.0, "solo": None}

    solo = mix.get("solo", None)
    master = mix.get("master", 1.0)
    weights = mix.get("weights", {})

    # Bestimme aktive Bereiche
    if solo is not None:
        active = {solo: 1.0}
    else:
        if master == 0.0:
            return True
        active = {k: w for k, w in weights.items() if w > 0.0}
        if not active:
            return True

    # lut ohne Pfad wirkt nicht
    if "lut" in active and not mix.get("lut"):
        del active["lut"]

    # curves ohne Daten oder mit Identitätskurven wirkt nicht
    if "curves" in active:
        if not mix.get("curves") or curves_is_identity(mix.get("curves")):
            del active["curves"]

    defaults = {s.name: s.default for s in SLIDERS}
    for key in active:
        if key == "reference":
            if values.get("light_strength", 1.0) != 0.0:
                return False
            if values.get("color_strength", 1.0) != 0.0:
                return False
        elif key == "lut":
            # LUT ist aktiv (Pfad gesetzt) → keine Identität
            return False
        elif key == "curves":
            # Kurven sind aktiv und nicht-identisch → keine Identität
            return False
        else:
            for name in SECTIONS[key]:
                if name == "clarity":
                    continue  # not in LUT
                if values.get(name, defaults[name]) != defaults[name]:
                    return False
    return True


def _pool_refs(s: Session, refs: list[str]) -> torch.Tensor:
    """Vorladen (gecacht) und Mittelwert bilden. Leere Liste → Source (Identität)."""
    if not refs:
        return s.source
    tensors = []
    for path in refs:
        if path not in s.refs:
            s.refs[path] = rawio.load_image(path, REF_SIDE)
        tensors.append(s.refs[path].to(s.source.device))
    return torch.cat(tensors, dim=0).mean(dim=0, keepdim=True)


def _curves_key(curves: dict | None) -> tuple:
    """Kurven als sortiertes Tupel für den Cache-Schlüssel (RGB + Hue)."""
    if not curves:
        return ()
    from .curves import HUE_KEYS
    all_keys = ("rgb", "r", "g", "b") + HUE_KEYS
    parts = []
    for ch in all_keys:
        if ch in curves:
            pts = tuple(tuple(float(v) for v in p) for p in curves[ch])
            parts.append((ch, pts))
    return tuple(sorted(parts))


def _lut_key(method: str, refs: list[str], values: dict, mix: dict | None,
             lut_size: int = LUT_SIZE) -> tuple:
    """Cache-Schlüssel: Methode + Vorlagenpfade + sortierte Werte + Mix (+ LUT + Kurven) + LUT-Größe."""
    if mix is None:
        mix_key = ()
    else:
        weights = tuple(sorted((k, round(v, 6)) for k, v in mix.get("weights", {}).items()))
        lut_path = mix.get("lut", "")
        lut_mtime = 0
        if lut_path:
            try:
                lut_mtime = int(Path(lut_path).stat().st_mtime)
            except OSError:
                lut_mtime = -1
        curves_k = _curves_key(mix.get("curves"))
        mix_key = (weights, round(mix.get("master", 1.0), 6), mix.get("solo"), lut_path, lut_mtime, curves_k,
                   chain_order(mix))
    return (method, tuple(refs), tuple(sorted((k, round(v, 6)) for k, v in values.items())), mix_key, lut_size)


def _build_method_fn(
    method: str,
    source: torch.Tensor,
    reference: torch.Tensor,
    image_weights: torch.Tensor | None,
    light_strength: float,
    color_strength: float,
) -> callable:
    """Methode-Callable bauen (sRGB → sRGB, pro Pixel)."""
    if method == "reinhard":
        return _reinhard(source, reference, image_weights, None, strength=light_strength)
    elif method == "mkl":
        return _mkl(source, reference, image_weights, None, strength=light_strength)
    elif method == "zonal":
        params = _zonal_analyze(source, reference, image_weights, None)
        lab_fn = _zonal_apply(params, light_strength, color_strength)
        from .color import rgb_to_lab, lab_to_rgb

        def pipeline(srgb: torch.Tensor) -> torch.Tensor:
            return lab_to_rgb(lab_fn(rgb_to_lab(srgb)))
        return pipeline
    else:
        raise ValueError(f"Unbekannte Methode: {method!r}. Verfügbar: {_VALID_METHODS}")


def _resolve_pair(s: Session, refs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the before/after pair from the reference list.

    Args:
        s: Session.
        refs: Reference paths (1 or 2 entries).

    Returns:
        (before, after) tensors [1,H,W,3] on the session device.

    Raises:
        ValueError: if the refs cannot form a valid pair.
    """
    for path in refs:                     # not pooled, but loaded the same way (cached)
        if path not in s.refs:
            s.refs[path] = rawio.load_image(path, REF_SIDE)
    if len(refs) == 2:
        before = s.refs[refs[0]].to(s.source.device)
        after = s.refs[refs[1]].to(s.source.device)
    elif len(refs) == 1:
        img = s.refs[refs[0]].to(s.source.device)
        H, W = img.shape[1], img.shape[2]
        if W / H >= 1.8:
            before, after = pair_split(img)
        else:
            raise ValueError(
                "PAIR needs a before and an after image (or one side-by-side image)"
            )
    else:
        raise ValueError(
            "PAIR needs a before and an after image (or one side-by-side image)"
        )

    # Resize after to before's size if needed
    if after.shape[1:3] != before.shape[1:3]:
        after = F.interpolate(
            after.permute(0, 3, 1, 2), size=before.shape[1:3],
            mode="area",
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

    return before, after


def _get_pair_lut(s: Session, refs: list[str], size: int) -> dict:
    """Fit (or return cached) the pair LUT.

    Returns:
        (lut_dict, pair_error) where pair_error is {"mean": float, "p95": float}.
    """
    key = (tuple(refs), size)
    if key in s.pair_luts:
        return s.pair_luts[key]

    before, after = _resolve_pair(s, refs)
    table = pair_fit(before, after, size=size)
    lut = pair_as_lut(table, title="PAIR")

    # Compute ΔE error on the pair itself
    from .lut import apply as lut_apply
    result = lut_apply(before, lut)
    lab_r = rgb_to_lab(result)
    lab_a = rgb_to_lab(after)
    d = (lab_r - lab_a).norm(dim=-1).flatten()
    pair_error = {"mean": d.mean().item(), "p95": d.quantile(0.95).item()}

    s.pair_luts[key] = (lut, pair_error)
    return s.pair_luts[key]


def _get_pair_shift(s: Session, refs: list[str]) -> float:
    """Phasenkorrelations-Verschiebung des Paares in px (gecacht)."""
    key = tuple(refs)
    if key not in s.pair_shifts:
        before, after = _resolve_pair(s, refs)
        s.pair_shifts[key] = _pair_shift(before, after)
    return s.pair_shifts[key]


def _build_pair_fn(
    s: Session,
    refs: list[str],
    light_strength: float,
    color_strength: float,
) -> callable:
    """Build the pair method function: LUT + Lab mix with strengths."""
    lut, _ = _get_pair_lut(s, refs, LUT_SIZE)

    def pipeline(x: torch.Tensor) -> torch.Tensor:
        if light_strength == 0 and color_strength == 0:
            return x
        from .lut import apply as lut_apply
        matched = lut_apply(x, lut)
        if light_strength == 1.0 and color_strength == 1.0:
            return matched
        lab_x = rgb_to_lab(x)
        lab_m = rgb_to_lab(matched)
        lab_out = lab_x.clone()
        lab_out[..., 0] = lab_x[..., 0] + light_strength * (lab_m[..., 0] - lab_x[..., 0])
        lab_out[..., 1] = lab_x[..., 1] + color_strength * (lab_m[..., 1] - lab_x[..., 1])
        lab_out[..., 2] = lab_x[..., 2] + color_strength * (lab_m[..., 2] - lab_x[..., 2])
        return lab_to_rgb(lab_out)

    return pipeline


def _load_cube_cached(s: Session, path: str) -> dict:
    """LUT-Datei laden, gecacht nach (Pfad, mtime).

    Raises:
        ValueError: Datei fehlt, ist unlesbar oder hat nicht die Endung .cube.
    """
    from .lut import read_cube
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"LUT not readable: {path}")
    if p.suffix.lower() != ".cube":
        raise ValueError(f"LUT not readable: {path}")
    mtime = int(p.stat().st_mtime)
    cache_key = (path, mtime)
    if cache_key not in s.cube_cache:
        s.cube_cache[cache_key] = read_cube(path)
    return s.cube_cache[cache_key]


def _section_fns(s: Session, refs: list[str], method: str, vals: dict, mix: dict,
                 timing: dict | None = None) -> dict[str, callable]:
    """Bereichsfunktionen (je sRGB → sRGB) — eine Quelle für LUT-Bau und AUTO-Eingang."""
    import time as _time
    _t0 = _time.perf_counter()

    # Vier Bereichsfunktionen bauen (jeweils sRGB → sRGB)
    source_fns: dict[str, callable] = {}

    # reference: protect(x, method_fn(x), v_ref)
    fn_key = (method, tuple(refs), vals["light_strength"], vals["color_strength"])
    if fn_key not in s.method_fns:
        if method == "pair":
            s.method_fns[fn_key] = _build_pair_fn(
                s, refs, vals["light_strength"], vals["color_strength"],
            )
        else:
            reference = _pool_refs(s, refs)
            s.method_fns[fn_key] = _build_method_fn(
                method, s.source, reference, s.ignore,
                vals["light_strength"], vals["color_strength"],
            )
    method_fn = s.method_fns[fn_key]
    v_ref = section_values(vals, "reference")
    source_fns["reference"] = lambda x: protect(x, method_fn(x), v_ref)
    if timing is not None:
        synchronize(s.source.device)
        timing["lut_method_fn"] = (_time.perf_counter() - _t0) * 1000.0

    # tone, color, wheels, hsl: grade(x, v_section)
    for sec in ("tone", "color", "wheels", "hsl"):
        v_sec = section_values(vals, sec)
        source_fns[sec] = (lambda x, _v=v_sec: grade(x, _v))

    # rgb: rgb_grade(x, ...) mit Zonenauswahl
    v_rgb = section_values(vals, "rgb")
    from .rgb import rgb_grade

    def _rgb_fn(x: torch.Tensor, _v=v_rgb) -> torch.Tensor:
        return rgb_grade(
            x,
            master_gamma=_v["rgb_m_gamma"],
            master_pedestal=_v["rgb_m_pedestal"],
            master_gain=_v["rgb_m_gain"],
            r_gamma=_v["rgb_r_gamma"],
            r_pedestal=_v["rgb_r_pedestal"],
            r_gain=_v["rgb_r_gain"],
            g_gamma=_v["rgb_g_gamma"],
            g_pedestal=_v["rgb_g_pedestal"],
            g_gain=_v["rgb_g_gain"],
            b_gamma=_v["rgb_b_gamma"],
            b_pedestal=_v["rgb_b_pedestal"],
            b_gain=_v["rgb_b_gain"],
            use_shadows=_v["rgb_use_shadows"],
            use_midtones=_v["rgb_use_midtones"],
            use_highlights=_v["rgb_use_highlights"],
        )

    source_fns["rgb"] = _rgb_fn

    # lut: geladene .cube, nur wenn Gewicht > 0 und Pfad gesetzt
    lut_path = mix.get("lut", "")
    if "lut" in mix.get("weights", {}) and mix["weights"].get("lut", 0) > 0.0 and lut_path:
        cube = _load_cube_cached(s, lut_path)
        from .lut import apply as lut_apply
        source_fns["lut"] = (lambda x, _c=cube: lut_apply(x, _c))

    # curves: RGB-Kurven + Hue-Kurven, nur wenn Gewicht > 0 und nicht-identisch
    curves_data = mix.get("curves")
    if "curves" in mix.get("weights", {}) and mix["weights"].get("curves", 0) > 0.0 and curves_data:
        curves_validate(curves_data)
        if not curves_is_identity(curves_data):
            from .curves import apply as curves_apply, HUE_KEYS
            from .huecurves import apply as huecurves_apply
            rgb_keys = ("rgb", "r", "g", "b")
            rgb_data = {k: v for k, v in curves_data.items() if k in rgb_keys}
            hue_data = {k: v for k, v in curves_data.items() if k in HUE_KEYS}

            def _curves_pipeline(x: torch.Tensor, _rgb=rgb_data, _hue=hue_data) -> torch.Tensor:
                y = curves_apply(x, _rgb) if _rgb else x
                if _hue:
                    y = huecurves_apply(y, _hue)
                return y.clamp(0.0, 1.0)

            source_fns["curves"] = _curves_pipeline

    # color_replace: HSL-Distanz-Maske, mehrere Instanzen
    from .color_replace import apply as cr_apply
    v_cr = section_values(vals, "color_replace")
    source_fns["color_replace"] = (lambda x, _v=v_cr: cr_apply(x, _v))

    return source_fns


def chain_input(s: Session, refs: list[str], method: str, values: dict, mix: dict | None,
                section: str, img: torch.Tensor) -> torch.Tensor:
    """Bild, wie es in der Kette beim Bereich `section` ankommt (alle Glieder davor, mit Amount).

    Grundlage der AUTO-Knöpfe: was davor schon korrigiert wurde, ist enthalten.
    """
    if mix is None:
        mix = {"weights": {k: 1.0 for k in SECTIONS}, "master": 1.0, "solo": None}
    vals = values_with_defaults(values)
    fns = _section_fns(s, refs, method, vals, mix)
    weights = mix.get("weights", {})
    y = img.clamp(0.0, 1.0)
    for key in chain_order(mix):
        if key == section:
            break
        w = weights.get(key, 0.0)
        if w > 0.0 and key in fns:
            y = chain_step(y, fns[key], w)
    return y


def build_lut(s: Session, refs: list[str], method: str, values: dict,
              mix: dict | None = None, timing: dict | None = None,
              lut_size: int | None = None) -> dict:
    """LUT bauen: Kette der Bereiche (Reihenfolge sections.DEFAULT_ORDER bzw. mix["order"]), gecacht.

    Args:
        s: Session.
        refs: Liste der Vorlagenpfade.
        method: "reinhard", "mkl" oder "zonal".
        values: Reglerwerte (unvollständig erlaubt).
        mix: {"weights": {key: 0..1}, "master": 0..1, "solo": key|None,
              "lut": optional Pfad zu .cube}.
             None → alle Gewichte 1, master 1, kein SOLO.
        timing: Optionaler Dict, in den Teilzeiten (ms) geschrieben werden.
        lut_size: Kantenlänge der LUT (K31b-6: 17 beim Ziehen, 33 sonst). None → 33.

    Returns:
        LUT-Dict (size, table, title).

    Raises:
        ValueError: bei unbekannter Methode oder unlesbarer LUT-Datei.
        KeyError: bei unbekanntem Bereichsschlüssel in mix.
    """
    if lut_size is None:
        lut_size = LUT_SIZE
    if method not in _VALID_METHODS:
        raise ValueError(f"Unbekannte Methode: {method!r}. Verfügbar: {_VALID_METHODS}")

    if mix is None:
        mix = {"weights": {k: 1.0 for k in SECTIONS}, "master": 1.0, "solo": None}
    else:
        # Validierung: unbekannte Schlüssel und Werte außerhalb 0..2
        # (2026-10-18: bis 2.0 — chain_step extrapoliert bei w > 1, z. B. Wheels-Amount 200 %)
        for key in mix.get("weights", {}):
            if key not in SECTIONS:
                raise KeyError(f"Unbekannter Bereich: {key!r}")
            if not (0.0 <= mix["weights"][key] <= 2.0):
                raise KeyError(f"Gewicht für {key!r} außerhalb 0..2")
        if mix.get("solo") is not None and mix["solo"] not in SECTIONS:
            raise KeyError(f"Unbekannter SOLO-Bereich: {mix['solo']!r}")
        if not (0.0 <= mix.get("master", 1.0) <= 1.0):
            raise KeyError("master außerhalb 0..1")

    vals = values_with_defaults(values)
    key = _lut_key(method, refs, vals, mix, lut_size)
    if key in s.luts:
        if timing is not None:
            timing["lut_method_fn"] = 0.0
            timing["lut_section_fns"] = 0.0
            timing["lut_build"] = 0.0
        return s.luts[key]

    import time as _time
    _t1 = _time.perf_counter()
    source_fns = _section_fns(s, refs, method, vals, mix, timing)
    if timing is not None:
        timing["lut_section_fns"] = (_time.perf_counter() - _t1) * 1000.0

    # Kette am Gitter, Präfix-Cache: nur ab dem ersten geänderten Glied neu rechnen
    _t2 = _time.perf_counter()
    lut = _build_lut_cached(s, source_fns, mix, method, refs, vals, lut_size)
    if timing is not None:
        synchronize(s.source.device)
        timing["lut_build"] = (_time.perf_counter() - _t2) * 1000.0
    s.luts[key] = lut
    return lut


def _section_grid_key(sec: str, vals: dict, mix: dict, method: str,
                      refs: list[str], lut_size: int) -> tuple:
    """Cache key for a section's output at the LUT grid."""
    if sec == "reference":
        return ("reference", method, tuple(refs), vals["light_strength"],
                vals["color_strength"], vals["protect_shadows"],
                vals["protect_highlights"], vals.get("protect_midtones", 0.0), lut_size)
    if sec == "lut":
        return ("lut", mix.get("lut", ""), lut_size)
    if sec == "curves":
        return ("curves", repr(mix.get("curves")), lut_size)
    sec_vals = section_values(vals, sec)
    return (sec, tuple(sorted(sec_vals.items())), lut_size)


_CHAIN_CACHE_MAX = 24


def _build_lut_cached(s: Session, source_fns: dict[str, callable], mix: dict,
                      method: str, refs: list[str], vals: dict,
                      lut_size: int) -> dict:
    """LUT als Kette am Gitter bauen, mit Präfix-Cache (gleiches Ergebnis wie sections.chain_fn)."""
    device = s.source.device
    r_vals = torch.linspace(0.0, 1.0, lut_size, device=device)
    g_vals = torch.linspace(0.0, 1.0, lut_size, device=device)
    b_vals = torch.linspace(0.0, 1.0, lut_size, device=device)
    r_3d = r_vals[None, None, :].expand(lut_size, lut_size, lut_size)
    g_3d = g_vals[None, :, None].expand(lut_size, lut_size, lut_size)
    b_3d = b_vals[:, None, None].expand(lut_size, lut_size, lut_size)
    grid = torch.stack([r_3d, g_3d, b_3d], dim=-1)  # [N,N,N,3]

    weights = mix.get("weights", {})
    master = mix.get("master", 1.0)
    solo = mix.get("solo")
    cache = s.section_grid_cache

    def _cached(key, compute):
        if key in cache:
            val = cache.pop(key)          # LRU: zuletzt benutzt ans Ende
        else:
            val = compute()
            while len(cache) >= _CHAIN_CACHE_MAX:
                cache.pop(next(iter(cache)))
        cache[key] = val
        return val

    if solo is not None:
        key = ("solo", _section_grid_key(solo, vals, mix, method, refs, lut_size))
        table = _cached(key, lambda: source_fns[solo](grid).clamp(0.0, 1.0))
    elif master == 0.0:
        table = grid.clamp(0.0, 1.0)
    else:
        y = grid.clamp(0.0, 1.0)
        prefix: tuple = ("chain", lut_size)
        for sec_key in chain_order(mix):
            w = weights.get(sec_key, 0.0)
            if w <= 0.0 or sec_key not in source_fns:
                continue
            prefix = prefix + ((_section_grid_key(sec_key, vals, mix, method, refs, lut_size),
                                round(w, 6)),)
            y = _cached(prefix, lambda _y=y, _f=source_fns[sec_key], _w=w: chain_step(_y, _f, _w))
        table = (grid + master * (y - grid)).clamp(0.0, 1.0)

    return {"size": lut_size, "table": table, "title": "RefGrade"}


def _histogram_l(img: torch.Tensor) -> list[float]:
    """1D-Histogramm über L* 0..100, normalisiert auf Summe 1."""
    lab = rgb_to_lab(img)
    L = (lab[..., 0] / 100.0).clamp(0.0, 1.0)
    L = L.cpu()
    # fixed 0..1 axis: without range torch spans min..max of the data, so source and result
    # were stretched differently and not comparable (debug round 2026-10-17)
    counts, _ = torch.histogram(L.flatten(), bins=HIST_BINS, range=(0.0, 1.0))
    total = counts.sum().clamp_min(1.0)
    return (counts / total).tolist()


def _histogram_rgb(img: torch.Tensor) -> dict[str, list[float]]:
    """128-Bin-Histogramme für R, G, B und L (Rec.709, gamma), je Summe 1."""
    r = img[..., 0].cpu()
    g = img[..., 1].cpu()
    b = img[..., 2].cpu()
    l = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 1.0)

    def _hist(t: torch.Tensor) -> list[float]:
        counts, _ = torch.histogram(t.flatten().clamp(0.0, 1.0), bins=RGB_HIST_BINS, range=(0.0, 1.0))
        total = counts.sum().clamp_min(1.0)
        return (counts / total).tolist()

    return {"r": _hist(r), "g": _hist(g), "b": _hist(b), "l": _hist(l)}


def preview(
    s: Session,
    refs: list[str],
    method: str,
    values: dict,
    max_side: int,
    mix: dict | None = None,
    stage: str = "full",
    timing: dict | None = None,
    lut_size: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Vorschau: Bild ≤ max_side + Histogramme.

    Args:
        s: Session.
        refs: Liste der Vorlagenpfade.
        method: Methodenname.
        values: Reglerwerte.
        max_side: Maximale längste Seite.
        mix: optional Mix-Param.
        stage: "lut" = nur LUT (ohne Clarity), "full" = komplett (mit Clarity).
        timing: Optionaler Dict, in den Teilzeiten (ms) geschrieben werden.
        lut_size: K31b-6: 17 beim Ziehen, 33 sonst (None → 33).

    Returns:
        (bild [1,H,W,3] float32, histogramme dict).
    """
    import time as _time
    vals = values_with_defaults(values)
    if lut_size is None:
        lut_size = LUT_SIZE

    img = _preview_source(s, max_side)

    if is_identity(vals, mix):
        result = img
        if timing is not None:
            timing["lut_method_fn"] = 0.0
            timing["lut_section_fns"] = 0.0
            timing["lut_build"] = 0.0
            timing["lut_total"] = 0.0
            timing["lut_apply"] = 0.0
    else:
        if timing is not None:
            _t_lut = _time.perf_counter()
        lut = build_lut(s, refs, method, vals, mix, timing=timing, lut_size=lut_size)
        if timing is not None:
            synchronize(s.source.device)
            timing["lut_total"] = (_time.perf_counter() - _t_lut) * 1000.0
        _t_apply = _time.perf_counter()
        result = _lut_apply(img, lut)
        if timing is not None:
            synchronize(s.source.device)
            timing["lut_apply"] = (_time.perf_counter() - _t_apply) * 1000.0

    # Spatial effects (not in the LUT); apply after (full stage only)
    if timing is not None:
        _t_spatial = _time.perf_counter()
    if stage == "full":
        full_size = max(s.source_full.shape[1], s.source_full.shape[2])
        reconstruct_val = vals.get("highlight_reconstruct", 0.0)
        if reconstruct_val != 0.0:
            result = reconstruct_apply(result, reconstruct_val)
        clarity_val = vals.get("clarity", 0.0)
        if clarity_val != 0.0:
            result = clarity_apply(result, clarity_val, full_size)
        texture_val = vals.get("texture", 0.0)
        if texture_val != 0.0:
            result = texture_apply(result, texture_val, full_size)
        dehaze_val = vals.get("dehaze", 0.0)
        if dehaze_val != 0.0:
            result = dehaze_apply(result, dehaze_val, full_size)
    if timing is not None:
        synchronize(s.source.device)
        timing["spatial"] = (_time.perf_counter() - _t_spatial) * 1000.0

    from .scope import scope_img
    _scope_small = scope_img(result)
    if timing is not None:
        _t_hist = _time.perf_counter()
    src_key, ref_key = ("source", max_side), ("reference", tuple(refs))
    if src_key not in s.hists:
        s.hists[src_key] = _histogram_l(img)
    if ref_key not in s.hists:
        if method == "pair":
            before, _ = _resolve_pair(s, refs)
            s.hists[ref_key] = _histogram_l(before)
        else:
            s.hists[ref_key] = _histogram_l(_pool_refs(s, refs))
    rgb_src_key = ("rgb_source", max_side)
    if rgb_src_key not in s.hists:
        s.hists[rgb_src_key] = _histogram_rgb(img)
    hists = {
        "source": s.hists[src_key],
        "reference": s.hists[ref_key],
        "result": _histogram_l(_scope_small),
        "rgb": {
            "source": s.hists[rgb_src_key],
            "result": _histogram_rgb(_scope_small),
        },
    }
    if timing is not None:
        synchronize(s.source.device)
        timing["histograms"] = (_time.perf_counter() - _t_hist) * 1000.0
    if method == "pair":
        _, pair_error = _get_pair_lut(s, refs, LUT_SIZE)
        hists["pair_error"] = pair_error
        hists["pair_shift"] = _get_pair_shift(s, refs)
    return result, hists


def _preview_source(s: Session, max_side: int) -> torch.Tensor:
    """Source scaled to ≤ max_side (cached per size)."""
    if max_side in s.previews:
        return s.previews[max_side]
    h, w = s.source.shape[1], s.source.shape[2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = F.interpolate(
            s.source.permute(0, 3, 1, 2), size=(new_h, new_w),
            mode="area",
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    else:
        img = s.source
    s.previews[max_side] = img
    return img


def _lut_apply(img: torch.Tensor, lut: dict) -> torch.Tensor:
    """LUT anwenden (import hier, um Zirkularität zu vermeiden)."""
    from .lut import apply as lut_apply
    return lut_apply(img, lut)


def apply_full(s: Session, refs: list[str], method: str, values: dict,
               mix: dict | None = None) -> torch.Tensor:
    """Volle Auflösung: LUT in Stücken ≤ APPLY_CHUNK Pixeln.

    Args:
        s: Session.
        refs: Liste der Vorlagenpfade.
        method: Methodenname.
        values: Reglerwerte.
        mix: optional Mix-Param.

    Returns:
        [1,H,W,3] float32 0..1 (CPU).
    """
    vals = values_with_defaults(values)
    if is_identity(vals, mix):
        result = s.source_full
    else:
        lut = build_lut(s, refs, method, vals, mix)
        img = s.source_full
        B, H, W, C = img.shape
        total = H * W

        if total <= APPLY_CHUNK:
            result = _lut_apply(img, lut)
        else:
            out = torch.empty_like(img)
            rows_done = 0
            while rows_done < H:
                chunk_rows = min(H - rows_done, max(1, APPLY_CHUNK // W))
                chunk = img[:, rows_done:rows_done + chunk_rows, :, :]
                out[:, rows_done:rows_done + chunk_rows, :, :] = _lut_apply(chunk, lut)
                rows_done += chunk_rows
            result = out

    # Spatial effects (not in the LUT); apply after
    full_size = max(s.source_full.shape[1], s.source_full.shape[2])
    reconstruct_val = vals.get("highlight_reconstruct", 0.0)
    if reconstruct_val != 0.0:
        result = reconstruct_apply(result, reconstruct_val)
    clarity_val = vals.get("clarity", 0.0)
    if clarity_val != 0.0:
        result = clarity_apply(result, clarity_val, full_size)
    texture_val = vals.get("texture", 0.0)
    if texture_val != 0.0:
        result = texture_apply(result, texture_val, full_size)
    dehaze_val = vals.get("dehaze", 0.0)
    if dehaze_val != 0.0:
        result = dehaze_apply(result, dehaze_val, full_size)
    return result


def fit(s: Session, refs: list[str], method: str) -> dict:
    """Regler fit: nur FITTABLE (18: tone, color, wheels), Rest auf Standard.

    Args:
        s: Session.
        refs: Liste der Vorlagenpfade.
        method: Methodenname.

    Returns:
        dict mit allen Reglernamen → float (reference-Regler auf Standard).

    Raises:
        ValueError: wenn method "pair" ist (FIT nicht verfügbar).
    """
    if method == "pair":
        raise ValueError("FIT is not available for PAIR method")
    reference = _pool_refs(s, refs)
    fitted, _ = _fit_sliders(s.source, reference, s.ignore, None,
                             names=list(FITTABLE))
    return values_with_defaults(fitted)


def rank(s: Session, pool_dir: str, k: int = 9, exclude: tuple = ()) -> list[tuple[str, float]]:
    """Passendste Vorlagen aus einem Ordner.

    Args:
        s: Session.
        pool_dir: Ordner mit Vorlagen.
        k: Maximale Anzahl Ergebnisse.
        exclude: Pfade, die weggelassen werden.

    Returns:
        Liste von (Pfad, Score), absteigend, höchstens k Einträge.
    """
    all_results = _rank(s.source, pool_dir, k=100)
    excluded = set(exclude)
    filtered = [(p, score) for p, score in all_results if p not in excluded]
    return filtered[:k]


def thumbnail(path: str, out: str, side: int = 256) -> str:
    """Quadratischer Mittelausschnitt als PNG.

    Args:
        path: Pfad zur Quelldatei.
        out: Zielpfad für das PNG.
        side: Kantenlänge.

    Returns:
        Zielpfad.
    """
    from PIL import Image, ImageOps

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((side, side), Image.LANCZOS)
    import numpy as np
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    rawio.save_png(tensor, out)
    return out


def crop(
    s: Session,
    refs: list[str],
    method: str,
    values: dict,
    rect: tuple[int, int, int, int],
    max_side: int,
    mix: dict | None = None,
) -> tuple[torch.Tensor, float]:
    """Ausschnitt des vollen Bildes (Koordinaten im Vollbild).

    Args:
        s: Session.
        refs: Vorlagenpfade.
        method: Methodenname.
        values: Reglerwerte.
        rect: (x, y, w, h) im Vollbild.
        max_side: Maximale längste Seite der Ausgabe.
        mix: optional Mix-Param.

    Returns:
        (bild [1,H,W,3] float32, scale: Ausgabepixel je Vollbildpixel).

    Raises:
        ValueError: wenn das geklemmte Rechteck leer ist.
    """
    x, y, w, h = rect
    H, W = s.source_full.shape[1], s.source_full.shape[2]
    # Outward to whole pixels: the window sends float rects in full-image coordinates
    x0 = max(0, math.floor(x))
    y0 = max(0, math.floor(y))
    x1 = min(math.ceil(x + w), W)
    y1 = min(math.ceil(y + h), H)
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        raise ValueError("Rechteck liegt außerhalb des Bildes")

    region = s.source_full[:, y0:y1, x0:x1, :]

    vals = values_with_defaults(values)
    if is_identity(vals, mix):
        result = region
    else:
        lut = build_lut(s, refs, method, vals, mix)
        result = _lut_apply(region, lut)

    # Spatial effects (not in the LUT); apply after (at full resolution)
    full_size = max(H, W)
    reconstruct_val = vals.get("highlight_reconstruct", 0.0)
    if reconstruct_val != 0.0:
        result = reconstruct_apply(result, reconstruct_val)
    clarity_val = vals.get("clarity", 0.0)
    if clarity_val != 0.0:
        result = clarity_apply(result, clarity_val, full_size, scale=1.0)   # region is 1:1
    texture_val = vals.get("texture", 0.0)
    if texture_val != 0.0:
        result = texture_apply(result, texture_val, full_size, scale=1.0)
    dehaze_val = vals.get("dehaze", 0.0)
    if dehaze_val != 0.0:
        result = dehaze_apply(result, dehaze_val, full_size, scale=1.0)

    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        out_h = int(h * scale)
        out_w = int(w * scale)
        result = F.interpolate(
            result.permute(0, 3, 1, 2), size=(out_h, out_w),
            mode="area",
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)

    return result, scale


def cube_path(directory: str, reference_path: str) -> Path:
    """Eindeutigen Cube-Pfad erzeugen: Stamm der Vorlage, nie überschreiben.

    Args:
        directory: Zielordner.
        reference_path: Pfad zur Vorlage (für den Namen).

    Returns:
        Pfad zur neuen .cube-Datei.
    """
    stem = Path(reference_path).stem
    clean = _INVALID_CHARS.sub("_", stem)
    base = Path(directory)
    candidate = base / f"{clean}.cube"
    counter = 2
    while candidate.exists():
        candidate = base / f"{clean}_{counter:02d}.cube"
        counter += 1
    return candidate


def write_cube(
    s: Session,
    refs: list[str],
    method: str,
    values: dict,
    directory: str,
    mix: dict | None = None,
    name: str | None = None,
) -> Path:
    """LUT als .cube schreiben. `name`: Dateiname aus dem Speichern-Dialog (der Nutzer hat ihn
    gewählt und ggf. das Überschreiben bestätigt); ohne ihn ein neuer Name nach der Vorlage.

    Args:
        s: Session.
        refs: Liste der Vorlagenpfade.
        method: Methodenname.
        values: Reglerwerte.
        directory: Zielordner.
        mix: optional Mix-Param.

    Returns:
        Pfad zur geschriebenen .cube-Datei.
    """
    lut = build_lut(s, refs, method, values, mix)
    ref_name = Path(refs[0]).stem
    lut["title"] = f"REF Grade · {ref_name}"
    if name:
        path = Path(directory) / (_INVALID_CHARS.sub("_", Path(name).stem) + ".cube")
    else:
        path = cube_path(directory, refs[0])
    lut_write_cube(lut, str(path))
    return path
