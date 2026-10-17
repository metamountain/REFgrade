"""HTTP-Worker um refgrade.session: Endpunkte, Sitzungs-LRU, GPU-Sperre."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from refgrade import auto
from refgrade import session as sess
from refgrade import whitebalance
from refgrade import version
from refgrade.curves import validate as curves_validate
from refgrade.device import pick_device
from refgrade.help import HELP
from refgrade.rawio import save_png, write_raw
from refgrade.sliders import SLIDERS

logger = logging.getLogger("RefGrade")

_MAX_SESSIONS = 2
_sessions: OrderedDict[str, sess.Session] = OrderedDict()
_sessions_lock = threading.Lock()
_gpu_lock = threading.Lock()
_tmp_dir: str = ""


def _get_session(sid: str) -> sess.Session:
    """Session aus dem LRU holen; KeyError bei unbekannt."""
    with _sessions_lock:
        if sid not in _sessions:
            raise KeyError(sid)
        _sessions.move_to_end(sid)
        return _sessions[sid]


def _put_session(s: sess.Session) -> str:
    """Session ins LRU eintragen, älteste verwerfen wenn > _MAX_SESSIONS."""
    sid = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[sid] = s
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
    return sid


def _save_png_tmp(img: torch.Tensor, prefix: str) -> str:
    """Anzeigebild in den Temp-Ordner schreiben, Pfad zurückgeben.

    BMP statt PNG: die Datei liest das Plugin genau einmal (QImage erkennt das Format am
    Inhalt); PNG kodieren kostete 50–100 ms je Vorschau, BMP ~2 ms (2026-10-11).
    """
    import numpy as np
    from PIL import Image
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.bmp"
    path = os.path.join(_tmp_dir, name)
    arr = (img.squeeze(0).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path, format="BMP")
    return path


def _save_parade_pngs(parade: dict[str, torch.Tensor]) -> dict[str, str]:
    """Parade-Kanäle als fertig gefärbte Neon-RGB-Bilder (256×256, BMP) speichern, Pfade zurückgeben.

    Der Leucht-Effekt (`scope.parade_neon`, Nutzerwahl „B“) braucht Unschärfe in mehreren Weiten —
    im Plugin ohne numpy nicht machbar, darum hier gerechnet; das Plugin zeichnet nur noch.
    """
    import numpy as np
    from PIL import Image
    from refgrade.scope import parade_neon_all

    paths: dict[str, str] = {}
    neon = parade_neon_all({k: parade[k] for k in ("r", "g", "b")})
    for key in ("r", "g", "b"):
        arr = (neon[key] * 255.0).round().to(torch.uint8).cpu().numpy()
        name = f"parade_{key}_{uuid.uuid4().hex[:8]}.bmp"   # BMP: ~1 ms statt ~10 ms je Kanal
        path = os.path.join(_tmp_dir, name)
        Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path, format="BMP")
        paths[key] = path
    return paths


def _save_wave(wave: torch.Tensor) -> str:
    """Luma-Waveform als Neon-Bild (weiß, Glow wie die Parade) speichern, Pfad zurückgeben."""
    import numpy as np
    from PIL import Image
    from refgrade.scope import WAVE_COLOR, parade_neon_all

    rgb = parade_neon_all({"l": wave}, {"l": WAVE_COLOR})["l"]
    arr = (rgb * 255.0).round().to(torch.uint8).cpu().numpy()
    path = os.path.join(_tmp_dir, f"wave_{uuid.uuid4().hex[:8]}.bmp")
    Image.fromarray(np.ascontiguousarray(arr), mode="RGB").save(path, format="BMP")
    return path


def _rank_score(session: sess.Session, path: str) -> float:
    """Score eines einzelnen Bildes zur Session (0..1)."""
    results = sess.rank(session, os.path.dirname(path), k=100)
    for p, score in results:
        if os.path.normcase(p) == os.path.normcase(path):
            return score
    return 0.0


def _validate_mix(mix: dict | None) -> None:
    """Mix-Validierung: Kurven-Format prüfen.

    Raises:
        _BadRequest: bei ungültigem Kurven-Format.
    """
    if mix and "curves" in mix:
        try:
            curves_validate(mix["curves"])
        except ValueError as e:
            raise _BadRequest(str(e))


class Handler(BaseHTTPRequestHandler):
    """HTTP-Handler mit JSON-Antworten."""

    def log_message(self, fmt: str, *args) -> None:
        pass  # Standard-Logging deaktivieren, wir loggen selbst

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _log_request(self, status: int, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"{self.path} {status} {ms:.0f}ms", flush=True)

    # ── GET ────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        t0 = time.perf_counter()
        try:
            if self.path == "/health":
                device = pick_device().type
                self._send_json(200, {"ok": True, "version": version.label(), "device": device,
                                      "auto_sections": list(auto.AUTO_SECTIONS)})
                self._log_request(200, t0)
            elif self.path == "/sliders":
                items = [
                    {"name": s.name, "group": s.group, "min": s.min,
                     "max": s.max, "default": s.default, "step": s.step, "core": s.core}
                    for s in SLIDERS
                ]
                self._send_json(200, {"sliders": items, "help": HELP})
                self._log_request(200, t0)
            else:
                self._send_json(404, {"error": "Pfad unbekannt"})
                self._log_request(404, t0)
        except Exception:
            logger.exception("GET %s", self.path)
            self._send_json(500, {"error": f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}"})
            self._log_request(500, t0)

    # ── POST ───────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        t0 = time.perf_counter()
        try:
            body = self._read_body() if self.path != "/shutdown" else {}

            if self.path == "/load":
                self._handle_load(body, t0)
            elif self.path == "/references":
                self._handle_references(body, t0)
            elif self.path == "/thumb":
                self._handle_thumb(body, t0)
            elif self.path == "/preview":
                self._handle_preview(body, t0)
            elif self.path == "/crop":
                self._handle_crop(body, t0)
            elif self.path == "/prerender":
                self._handle_prerender(body, t0)
            elif self.path == "/whitebalance":
                self._handle_whitebalance(body, t0)
            elif self.path == "/auto":
                self._handle_auto(body, t0)
            elif self.path == "/fit":
                self._handle_fit(body, t0)
            elif self.path == "/apply":
                self._handle_apply(body, t0)
            elif self.path == "/cube":
                self._handle_cube(body, t0)
            elif self.path == "/shutdown":
                self._send_json(200, {"ok": True})
                self._log_request(200, t0)
                threading.Thread(target=os._exit, args=(0,), daemon=True).start()
            else:
                self._send_json(404, {"error": "Pfad unbekannt"})
                self._log_request(404, t0)
        except _BadRequest as e:
            self._send_json(400, {"error": str(e)})
            self._log_request(400, t0)
        except KeyError:
            self._send_json(409, {"error": "Sitzung unbekannt"})
            self._log_request(409, t0)
        except Exception:
            logger.exception("POST %s", self.path)
            self._send_json(500, {"error": f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}"})
            self._log_request(500, t0)

    # ── Endpunkt-Implementierungen ─────────────────────────────────────────

    def _handle_load(self, body: dict, t0: float) -> None:
        raw = body.get("raw")
        width = body.get("width")
        height = body.get("height")
        depth = body.get("depth")
        if raw is None or width is None or height is None or depth is None:
            raise _BadRequest("Felder raw, width, height, depth sind erforderlich")
        ignore_mask = body.get("ignore_mask")
        with _gpu_lock:
            s = sess.open_session(raw, width, height, depth, ignore_mask)
        sid = _put_session(s)
        self._send_json(200, {"session": sid, "width": width, "height": height, "depth": depth})
        self._log_request(200, t0)

    def _handle_references(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        pool = body.get("pool")
        if sid is None or pool is None:
            raise _BadRequest("Felder session und pool sind erforderlich")
        k = body.get("k", 6)
        exclude = body.get("exclude", [])
        s = _get_session(sid)
        # Außerhalb der GPU-Sperre (liest von der Platte)
        slow = os.environ.get("REFGRADE_TEST_SLOW_RANK")
        if slow:
            time.sleep(float(slow))
        results = sess.rank(s, pool, k=k, exclude=tuple(exclude))
        items = []
        for path, score in results:
            thumb_name = f"ref_{uuid.uuid4().hex[:8]}.png"
            thumb_path = os.path.join(_tmp_dir, thumb_name)
            sess.thumbnail(path, thumb_path, side=256)
            items.append({
                "path": path,
                "name": Path(path).stem,
                "score": score,
                "thumb": thumb_path,
            })
        self._send_json(200, {"items": items})
        self._log_request(200, t0)

    def _handle_thumb(self, body: dict, t0: float) -> None:
        path = body.get("path")
        if path is None:
            raise _BadRequest("Feld path ist erforderlich")
        sid = body.get("session")
        thumb_name = f"thumb_{uuid.uuid4().hex[:8]}.png"
        thumb_path = os.path.join(_tmp_dir, thumb_name)
        sess.thumbnail(path, thumb_path, side=256)
        resp: dict = {"thumb": thumb_path}
        if sid is not None:
            s = _get_session(sid)
            resp["score"] = _rank_score(s, path)
        self._send_json(200, resp)
        self._log_request(200, t0)

    def _handle_preview(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        refs = body.get("references")
        method = body.get("method")
        values = body.get("values", {})
        max_side = body.get("max_side")
        mix = body.get("mix")
        stage = body.get("stage", "full")
        scopes = body.get("scopes")  # K31b-D: list of scope names to compute
        lut_size = body.get("lut_size")  # K31b-6: 17 during drag, 33 on release
        clip_view = bool(body.get("clip_view", False))
        want_timing = bool(body.get("timing"))
        if sid is None or refs is None or method is None or max_side is None:
            raise _BadRequest("Felder session, references, method, max_side sind erforderlich")
        _validate_mix(mix)
        s = _get_session(sid)
        timing: dict | None = {} if want_timing else None
        with _gpu_lock:
            try:
                img, hists = sess.preview(s, refs, method, values, max_side, mix,
                                          stage=stage, timing=timing, lut_size=lut_size)
            except KeyError as e:
                raise _BadRequest(str(e))
            except ValueError as e:
                if str(e).startswith("LUT not readable"):
                    raise _BadRequest(str(e))
                raise
        from refgrade.scope import vectorscope, rgb_parade, clipping, luma_waveform, scope_img as _scope_img, clip_overlay
        from refgrade.session import _histogram_l, _histogram_rgb
        # K31b-D: only compute requested scopes (histogram always)
        want_vs = scopes is None or "vectorscope" in scopes
        want_parade = scopes is None or "parade" in scopes
        small = _scope_img(img)  # scopes always from the original image
        if clip_view:
            img = clip_overlay(img)
        if timing is not None:
            _t = time.perf_counter()
        image_path = _save_png_tmp(img, "preview")
        if timing is not None:
            timing["png_write"] = (time.perf_counter() - _t) * 1000.0
            _t = time.perf_counter()
        vs_result = vectorscope(small) if want_vs else None
        # the original never changes within a session: once, not on every slider move
        # (2026-10-11: ~300 ms per preview at full resolution)
        if want_vs and ("vs_source", None) not in s.hists:
            s.hists[("vs_source", None)] = vectorscope(_scope_img(s.source))
        vs_source = s.hists.get(("vs_source", None)) if want_vs else None
        parade_result = rgb_parade(small) if want_parade else None
        wave_result = luma_waveform(small) if want_parade else None
        clip_result = clipping(small)
        if timing is not None:
            timing["scopes"] = (time.perf_counter() - _t) * 1000.0
            _t = time.perf_counter()
        parade_paths = _save_parade_pngs(parade_result) if want_parade else {}
        wave_path = _save_wave(wave_result) if want_parade else ""
        if timing is not None:
            timing["scope_save"] = (time.perf_counter() - _t) * 1000.0
            _t = time.perf_counter()
        resp = {"image": image_path, "histogram": hists, "ms": (time.perf_counter() - t0) * 1000.0,
                "clip": clip_result}
        if want_vs:
            resp["vectorscope"] = vs_result
            resp["vectorscope_source"] = vs_source
        if want_parade:
            resp["parade"] = parade_paths
            resp["wave"] = wave_path
        if "pair_error" in hists:
            resp["pair_error"] = hists.pop("pair_error")
        if "pair_shift" in hists:
            resp["pair_shift"] = hists.pop("pair_shift")
        if timing is not None:
            timing["json"] = (time.perf_counter() - _t) * 1000.0
            resp["timing"] = timing
        self._send_json(200, resp)
        self._log_request(200, t0)

    def _handle_crop(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        refs = body.get("references")
        method = body.get("method")
        values = body.get("values", {})
        rect = body.get("rect")
        max_side = body.get("max_side")
        mix = body.get("mix")
        if sid is None or refs is None or method is None or rect is None or max_side is None:
            raise _BadRequest("Felder session, references, method, rect, max_side sind erforderlich")
        if len(rect) != 4:
            raise _BadRequest("rect muss [x, y, w, h] sein")
        _validate_mix(mix)
        s = _get_session(sid)
        with _gpu_lock:
            try:
                img, scale = sess.crop(s, refs, method, values, tuple(rect), max_side, mix)
            except ValueError as e:
                raise _BadRequest(str(e))
            except KeyError as e:
                raise _BadRequest(str(e))
        image_path = _save_png_tmp(img, "crop")
        from refgrade.scope import vectorscope, rgb_parade
        vs_result = vectorscope(img)
        parade_result = rgb_parade(img)
        # the same clamped full-resolution window as sess.crop: rect is float, in full-image
        # coordinates — s.source is the downscaled preview (wrong area / TypeError before)
        x, y, w, h = rect
        H, W = s.source_full.shape[1], s.source_full.shape[2]
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(math.ceil(x + w), W), min(math.ceil(y + h), H)
        source_crop = s.source_full[:, y0:y1, x0:x1, :]
        vs_source = vectorscope(source_crop)
        self._send_json(200, {"image": image_path, "scale": scale,
                              "vectorscope": vs_result, "vectorscope_source": vs_source,
                              "parade": _save_parade_pngs(parade_result)})
        self._log_request(200, t0)

    def _handle_prerender(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        reference = body.get("reference")
        method = body.get("method")
        values = body.get("values", {})
        max_side = body.get("max_side")
        mix = body.get("mix")
        if sid is None or reference is None or method is None or max_side is None:
            raise _BadRequest("Felder session, reference, method, max_side sind erforderlich")
        _validate_mix(mix)
        s = _get_session(sid)
        with _gpu_lock:
            try:
                img, _ = sess.preview(s, [reference], method, values, max_side, mix)
            except KeyError as e:
                raise _BadRequest(str(e))
            except ValueError as e:
                if str(e).startswith("LUT not readable"):
                    raise _BadRequest(str(e))
                raise
        image_path = _save_png_tmp(img, "prerender")
        self._send_json(200, {"image": image_path})
        self._log_request(200, t0)

    def _handle_fit(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        refs = body.get("references")
        method = body.get("method")
        if sid is None or refs is None or method is None:
            raise _BadRequest("Felder session, references, method sind erforderlich")
        if method == "pair":
            raise _BadRequest("FIT is not available for PAIR method")
        s = _get_session(sid)
        with _gpu_lock:
            values = sess.fit(s, refs, method)
        self._send_json(200, {"values": values})
        self._log_request(200, t0)

    def _handle_whitebalance(self, body: dict, t0: float) -> None:
        """{session, mode: "auto" | "pick", point?: [x, y] (full-image px)} → {temperature, tint}."""
        sid, mode = body.get("session"), body.get("mode")
        if sid is None or mode not in ("auto", "pick"):
            raise _BadRequest("Felder session und mode (auto|pick) sind erforderlich")
        s = _get_session(sid)
        if mode == "auto":
            with _gpu_lock:
                values = whitebalance.estimate(s.source)
        else:
            point = body.get("point")
            if not (isinstance(point, list) and len(point) == 2):
                raise _BadRequest("mode pick braucht point [x, y]")
            try:
                values = whitebalance.sample(s.source_full, float(point[0]), float(point[1]))
            except ValueError as e:
                raise _BadRequest(str(e)) from e
        self._send_json(200, values)
        self._log_request(200, t0)

    def _handle_auto(self, body: dict, t0: float) -> None:
        """{session, section, values[, references, method, mix]} → {values: {...}} oder {curves: {...}}.

        AUTO sieht das Bild, wie es beim Bereich in der Kette ankommt (2026-10-13): references,
        method und mix wie bei /preview; fehlen sie, zählt nur dieser Bereich auf dem Original.
        """
        sid = body.get("session")
        section = body.get("section")
        values = body.get("values", {})
        refs = body.get("references") or []
        method = body.get("method") or "reinhard"
        mix = body.get("mix")
        if sid is None or section is None:
            raise _BadRequest("Felder session und section sind erforderlich")
        _validate_mix(mix)
        s = _get_session(sid)
        with _gpu_lock:
            try:
                src = sess._preview_source(s, auto.STAT_SIDE)
                chain_in = sess.chain_input(s, refs, method, values, mix, section, src)
                result = auto.auto(chain_in, section)
            except ValueError as e:
                raise _BadRequest(str(e))
        if section == "curves":
            self._send_json(200, result)
        else:
            self._send_json(200, {"values": result})
        self._log_request(200, t0)

    def _handle_apply(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        refs = body.get("references")
        method = body.get("method")
        values = body.get("values", {})
        out = body.get("out")
        mix = body.get("mix")
        if sid is None or refs is None or method is None or out is None:
            raise _BadRequest("Felder session, references, method, out sind erforderlich")
        _validate_mix(mix)
        s = _get_session(sid)
        with _gpu_lock:
            try:
                # Apply without clarity (delivered separately as filter mask params)
                values_no_clarity = dict(values)
                values_no_clarity["clarity"] = 0.0
                result = sess.apply_full(s, refs, method, values_no_clarity, mix)
            except KeyError as e:
                raise _BadRequest(str(e))
            except ValueError as e:
                if str(e).startswith("LUT not readable"):
                    raise _BadRequest(str(e))
                raise
            write_raw(out, result, s.alpha, s.depth)

        resp: dict = {"raw": out, "ms": (time.perf_counter() - t0) * 1000.0}
        clarity_val = values.get("clarity", 0.0)
        if clarity_val != 0.0:
            from refgrade.clarity import CLARITY_AMOUNT, CLARITY_RADIUS
            from refgrade.color import rgb_to_lab
            H, W = s.source_full.shape[1], s.source_full.shape[2]
            full_size = max(H, W)
            half_size = CLARITY_RADIUS * full_size
            amount = CLARITY_AMOUNT * clarity_val
            # Midtone mask: round(255 * 4L(1-L)) of result (after LUT, before clarity)
            lab = rgb_to_lab(result)
            L01 = lab[..., 0] / 100.0
            mask = torch.round(255.0 * 4.0 * L01 * (1.0 - L01)).clamp(0, 255).to(torch.uint8)
            mask_path = os.path.join(_tmp_dir, f"clarity_mask_{uuid.uuid4().hex[:8]}.raw")
            mask.squeeze().cpu().numpy().tofile(mask_path)
            resp["clarity"] = {"halfSize": half_size, "amount": amount, "mask": mask_path}
        self._send_json(200, resp)
        self._log_request(200, t0)

    def _handle_cube(self, body: dict, t0: float) -> None:
        sid = body.get("session")
        refs = body.get("references")
        method = body.get("method")
        values = body.get("values", {})
        directory = body.get("dir")
        mix = body.get("mix")
        if sid is None or refs is None or method is None or directory is None:
            raise _BadRequest("Felder session, references, method, dir sind erforderlich")
        _validate_mix(mix)
        s = _get_session(sid)
        with _gpu_lock:
            try:
                path = sess.write_cube(s, refs, method, values, directory, mix, body.get("name"))
            except KeyError as e:
                raise _BadRequest(str(e))
            except ValueError as e:
                if str(e).startswith("LUT not readable"):
                    raise _BadRequest(str(e))
                raise
        self._send_json(200, {"path": str(path)})
        self._log_request(200, t0)


class _BadRequest(Exception):
    """400-Fehler mit deutscher Meldung."""


def main() -> None:
    parser = argparse.ArgumentParser(description="REF Grade Worker")
    parser.add_argument("--port", type=int, default=47123)
    parser.add_argument("--tmp", type=str, default=os.path.join(os.environ.get("TEMP", "/tmp"), "ref_grade"))
    args = parser.parse_args()

    global _tmp_dir
    _tmp_dir = args.tmp
    os.makedirs(_tmp_dir, exist_ok=True)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("Worker lauft auf 127.0.0.1:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
