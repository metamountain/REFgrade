"""WorkerMixin: HTTP task execution, preview, crop, apply, fit (K26)."""

from __future__ import annotations

import math
import time
from pathlib import Path

from PyQt6.QtCore import (QEvent, QObject, QRectF, QRunnable, QThreadPool,
                          QTimer, Qt, pyqtSignal)
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QFileDialog

from . import config, scale, style, wheels
from .widgets import Parade, Vectorscope
from .dlg_common import _DEBOUNCE_MS, _CROP_IDLE_MS

class _ResultSignal(QObject):
    """Thread-safe signal: (req_id, result_dict_or_None, error_str_or_None)."""
    result = pyqtSignal(int, object, object)



class _WorkerTask(QRunnable):
    """Runs one client call in a thread; emits the result back to the GUI thread."""

    def __init__(self, client, port: int, path: str, payload: dict, req_id: int,
                 sig: _ResultSignal):
        super().__init__()
        self._client = client
        self._port = port
        self._path = path
        self._payload = payload
        self._req_id = req_id
        self._sig = sig

    def run(self) -> None:
        try:
            result = self._client(self._port, self._path, self._payload)
            self._sig.result.emit(self._req_id, result, None)
        except Exception as e:
            self._sig.result.emit(self._req_id, None, str(e))



def _histogram_of(image: QImage) -> list[float]:
    """64-bin luminance histogram of a 128-px copy (display only, Rec.709 luma)."""
    small = image.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    bins = [0.0] * 64
    for y in range(small.height()):
        for x in range(small.width()):
            c = QColor(small.pixel(x, y))
            bins[min(63, int((0.2126 * c.red() + 0.7152 * c.green() + 0.0722 * c.blue()) / 4))] += 1
    total = sum(bins) or 1.0
    return [b / total for b in bins]




class WorkerMixin:
    """Mixin for RefGradeDialog: HTTP task execution, preview, crop, apply, fit."""

    @staticmethod
    def _fmt_clip(clip: dict) -> str:
        """Stärkster Kanal je Richtung in %, eine Nachkommastelle unter 10; unter 0,1 % weglassen."""
        def _max(ch: str) -> float:
            return max(clip.get(f"{c}_{ch}", 0.0) for c in "rgb")
        parts = []
        hi = _max("hi")
        lo = _max("lo")
        if hi >= 0.001:
            p = hi * 100.0
            parts.append(f"▲{p:.1f}" if p < 10 else f"▲{p:.0f}")
        if lo >= 0.001:
            p = lo * 100.0
            parts.append(f"▼{p:.1f}" if p < 10 else f"▼{p:.0f}")
        return " ".join(parts)

    def _values(self) -> dict:
        """All slider values + wheel values as one dict (amount_* excluded: they are weights)."""
        vals: dict[str, float] = {}
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                if r.name != "master" and not r.name.startswith("amount_"):
                    vals[r.name] = r.value()
        # HSL: all 18 sel_* values (stored in _hsl_sliders, not in sections rows)
        for band, (hue, sat, light) in self._hsl_sliders.items():
            vals[hue.name] = hue.value()
            vals[sat.name] = sat.value()
            vals[light.name] = light.value()
        for kind in ("lift", "gamma", "gain"):
            vals.update(self.wheel_values(kind))
        # Color Replace: custom values from the CR section
        if hasattr(self, "_cr_section"):
            vals.update(self._cr_section.values())
        return vals


    def _max_side(self) -> int:
        """Longer side of the image area in device pixels; half during drag (K29c)."""
        r = self.view.devicePixelRatioF()
        full = max(1, max(int(self.view.width() * r), int(self.view.height() * r)))
        return full // 2 if self._dragging else full

    def _lut_size(self) -> int:
        """LUT edge length: 17 during drag (K31b-6), 33 on release / otherwise."""
        return 17 if self._dragging else 33


    def _next_req(self) -> int:
        self._req_counter += 1
        return self._req_counter


    def _run_task(self, path: str, payload: dict, req_id: int, route: str | None = None) -> None:
        """Run a client call in the thread pool. A reply is stale only if a newer request with
        the same route exists (a hover /prerender must not drop the running /preview). `route`
        defaults to the endpoint; "original" is a /preview with nothing on."""
        key = route or path
        self._req_path[req_id] = key
        self._latest[key] = req_id
        # K27e: show loading bar immediately
        op_map = {"/preview": "preview", "/apply": "apply", "/cube": "cube",
                  "/fit": "preview", "/crop": "preview", "/prerender": "preview",
                  "/references": "preview", "wb": "preview", "original": "preview"}
        op = op_map.get(key, "preview")
        mp = (self._orig_w * self._orig_h) / 1_000_000.0 if self._orig_w else 0.5
        self._loading_op = op
        self._loading_mp = mp
        self._start_loading(op, mp)
        task = _WorkerTask(self._client, self._port, path, payload, req_id, self._result_sig)
        QThreadPool.globalInstance().start(task)


    def _schedule_preview(self) -> None:
        """Throttle: fire /preview at most every 30 ms (timer not restarted while active)."""
        if not self._debounce_timer.isActive():
            self._debounce_timer.start()


    def _visible_scopes(self) -> list[str]:
        """Scope names currently visible (K31b-D: skip invisible ones)."""
        scopes: list[str] = ["hist"]
        if getattr(self, "vectorscope", None) is not None and self.vectorscope.isVisible():
            scopes.append("vectorscope")
        parade = getattr(self, "_parade", None)
        wave = getattr(self, "_wave", None)       # the worker computes wave together with parade
        if ((parade is not None and parade.isVisible())
                or (wave is not None and wave.isVisible())):
            scopes.append("parade")
        return scopes

    def _trigger_preview(self) -> None:
        """Send a /preview with stage='lut'; schedule 'full' if clarity is active."""
        if self._client is None or self._session_id is None:
            return
        if self._latest.get("/preview") in self._req_path:
            self._preview_dirty = True
            return
        self._preview_dirty = False
        refs = self._references()
        vals = self._values()
        payload = {
            "session": self._session_id,
            "references": refs,
            "method": self._method(),
            "values": vals,
            "max_side": self._max_side(),
            "lut_size": self._lut_size(),
            "mix": self.mode(),
            "stage": "lut",
            "scopes": self._visible_scopes(),
            "clip_view": self._clip_view,
        }
        req_id = self._next_req()
        self._run_task("/preview", payload, req_id)
        self._crop_timer.stop()
        # Schedule full stage if any spatial effect is active and not dragging
        spatial_active = any(vals.get(n, 0.0) != 0.0
                             for n in ("clarity", "texture", "dehaze", "highlight_reconstruct"))
        self._full_pending = spatial_active and not self._dragging

    def _send_full_preview(self) -> None:
        """Send the full-resolution preview (with clarity) after the lut stage."""
        if self._client is None or self._session_id is None:
            return
        if self._latest.get("/preview") in self._req_path:
            # A preview is still running: send the full stage after its result. Marking the
            # preview dirty here re-triggered a lut stage, which re-armed this timer → endless
            # re-render loop whenever clarity != 0 and a render took > ~90 ms (2026-10-10).
            self._full_pending = True
            return
        refs = self._references()
        payload = {
            "session": self._session_id,
            "references": refs,
            "method": self._method(),
            "values": self._values(),
            "max_side": self._max_side(),
            "lut_size": self._lut_size(),
            "mix": self.mode(),
            "stage": "full",
            "scopes": self._visible_scopes(),
            "clip_view": self._clip_view,
        }
        req_id = self._next_req()
        self._run_task("/preview", payload, req_id)
        self._crop_timer.stop()


    def _trigger_crop(self) -> None:
        """Send /crop for the visible rect when zoom exceeds preview resolution."""
        if self._client is None or self._session_id is None:
            return
        if self.view.zoom <= self._preview_scale:
            return
        refs = self._references()
        if not refs:
            return
        rect = self.view.visible_rect()          # already full-image coordinates
        x0, y0 = math.floor(rect.x()), math.floor(rect.y())       # whole pixels, outward
        x1, y1 = math.ceil(rect.x() + rect.width()), math.ceil(rect.y() + rect.height())
        orig_rect = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        payload = {
            "session": self._session_id,
            "references": refs,
            "method": self._method(),
            "values": self._values(),
            "rect": orig_rect,
            "max_side": self._max_side(),
            "mix": self.mode(),
        }
        self._crop_rect = QRectF(x0, y0, orig_rect[2], orig_rect[3])   # full-image coords
        req_id = self._next_req()
        self._run_task("/crop", payload, req_id)


    def _on_result(self, req_id: int, result, error) -> None:
        """Handle a worker reply: dispatch by endpoint, drop stale replies per endpoint."""
        path = self._req_path.pop(req_id, None)
        if path == "/preview" and self._preview_dirty:
            self._debounce_timer.start()
        if path is None or req_id != self._latest.get(path):
            return
        # K27e: stop loading bar with measured time
        measured_ms = result.get("ms") if isinstance(result, dict) else None
        self._stop_loading(measured_ms)
        if error is not None:
            self._set_mark_error(error)
            return
        if result is None:
            return
        if path and path.startswith("auto_"):
            try:
                self._apply_auto(path[5:], result)
            except Exception as ex:  # noqa: BLE001 - boundary: worker reply → UI
                self._set_mark_error(f"{path}: {type(ex).__name__}: {ex}")
            return
        handler = {"/preview": self._apply_preview, "original": self._apply_original, "/crop": self._apply_crop,
                   "/references": self._apply_references, "/fit": self._apply_fit,
                   "/prerender": self._apply_prerender, "/apply": self._apply_apply,
                   "/cube": self._apply_cube, "wb": self._apply_wb}.get(path)
        if handler:
            # an exception escaping a Qt slot makes PyQt6 abort the whole process — Krita
            # included. Seen in the suite: a reply without "image" (2026-10-11).
            try:
                handler(result)
            except Exception as ex:  # noqa: BLE001 - boundary: worker reply → UI
                self._set_mark_error(f"{path}: {type(ex).__name__}: {ex}")
                return
        # K27f: after a lut-stage preview, send the full stage if pending
        if path == "/preview" and self._full_pending:
            self._full_pending = False
            self._full_timer.start()

    # ── white balance (user 2026-10-08: auto + eyedropper) ──────────────────────

    def _apply_original(self, result: dict) -> None:
        """The layer itself, exactly as the worker sees it — used for press/split 'original'."""
        img = QImage(result.get("image") or "")
        if img.isNull():
            return
        self._original_image = img
        if hasattr(self.view, "set_original"):
            self.view.set_original(img)


    def _apply_preview(self, result: dict) -> None:
        """Display preview image + histogram; the view (zoom/pan) does not move."""
        img = QImage(result.get("image") or "")
        if img.isNull():
            return
        self._last_preview = img
        self._preview_scale = img.width() / self._orig_w if self._orig_w else 1.0
        if not self._worker_original_requested:        # the original must show exactly the
            self._worker_original_requested = True     # layer the preview shows (not the whole
            payload = {"session": self._session_id,    # document): one identity /preview
                       "references": self._references(), "method": self._method(),
                       "values": {}, "max_side": self._max_side(),
                       "mix": {"weights": {}, "master": 0.0}, "purpose": "original"}
            self._run_task("/preview", payload, self._next_req(), route="original")
        if not (self._showing_original or self._hover_tile is not None):
            self.view.show_overlay = True
            self.view.replace_image(img)
        hists = result.get("histogram", {})
        # New RGB histogram data
        if "rgb" in hists:
            self.histogram.set_rgb_data(hists["rgb"])
        else:
            self.histogram.set_data(hists)
        # Vectorscope data (96×96 matrices)
        scope = result.get("vectorscope")
        scope_src = result.get("vectorscope_source")
        if scope is not None:
            self.vectorscope.set_data(scope, scope_src)
        # K27a: RGB Parade data (single instance in scope block)
        parade = result.get("parade")
        if parade is not None:
            self._parade.set_data(parade)
        wave = result.get("wave")
        if wave is not None:
            self._wave.set_data(wave)
            self.histogram.set_wave(wave)
        # K28a: Clipping data
        clip = result.get("clip")
        if clip is not None:
            self.histogram.set_clip_data(clip)
            self._parade.set_clip_data(clip)
            self.view.set_clip_text(self._fmt_clip(clip))
        self._update_mark()
        shift = result.get("pair_shift")
        if shift is not None and shift > 5:
            self.pair_hint.setText(
                f"Before and after do not line up (shift ≈ {int(round(shift))} px) — result may be off.")
            self.pair_hint.setVisible(True)
        else:
            self.pair_hint.setVisible(False)
        self._maybe_schedule_crop()


    def _apply_crop(self, result: dict) -> None:
        """Full-resolution crop: drawn over exactly the area it was requested for; zoom and
        pan stay as they are."""
        img = QImage(result.get("image") or "")
        if img.isNull() or self._crop_rect is None:
            return
        self.view.set_overlay(img, self._crop_rect)
        self._update_mark()


    def _apply_fit(self, result: dict) -> None:
        """Set all sliders and wheels from /fit response, then one preview."""
        values = result["values"]
        bad = [k for k, v in values.items() if not math.isfinite(v)]
        if bad:                                  # a broken fit must not touch any control
            self._set_mark_error(f"FIT returned invalid values ({', '.join(bad[:3])}…)")
            return
        self._block_signals()
        try:
            for key, (_h, rows) in self.sections.items():
                for r in rows:
                    if r.name in values:
                        r.set_value(values[r.name])
            for band, (hue, sat, light) in self._hsl_sliders.items():
                for r in (hue, sat, light):
                    if r.name in values:
                        r.set_value(values[r.name])
            for kind in ("lift", "gamma", "gain"):
                u, v, master, rest = wheels.to_point(kind, values)
                self.wheels[kind].set_point(u, v)
                self.masters[kind].set_value(master)
                self._wheel_rest[kind] = rest
        finally:
            self._unblock_signals()
        # Reference weight → 0 %
        ref_header = self.sections.get("reference", (None, None))[0]
        if ref_header and ref_header.amount_slider is not None:
            ref_header.amount_slider.slider.blockSignals(True)
            ref_header.amount_slider.set_value(0.0)
            ref_header.amount_slider.slider.blockSignals(False)
        self._set_status("FIT · reference set to 0 %")
        self._schedule_preview()


    def _on_auto_requested(self, key: str) -> None:
        """K30c: "A" in a section header → /auto with the same context as /preview."""
        if self._client is None or self._session_id is None:
            return
        self._run_task("/auto", {"session": self._session_id, "section": key,
                                 "references": self._references(),
                                 "method": self._method(),
                                 "values": self._values(),
                                 "mix": self.mode()},
                       self._next_req(), route=f"auto_{key}")


    def _apply_auto(self, key: str, result: dict) -> None:
        """K30c: set the section's sliders (or curves) from the /auto response, turn it ON,
        then trigger exactly one preview. Undo = the section's existing ↺ (reset to default)."""
        bad = [k for k, v in result.get("values", {}).items()
               if isinstance(v, (int, float)) and not math.isfinite(v)]
        if bad:                                  # a broken auto must not touch any control
            self._set_mark_error(f"AUTO {key.upper()} returned invalid values ({', '.join(bad[:3])}…)")
            return
        self._block_signals()
        try:
            if key == "curves":
                for ch, pts in result.get("curves", {}).items():
                    self.curves_editor.set_curve(ch, pts)
            else:
                values = result.get("values", {})
                for name, val in values.items():
                    self._set_slider_value(name, val)
                if key == "wheels":
                    for kind in ("lift", "gamma", "gain"):
                        u, v, master, rest = wheels.to_point(kind, values)
                        self.wheels[kind].set_point(u, v)
                        self.masters[kind].set_value(master)
                        self._wheel_rest[kind] = rest
        finally:
            self._unblock_signals()
        header = self.sections.get(key, (None, None))[0]
        if header is not None and not header.is_on():
            header.on.setChecked(True)
        self._set_status(f"AUTO {key.upper()} applied")
        self._schedule_preview()


    def _apply_prerender(self, result: dict) -> None:
        """Display prerender image (tile hover)."""
        img = QImage(result.get("image") or "")
        if img.isNull():
            return
        self._prerender_image = img
        self.view.show_overlay = False
        self.view.replace_image(img, keep_overlay=True)
        self._update_mark()


    _SAVEAS_FILTERS = {"PNG (*.png)": ".png", "JPEG (*.jpg *.jpeg)": ".jpg",
                       "TIFF (*.tif *.tiff)": ".tif"}

    def _do_saveas(self) -> None:
        """One file dialog (user 2026-10-16: the extra folder step was annoying); the
        folder is remembered, the name defaults to <source>_grad.png."""
        last = self._settings.value("saveas_folder", str(config.ROOT))
        src_path = self._source.get("path") if self._source else None
        stem = Path(src_path).stem if src_path else "result"
        out, flt = QFileDialog.getSaveFileName(
            self, "Save graded image", str(Path(last) / f"{stem}_grad.png"),
            ";;".join(self._SAVEAS_FILTERS))
        if not out:
            return
        target = Path(out)
        if target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            target = target.with_suffix(self._SAVEAS_FILTERS.get(flt, ".png"))
        self._settings.setValue("saveas_folder", str(target.parent))
        self._saveas_out = str(target)
        self._do_ok()

    def _apply_apply(self, result: dict) -> None:
        """Handle /apply response: call on_ok and close."""
        raw = result["raw"]
        if getattr(self, "_saveas_out", None):
            # Qt only: this runs inside Krita's Python, which has neither numpy nor PIL.
            depth = (self._source or {}).get("depth", "U8")
            data = Path(raw).read_bytes()
            w, h = self._orig_w, self._orig_h
            if depth == "U16":   # BGRA 16-bit LE → RGBA64 needs R/B swapped
                img = QImage(data, w, h, w * 8, QImage.Format.Format_RGBA64).rgbSwapped()
            else:                # BGRA 8-bit == ARGB32 in little-endian memory
                img = QImage(data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
            ok = img.save(self._saveas_out, None, 95)
            self._set_status(("Saved: " if ok else "Could not save: ") + self._saveas_out)
            self._saveas_out = None
            return
        clarity_info = result.get("clarity")
        if self._on_ok:
            # clarity_info is an optional trailing argument: only pass it when present,
            # so callbacks that predate Clarity (2-arg) keep working.
            if clarity_info is not None:
                self._on_ok(raw, self._layer_name(), clarity_info)
            else:
                self._on_ok(raw, self._layer_name())
        self.accept()


    def _maybe_schedule_crop(self) -> None:
        """If zoom exceeds preview resolution, schedule a crop after idle."""
        if self.view.zoom > self._preview_scale:
            self._crop_timer.start()
        else:
            self._crop_timer.stop()


    def _do_fit(self) -> None:
        if self._client is None or self._session_id is None:
            return
        refs = self._references()
        if not refs:
            return
        payload = {
            "session": self._session_id,
            "references": refs,
            "method": self._method(),
        }
        req_id = self._next_req()
        self._run_task("/fit", payload, req_id)

    # ── LUT ───────────────────────────────────────────────────────────────────

    def _do_ok(self) -> None:
        if self._client is None or self._session_id is None:
            self.accept()
            return
        refs = self._references()
        out_path = str(config.temp_dir() / "result.raw")
        payload = {
            "session": self._session_id,
            "references": refs,
            "method": self._method(),
            "values": self._values(),
            "out": out_path,
            "mix": self.mode(),
        }
        req_id = self._next_req()
        self._run_task("/apply", payload, req_id)

    # ── Pipette (Ctrl+Click) ─────────────────────────────────────────────────


    def _restore_computed(self) -> None:
        """Back to the current grading after the original or a reference preview; view stays."""
        if self._last_preview is not None:
            self.view.show_overlay = True
            self.view.replace_image(self._last_preview, keep_overlay=True)
        self._update_mark()

    def _layer_name(self) -> str:
        """Layer name for the result (same logic as pixel_rules.layer_name)."""
        from .pixel_rules import layer_name
        refs = self._references()
        return layer_name(refs[0]) if refs else "REF · unnamed"

    # ── K27e: loading bar ───────────────────────────────────────────────────

    def _start_loading(self, op: str, mp: float) -> None:
        """Show the loading bar immediately; estimate from render_est."""
        from . import render_est
        seed = render_est.load_seed()
        a = seed.get(op, {"a": 100.0, "b": 50.0})["a"]
        b = seed.get(op, {"a": 100.0, "b": 50.0})["b"]
        # Check QSettings for learned b
        key = f"render_est_{op}"
        learned_b = self._settings.value(key, None, type=float)
        if learned_b is not None:
            b = learned_b
        self._loading_est_ms = render_est.est(op, mp, a, b)
        self._loading_start = time.perf_counter()
        self.view.set_loading(True, 0.0)
        self._loading_timer.start()

    def _stop_loading(self, measured_ms: float | None = None) -> None:
        """Hide the loading bar; update the learned b if we have a measurement."""
        self._loading_timer.stop()
        self.view.set_loading(False)
        if measured_ms is not None and measured_ms > 0:
            from . import render_est
            op = self._loading_op
            mp = self._loading_mp
            key = f"render_est_{op}"
            seed = render_est.load_seed()
            b_old = self._settings.value(key, None, type=float)
            if b_old is None:
                b_old = seed.get(op, {"b": 50.0})["b"]
            b_new = render_est.update_b(b_old, measured_ms, mp)
            self._settings.setValue(key, b_new)

    def _loading_tick(self) -> None:
        """Update the loading bar progress every 100 ms."""
        elapsed = (time.perf_counter() - self._loading_start) * 1000.0
        from . import render_est
        prog = render_est.progress(elapsed, self._loading_est_ms)
        self.view.set_loading(True, prog)

    def _do_update(self) -> None:
        """UPDATE: check GitHub for a newer version and install if available."""
        from . import updater
        self._set_status("Prüfe GitHub…")

        def _progress(msg: str) -> None:
            self._set_status(msg)

        ok, msg = updater.check_and_update(progress_cb=_progress)
        self._set_status(msg)
        if ok and "Krita schließt" in msg:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(3000, self.close)

    # ── K27f: drag tracking ─────────────────────────────────────────────────

    def _on_drag_started(self) -> None:
        self._dragging = True

    def _on_drag_stopped(self) -> None:
        self._dragging = False
        # After release: trigger a fresh preview (lut + full if clarity active)
        if self._client is not None and self._session_id is not None:
            self._trigger_preview()

