"""LayoutMixin: geometry, layout mode, resize, scope buttons (K26)."""

from __future__ import annotations

from PyQt6.QtCore import QRegularExpression, QSize, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from . import scale, schemes, style, win_chrome
from .version import __version__
from .widgets import (Histogram, Parade, ScopeButton, Vectorscope, Waveform,
                      ColorWheel, SliderRow, small_button)
from .dlg_common import WHEELS
from .dlg_sections import _apply_label_width

class LayoutMixin:
    """Mixin for RefGradeDialog: geometry, layout mode, resize, scope buttons."""

    def _build_wheel_strip(self) -> QWidget:
        """K25b: wheel strip — only the 4 squares (VS + 3 wheels with masters).
        No layout; squares are positioned by setGeometry in _apply_geometry."""
        strip = QWidget(self)
        strip.setObjectName("wheel_strip")
        by_name = {s.name: s for s in self.sliders}
        self.wheels: dict[str, ColorWheel] = {}
        self.masters: dict[str, SliderRow] = {}
        self._wheel_reset_btns: dict[str, QPushButton] = {}
        self._wheel_names: dict[str, QLabel] = {}
        _rh = scale.row_h(style.current_s())
        for kind, title in WHEELS:
            wheel = ColorWheel(kind)
            wheel.setParent(strip)
            wheel.setObjectName(f"wheel_{kind}")
            wheel.changed.connect(self._wheel_changed)
            wheel.reset_requested.connect(self._reset_wheel)
            self.wheels[kind] = wheel
            # K28f: per-wheel reset button (↺) overlay, top-right of the square
            # same small button as the ↺ in the section headers: its own stylesheet drew
            # no glyph at all (2026-10-11, measured: lightness 0 over the whole button)
            rbtn = small_button("↺", tip=f"Reset {title} wheel")
            rbtn.setObjectName(f"wheel_reset_{kind}")
            rbtn.setParent(strip)
            rbtn.setFixedSize(_rh, _rh)
            rbtn.setCursor(Qt.CursorShape.PointingHandCursor)
            rbtn.clicked.connect(lambda _c=False, k=kind: self._reset_wheel(k))
            self._wheel_reset_btns[kind] = rbtn
            m = by_name[f"{kind}_m"]
            # user 2026-10-11: no "Master"/H-M-L label in the row; the wheel's name sits
            # directly above the slider line, flush with the slider's left edge
            self.masters[kind] = SliderRow(m.name, m.min, m.max, m.default, m.step, "", 0)
            self.masters[kind].setToolTip(f"{wheel._LABELS[kind]}: overall level of this wheel")
            self.masters[kind].setParent(strip)
            name = QLabel(wheel._LABELS[kind], strip)
            name.setObjectName(f"wheel_name_{kind}")
            name.setFont(style.font("small"))
            name.setProperty("font_role", "small")
            name.setProperty("muted", True)
            self._wheel_names[kind] = name
        # Vectorscope is one of the 4 squares (moved from scope block)
        self.vectorscope = Vectorscope()
        self.vectorscope.setParent(strip)
        self.vectorscope.setObjectName("vectorscope")
        return strip


    def _build_scope_block(self) -> QWidget:
        """K27a: bottom block — histogram/parade fill the block; buttons as direct children at the top."""
        w = QWidget(self)
        w.setObjectName("scope_block")

        # Histogram and Parade both fill the entire block (one visible at a time)
        self.histogram = Histogram(w)
        self.histogram.setObjectName("histogram")
        self._parade = Parade(w)
        self._parade.setObjectName("parade")
        self._wave = Waveform(w)
        self._wave.setObjectName("waveform")

        # Scope view: "hist" or "parade", default "parade"
        self._scope_view = self._settings.value("scope_view", "parade", type=str)

        _rh = scale.row_h(style.current_s())
        _btn_style = (f"background-color: rgba(0,0,0,160); color: {style.TEXT};"
                      f" border: none; padding: 0;")

        # View-toggle buttons
        self._btn_hist = QPushButton("HIST", w)
        self._btn_hist.setCheckable(True)
        self._btn_hist.setStyleSheet(_btn_style)
        self._btn_hist.setFixedHeight(_rh)
        self._btn_hist.setToolTip("Histogram view: channel distribution")
        self._btn_hist.clicked.connect(lambda: self._set_scope_view("hist"))

        self._btn_parade = QPushButton("PARADE", w)
        self._btn_parade.setCheckable(True)
        self._btn_parade.setStyleSheet(_btn_style)
        self._btn_parade.setFixedHeight(_rh)
        self._btn_parade.setToolTip("Parade: R, G and B waveforms side by side — brightness per channel across the image width.")
        self._btn_parade.clicked.connect(lambda: self._set_scope_view("parade"))

        self._btn_wave = QPushButton("WAVE", w)
        self._btn_wave.setCheckable(True)
        self._btn_wave.setStyleSheet(_btn_style)
        self._btn_wave.setFixedHeight(_rh)
        self._btn_wave.setToolTip("Luma waveform: brightness (up) across the image width")
        self._btn_wave.clicked.connect(lambda: self._set_scope_view("wave"))

        # Channel buttons (ScopeButtons)
        self._scope_btns: dict[str, ScopeButton] = {}
        for ch in ("rgb", "r", "g", "b", "l"):
            b = ScopeButton(ch, w)
            b.clicked.connect(lambda _c=False, n=ch: self._scope_btn_clicked(n))
            self._scope_btns[ch] = b

        # Layout-mode buttons (right-aligned)
        self._btn_h = QPushButton("H", w)
        self._btn_h.setToolTip("Layout H: colour wheels in a row — usually better for landscape images.")
        self._btn_v = QPushButton("V", w)
        self._btn_v.setToolTip("Layout V: colour wheels stacked beside the controls, picture uses the full height — usually better for portrait images.")
        self._btn_auto = QPushButton("AUTO", w)
        self._btn_auto.setToolTip("Layout AUTO: picks H or V, whichever shows the picture larger.")
        for b in (self._btn_h, self._btn_v, self._btn_auto):
            b.setCheckable(True)
            b.setStyleSheet(_btn_style)
            b.setFixedHeight(_rh)
        self._btn_h.clicked.connect(lambda: self._set_layout_mode("H"))
        self._btn_v.clicked.connect(lambda: self._set_layout_mode("V"))
        self._btn_auto.clicked.connect(lambda: self._set_layout_mode("AUTO"))

        # Apply initial state
        self.histogram.show_luma = self._settings.value("scope_luma", True, type=bool)
        self._scope_btns["l"].set_active(self.histogram.show_luma)
        self._scope_btns["rgb"].set_active(self.histogram.visible_channels == {"r", "g", "b"})
        self._apply_scope_view()
        self._update_layout_mode_btns()

        # Raise all buttons above histogram/parade
        for b in (self._btn_hist, self._btn_parade, self._btn_wave, *self._scope_btns.values(),
                  self._btn_h, self._btn_v, self._btn_auto):
            b.raise_()
        return w

    def _wheel_squares(self, mode: str, rects: dict, rh: int) -> list[tuple[int, int, int, int]]:
        """The 4 squares (VS, Lights, Mids, Lows) in strip coordinates, and the WHEELS chapter
        placed next to them (user 2026-10-11: its controls belong with the wheels, not in the
        column). V: chapter under the wheel column, squares shrink to make room. H: the band
        below the image is too low for that, so the chapter sits left of the vectorscope."""
        strip_x, strip_y, strip_w, strip_h = rects["strip"]
        sq = [(x - strip_x, y - strip_y, w, h) for x, y, w, h in rects["squares"]]
        sec = self._wheel_section
        sec_h = sec.sizeHint().height()
        gap = rh // 2
        if mode == "V":
            side = max(1, (strip_h - sec_h - gap) // 4)
            # half a row clear of the column: its values must not touch the wheel names
            sq = [(gap, i * side, strip_w - gap, side) for i in range(4)]
            # pin to strip width so the right edge stays flush with the scope block
            sec_w = strip_w - 2 * gap
            sec.setGeometry(strip_x + gap, strip_y + 4 * side + gap, sec_w, sec_h)
        else:
            # the strip is exactly four squares wide: shrink them so the chapter fits on the left
            sec_w = max(sec.sizeHint().width(), round(10 * rh))
            side = max(1, min(strip_h, (strip_w - sec_w - 2 * gap) // 4))
            x0 = strip_w - 4 * side
            y0 = strip_h - side
            sq = [(x0 + i * side, y0, side, side) for i in range(4)]
            sec.setGeometry(strip_x + gap, strip_y + strip_h - sec_h - gap, x0 - 2 * gap, sec_h)
        sec.show()
        sec.raise_()
        return sq


    def _apply_layout_mode(self) -> None:
        """K25b: resolve AUTO and apply geometry via layout_rects + setGeometry."""
        if self._layout_mode == "AUTO":
            iw = getattr(self, "_orig_w", 0)
            ih = getattr(self, "_orig_h", 0)
            # K28f: fall back to preview image size if source dimensions missing
            if iw <= 0 or ih <= 0:
                img = getattr(self, "_original_image", None)
                if img is not None and not img.isNull():
                    iw = max(iw, img.width())
                    ih = max(ih, img.height())
            result = scale.choose_mode(self.width(), self.height(),
                                       iw, ih,
                                       getattr(self, "_effective_mode", None))
            if result is not None:
                self._effective_mode = result
            else:
                self._effective_mode = getattr(self, "_effective_mode", None) or "V"
        else:
            self._effective_mode = self._layout_mode
        self._apply_geometry()


    def _apply_geometry(self) -> None:
        """Set geometry on all 4 blocks and the 4 squares from layout_rects."""
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0:
            return
        mode = getattr(self, "_effective_mode", "V")
        iw = getattr(self, "_orig_w", 0)
        ih = getattr(self, "_orig_h", 0)
        rects = scale.layout_rects(W, H, mode, iw, ih)

        self.panel.setGeometry(*rects["panel"])
        self._wheel_strip.setGeometry(*rects["strip"])
        self._scope_block.setGeometry(*rects["bottom"])
        self.view.setGeometry(*rects["view"])

        # Histogram and Parade fill the entire scope block
        bw, bh = rects["bottom"][2], rects["bottom"][3]
        self.histogram.setGeometry(0, 0, bw, bh)
        self._parade.setGeometry(0, 0, bw, bh)
        self._wave.setGeometry(0, 0, bw, bh)

        # Position the 4 squares within the strip
        strip_x, strip_y, strip_w, strip_h = rects["strip"]
        squares = rects["squares"]
        _rh = scale.row_h(style.current_s())
        _margin = 2
        squares = self._wheel_squares(mode, rects, _rh)
        name_h = QFontMetrics(style.font("small")).height()
        for i, kind in enumerate(("gain", "gamma", "lift")):
            sx, sy, sw, sh = squares[i + 1]
            # the disc's own box (square), then name + slider directly beneath it; spare height
            # goes below the slider, so a slider sits nearer its own wheel than the next one
            # (user 2026-10-11)
            # V: one row kept as air below the slider, before the next wheel; H: nothing below,
            # so the disc keeps that row and the group sits on the bottom edge
            air = 2 * _rh if mode == "V" else _rh
            wheel_h = min(sh - name_h - air - _margin + (0 if mode == "V" else 6), sw)
            top = sy if mode == "V" else sy + sh - (wheel_h - 6 + name_h + _rh)
            self.wheels[kind].setGeometry(sx, top, sw, wheel_h)
            name_y = top + wheel_h - 6      # into the disc box's empty 10 px rim margin
            master_y = name_y + name_h
            # gap of half a row on the right: the value must not touch the next square
            self.masters[kind].setGeometry(sx, master_y, sw - _rh // 2, _rh)
            self.masters[kind].layout().activate()
            # the wheel's name directly above the slider line, flush with the slider
            slider_x = self.masters[kind].slider.geometry().x()
            self._wheel_names[kind].setGeometry(sx + slider_x, name_y, sw - slider_x, name_h)
            # K28f: ↺ button at top-right of the wheel square
            self._wheel_reset_btns[kind].setGeometry(sx + sw - _rh, top, _rh, _rh)
        # vectorscope circle exactly as large as a wheel disc (user 2026-10-11)
        c, r = self.wheels["gain"]._geometry()
        vx, vy, vw, _vh = squares[0]
        d = round(2 * r)
        dy = self.wheels["gain"].y() - squares[1][1]     # same vertical place as the discs
        self.vectorscope.setGeometry(vx + round((vw - d) / 2), vy + dy + round(c.y() - r), d, d)

        # Position scope buttons at the top of the scope block
        self._position_scope_buttons()


    def _set_layout_mode(self, mode: str) -> None:
        """H / V / AUTO button: set the layout mode and persist it."""
        self._settings.setValue("layout_mode", mode)
        self._layout_mode = mode
        self._update_layout_mode_btns()
        self._apply_layout_mode()

    def _update_layout_mode_btns(self) -> None:
        """Sync the checked state of H/V/AUTO buttons with the current layout mode."""
        mode = getattr(self, "_layout_mode", "AUTO")
        self._btn_h.setChecked(mode == "H")
        self._btn_v.setChecked(mode == "V")
        self._btn_auto.setChecked(mode == "AUTO")

    def _set_scope_view(self, view: str) -> None:
        """Switch between histogram, parade and waveform; persist the choice."""
        self._scope_view = view
        self._settings.setValue("scope_view", view)
        self._apply_scope_view()
        if hasattr(self, "_trigger_preview"):    # parade/wave are only computed when visible
            self._trigger_preview()

    def _apply_scope_view(self) -> None:
        """Show the active scope widget, hide the other; sync button states."""
        self.histogram.setVisible(self._scope_view == "hist")
        self._parade.setVisible(self._scope_view == "parade")
        self._wave.setVisible(self._scope_view == "wave")
        self._btn_hist.setChecked(self._scope_view == "hist")
        self._btn_parade.setChecked(self._scope_view == "parade")
        self._btn_wave.setChecked(self._scope_view == "wave")

    def _position_scope_buttons(self) -> None:
        """Position all scope buttons in one row at the top of the scope block."""
        _rh = scale.row_h(style.current_s())
        margin = _rh // 4
        spacing = 4
        gap = _rh // 2
        y = margin
        block_w = self._scope_block.width()

        fm = self._btn_hist.fontMetrics()
        pad = margin

        def text_w(label: str) -> int:
            return max(_rh, fm.horizontalAdvance(label) + 2 * pad)

        # Left group: HIST, PARADE, (gap), RGB, R, G, B, L, ▭
        x = margin
        self._btn_hist.setGeometry(x, y, text_w("HIST"), _rh)
        x += text_w("HIST") + spacing
        self._btn_parade.setGeometry(x, y, text_w("PARADE"), _rh)
        x += text_w("PARADE") + spacing
        self._btn_wave.setGeometry(x, y, text_w("WAVE"), _rh)
        x += text_w("WAVE") + gap

        for ch in ("rgb", "r", "g", "b", "l"):
            self._scope_btns[ch].setGeometry(x, y, _rh, _rh)
            x += _rh + spacing

        # Right group: H, V, AUTO (right-aligned)
        auto_w = text_w("AUTO")
        v_w = text_w("V")
        h_w = text_w("H")
        x = block_w - margin - auto_w
        self._btn_auto.setGeometry(x, y, auto_w, _rh)
        x -= v_w + spacing
        self._btn_v.setGeometry(x, y, v_w, _rh)
        x -= h_w + spacing
        self._btn_h.setGeometry(x, y, h_w, _rh)


    def _restyle_chrome(self) -> None:
        """Re-apply the persistent inline chrome styles that captured scheme colours at build time."""
        _btn_style = (f"background-color: rgba(0,0,0,160); color: {style.TEXT};"
                      f" border: none; padding: 0;")
        for b in (getattr(self, "_btn_hist", None), getattr(self, "_btn_parade", None),
                  getattr(self, "_btn_wave", None), getattr(self, "_btn_h", None),
                  getattr(self, "_btn_v", None), getattr(self, "_btn_auto", None)):
            if b is not None:
                b.setStyleSheet(_btn_style)

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        if getattr(self, "_building", False):
            return
        new_s = scale.unit(self.width(), self.height())
        if scale.base_px(new_s) != scale.base_px(style.current_s()) and not getattr(self, "_updating_scale", False):
            self._updating_scale = True
            try:
                style.set_scale(new_s)
                style.apply_fonts(self)
                self._update_main_button_height()
                self._update_value_widths()
                self._cr_section._update_wheel_sizes()
                self.update()
            finally:
                self._updating_scale = False
        self._apply_layout_mode()
        if self.isVisible():
            self._fit_sections()


    def _build_wordmark(self) -> QWidget:
        """K41a: 'REF' (accent, bold) + 'grade' (text, light) + small dim 'v<version>'."""
        box = QWidget()
        box.setObjectName("wordmark")
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(0)
        self._wm_ref = QLabel("REF")
        self._wm_grade = QLabel("grade")
        self._wm_version = QLabel(f" v{__version__}")
        for lbl, role, weight in ((self._wm_ref, "title", QFont.Weight.Bold),
                                  (self._wm_grade, "title", QFont.Weight.Light),
                                  (self._wm_version, "small", QFont.Weight.Normal)):
            lbl.setFont(style.font(role, weight))
            lbl.setProperty("font_role", role)
            lbl.setProperty("font_weight", weight)
            lay.addWidget(lbl)
        self._wm_ref.setObjectName("wm_ref")
        self._wm_version.setObjectName("wm_version")
        self._wm_ref.setProperty("wordmark", "ref")
        self._wm_grade.setProperty("wordmark", "grade")
        self._wm_version.setProperty("wordmark", "ver")
        self._restyle_wordmark()
        return box

    def _restyle_wordmark(self) -> None:
        """Colours come from the stylesheet (QLabel[wordmark=...]); re-polish after a scheme change."""
        for lbl in (self._wm_ref, self._wm_grade, self._wm_version):
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        win_chrome.apply(self, schemes.current_scheme())
        self._apply_layout_mode()
        self._fit_sections()
        self._setup_lut_combo_view()   # K27e: idempotent; combo.view() blocks pre-show

    # ── reference slots ───────────────────────────────────────────────────────

    def _update_main_button_height(self) -> None:
        """K25c: set main button height to round(button_h * 1.5)."""
        h = round(scale.button_h(style.current_s()) * 1.5)
        for b in (self._btn_reset, self._btn_update, self._btn_cancel, self._btn_ok):
            b.setFixedHeight(h)   # UPDATE too: it was the odd one out at 28 px


    def _update_value_widths(self) -> None:
        """K25i: update value label widths and button sizes when the scale changes."""
        w = scale.value_w(style.current_s())
        _rh = scale.row_h(style.current_s())
        for _key, (_hdr, rows) in self.sections.items():
            for r in rows:
                r.set_value_width(w)
                r.setFixedHeight(_rh)
        for _band_rows in self._hsl_sliders.values():
            for r in _band_rows:
                r.set_value_width(w)
                r.setFixedHeight(_rh)
        # amount_hsl / amount_lut sit in no rows list and kept their build-time width
        for _r in self.findChildren(SliderRow):
            _r.set_value_width(w)
            _r.setFixedHeight(_rh)
        for _b in self.findChildren(QPushButton, QRegularExpression("^(hsl_mode_|wb_auto$|wb_pick$)")):
            _b.setFixedHeight(_rh)   # wb_*: kept the build-time 17 px at 2560x1392 (2026-10-11)
        # K25i: update the general help button size and master row height
        if hasattr(self, "master"):
            self.master.setFixedHeight(_rh)
        self._general_help_btn.setFixedSize(_rh, _rh)
        if hasattr(self, "_chain_order_btn"):      # ⇅ neben MASTER (K35a), gleiche Größe wie „?“
            self._chain_order_btn.setFixedSize(_rh, _rh)
        if hasattr(self, "_scheme_combo"):         # Farbschema-Dropdown
            _cf = style.font("small")
            _cfm = QFontMetrics(_cf)
            _cw = max(_cfm.horizontalAdvance(n) for n in schemes.SCHEMES) + 30
            self._scheme_combo.setFixedSize(_cw, _rh)
        self._master_row.setFixedHeight(_rh)
        # K25i: update section header button sizes
        for _key, (_hdr, _rows) in self.sections.items():
            _hdr.update_scale()
        # K25i: update method button heights
        if hasattr(self, "methods"):
            for _b in self.methods.buttons():
                _b.setFixedHeight(_rh)
        if hasattr(self, "fit_btn"):
            self.fit_btn.setFixedHeight(_rh)
        if hasattr(self, "_pop_btn"):
            self._pop_btn.setFixedHeight(_rh)
        # K25i: update wheel master slider heights
        if hasattr(self, "masters"):
            for _m in self.masters.values():
                _m.setFixedHeight(_rh)
        # K25i: update scope button sizes
        if hasattr(self, "_scope_btns"):
            for _b in self._scope_btns.values():
                _b.setFixedSize(_rh, _rh)
        # K27a: update scope block button heights and reposition
        if hasattr(self, "_btn_hist"):
            for _b in (self._btn_hist, self._btn_parade, self._btn_wave,
                       self._btn_h, self._btn_v, self._btn_auto):
                _b.setFixedHeight(_rh)
        # K27d: update curves channel button heights
        for _b in self.findChildren(QPushButton, QRegularExpression("^curves_ch_")):
            _b.setFixedHeight(_rh)
        # K28f: per-wheel ↺ buttons keep build-time size otherwise
        for _b in self._wheel_reset_btns.values():
            _b.setFixedSize(_rh, _rh)
        # LUT row buttons/combo use button_h and were built at the stale default scale
        _bh = scale.button_h(style.current_s())
        for _w in (self._lut_folder_btn, self._lut_save_btn, self._lut_combo):
            _w.setFixedHeight(_bh)
        self._position_scope_buttons()
        # K25i: defer label width update to next event-loop tick to avoid layout loop
        QTimer.singleShot(0, self._deferred_label_update)


    def _deferred_label_update(self) -> None:
        """K25i: recompute label widths after scale change (deferred to avoid layout loop)."""
        for _key, (_hdr, rows) in self.sections.items():
            _apply_label_width(rows)
        for _band_rows in self._hsl_sliders.values():
            _apply_label_width(_band_rows)
        self._layout_not_in_lut()   # after the pass above, which squeezed '(no LUT!)'
        # master row
        if hasattr(self, "master"):
            fm = QFontMetrics(style.font("base"))
            pad = scale.row_h(style.current_s()) // 2
            self.master.set_label_width(fm.horizontalAdvance("MASTER") + pad)


    def show_screen(self) -> None:
        """Open frameless over the screen's free area (everything but the taskbar), so nothing
        hides under it (user 2026-10-07, 2560×1440 with a 46 px taskbar). F11 = real fullscreen."""
        screen = (self.parentWidget().screen() if self.parentWidget() else None) or self.screen()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setGeometry(screen.availableGeometry())
        self.show()
        self.raise_()
        self.activateWindow()


    def _toggle_split(self) -> None:
        """Toggle split view (S key or split button in view); affects the image only."""
        self.view.split_mode = not self.view.split_mode
        self._settings.setValue("split_mode", int(self.view.split_mode))
        if self._original_image is not None and not self._original_image.isNull():
            self.view.set_original(self._original_image)
        self.view.update()

    def _toggle_clip_view(self) -> None:
        """Toggle clipping view (C key or C button in view); re-requests the preview."""
        self._clip_view = not self._clip_view
        self.view._clip_view = self._clip_view
        self._settings.setValue("clip_view", int(self._clip_view))
        self.view.update()
        self._schedule_preview()

    # ── Scope buttons ─────────────────────────────────────────────────────────

    def _scope_btn_clicked(self, name: str) -> None:
        """RGB/R/G/B select the colour channels (exclusive); L is an independent switch: the
        luminance lies as a white ghost glow behind whatever channels are shown (user
        2026-10-11: "Luminance kann auch zeitgleich hinter Histos gelegt werden")."""
        if name == "l":
            on = not getattr(self.histogram, "show_luma", True)
            self.histogram.show_luma = on
            self._settings.setValue("scope_luma", on)
            self._scope_btns["l"].set_active(on)
            self.histogram.update()
            return
        channels = {"r", "g", "b"} if name == "rgb" else {name}
        self.histogram.visible_channels = channels
        self._parade.visible_channels = set(channels)
        self._scope_btns["rgb"].set_active(name == "rgb")
        for ch in ("r", "g", "b"):
            self._scope_btns[ch].set_active(ch == name)
        self.histogram.update()
        self._parade.update()

