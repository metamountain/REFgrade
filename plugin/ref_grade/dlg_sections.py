"""SectionsMixin: section building, state, reset, wheel, slider helpers (K26)."""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor, QFontMetrics
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from . import scale, style
from .curves_ui import CurvesEditor, CHANNELS, HUE_CHANNELS, ALL_CHANNELS
from .widgets import ColorWheel, SectionHeader, SliderRow
from .dlg_common import LABELS, DEFAULT_ON, WHEELS, POP_RECIPE, AUTO_SECTIONS

# Mouse-over help for the curve channel buttons (user 2026-10-12: the titles are cryptic).
# Hue curves: the flat middle line means "no change"; drag a point up or down from it.
CURVE_TIPS = {
    "rgb": "RGB — brightness curve for all three channels together.\n"
           "Left = shadows, right = highlights. Pull a point up to brighten that range,\n"
           "down to darken it. An S-shape adds contrast.",
    "r": "Red channel curve. Up adds red, down adds cyan in that brightness range.",
    "g": "Green channel curve. Up adds green, down adds magenta in that brightness range.",
    "b": "Blue channel curve. Up adds blue, down adds yellow in that brightness range.",
    "hue_hue": "HvH — Hue vs Hue.\nAcross: the colour of a pixel. Up/down: shift that colour\n"
               "to a neighbouring hue (e.g. turn orange skin more red, green grass more yellow).",
    "hue_sat": "HvS — Hue vs Saturation.\nAcross: the colour. Up: that colour gets more intense,\n"
               "down: it fades toward grey (e.g. calm a loud blue sky).",
    "hue_lum": "HvL — Hue vs Lightness.\nAcross: the colour. Up: that colour gets brighter,\n"
               "down: darker (e.g. darken a blue sky, brighten foliage).",
    "lum_sat": "LvS — Lightness vs Saturation.\nAcross: dark to bright pixels. Up/down: more or\n"
               "less colour there (e.g. take the colour out of the deep shadows).",
    "sat_sat": "SvS — Saturation vs Saturation.\nAcross: grey to fully coloured pixels. Up/down:\n"
               "boost or reduce them (e.g. lift muted colours without pushing loud ones further).",
}


class _DimGuard(QObject):
    """Event filter: first mouse press in a dimmed (OFF) section turns it ON."""

    def __init__(self, content: QWidget, header: SectionHeader):
        super().__init__()
        self._content = content
        self._header = header

    def eventFilter(self, obj, event) -> bool:
        if (event.type() == QEvent.Type.MouseButtonPress
                and self._content.property("dim") is True
                and not self._header.is_on()):
            self._header.on.setChecked(True)
        return False


def _separator() -> QWidget:
    """Thin horizontal line (style.LINE) between the Amount row and the section sliders."""
    from PyQt6.QtWidgets import QFrame
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {style.LINE}; border: none;")
    return sep



def _group(title: str, names: tuple, sliders: list, key: str | None = None) -> tuple[QWidget, SectionHeader, list[SliderRow], QWidget]:
    """Section header (ON/OFF, ↺) + Amount row + separator + one row per slider.
    Returns (widget, header, rows, content) where content is the collapsible area."""
    by_name = {s.name: s for s in sliders}
    key = key or title.lower()
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    box = QVBoxLayout(w)
    box.setContentsMargins(0, 0, 0, 2)
    box.setSpacing(1)
    header = SectionHeader(key, title, on=key in DEFAULT_ON, auto=(key in AUTO_SECTIONS))
    box.addWidget(header)
    content = QWidget()
    cbox = QVBoxLayout(content)
    cbox.setContentsMargins(0, 0, 0, 0)
    cbox.setSpacing(1)
    rows = []
    # Amount slider as the first content row
    amount = SliderRow(f"amount_{key}", 0.0, 1.0, 1.0, 0.01, "Amount")
    amount.setObjectName(f"amount_{key}")
    rows.append(amount)
    cbox.addWidget(amount)
    header.set_amount_slider(amount)
    # Separator line
    sep = _separator()
    cbox.addWidget(sep)
    for n in names:
        s = by_name[n]
        rows.append(SliderRow(s.name, s.min, s.max, s.default, s.step, LABELS[n],
                              track=style.TRACKS.get(n)))
        cbox.addWidget(rows[-1])
    box.addWidget(content)
    return w, header, rows, content



def _apply_label_width(rows: list[SliderRow]) -> None:
    """K25i: set label width per section = max horizontalAdvance + row_h/2."""
    if not rows:
        return
    fm = QFontMetrics(style.font("base"))
    max_w = 0
    for r in rows:
        if r.label_width > 0:
            w = fm.horizontalAdvance(r.label_text())
            if w > max_w:
                max_w = w
    if max_w > 0:
        pad = scale.row_h(style.current_s()) // 2
        target = max_w + pad
        for r in rows:
            r.set_label_width(target)



def _curves_section(sliders: list) -> tuple[QWidget, SectionHeader, CurvesEditor, QWidget]:
    """Curves section: header + Amount + separator + channel buttons (RGB + Hue) + CurvesEditor."""
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    box = QVBoxLayout(w)
    box.setContentsMargins(0, 0, 0, 2)
    box.setSpacing(1)
    header = SectionHeader("curves", "Curves", on="curves" in DEFAULT_ON, auto=("curves" in AUTO_SECTIONS))
    box.addWidget(header)
    content = QWidget()
    cbox = QVBoxLayout(content)
    cbox.setContentsMargins(0, 0, 0, 0)
    cbox.setSpacing(2)
    # Amount slider as the first content row
    amount = SliderRow("amount_curves", 0.0, 1.0, 1.0, 0.01, "Amount")
    amount.setObjectName("amount_curves")
    cbox.addWidget(amount)
    header.set_amount_slider(amount)
    cbox.addWidget(_separator())
    # RGB channel buttons
    btn_row = QHBoxLayout()
    btn_row.setSpacing(2)
    for ch in CHANNELS:
        b = QPushButton(ch.upper())
        b.setObjectName(f"curves_ch_{ch}")
        b.setCheckable(True)
        b.setChecked(ch == "rgb")
        b.setToolTip(CURVE_TIPS[ch])
        b.setFont(style.font("small"))
        b.setProperty("font_role", "small")
        b.setFixedHeight(scale.row_h(style.current_s()))
        b.setProperty("small", True)
        btn_row.addWidget(b)
    btn_row.addSpacing(6)  # Abstand zwischen RGB und Hue
    # Hue channel buttons
    hue_labels = {"hue_hue": "HvH", "hue_sat": "HvS", "hue_lum": "HvL",
                  "lum_sat": "LvS", "sat_sat": "SvS"}
    for ch in HUE_CHANNELS:
        b = QPushButton(hue_labels[ch])
        b.setObjectName(f"curves_ch_{ch}")
        b.setCheckable(True)
        b.setChecked(False)
        b.setToolTip(CURVE_TIPS[ch])
        b.setFont(style.font("small"))
        b.setProperty("font_role", "small")
        b.setFixedHeight(scale.row_h(style.current_s()))
        b.setProperty("small", True)
        btn_row.addWidget(b)
    btn_row.addSpacing(4)
    reset_btn = QPushButton("↺")
    reset_btn.setObjectName("curves_ch_reset")
    reset_btn.setFont(style.font("small"))
    reset_btn.setProperty("font_role", "small")
    reset_btn.setFixedHeight(scale.row_h(style.current_s()))
    reset_btn.setProperty("small", True)
    reset_btn.setToolTip("Reset active channel")
    btn_row.addWidget(reset_btn)
    cbox.addLayout(btn_row)
    # Editor
    editor = CurvesEditor()
    editor.setObjectName("curves_editor")
    cbox.addWidget(editor, 1)
    box.addWidget(content)
    return w, header, editor, content


HSL_BANDS = ("red", "yellow", "green", "cyan", "blue", "magenta")
HSL_HUE = {"red": 0, "yellow": 60, "green": 120, "cyan": 180, "blue": 240, "magenta": 300}
HSL_MODES = (("hue", "HUE"), ("sat", "SAT"), ("light", "LUM"))



def _hsl_track(band: str, mode: str) -> tuple:
    """Superslim colour track of one HSL slider (user: like Lightroom, everything visible)."""
    c = HSL_HUE[band]
    hx = lambda h, s=255, v=230: QColor.fromHsv(int(h) % 360, s, v).name()  # noqa: E731
    if mode == "hue":
        return ((0.0, hx(c - 30)), (0.5, hx(c)), (1.0, hx(c + 30)))
    if mode == "sat":
        return ((0.0, "#5A5A5A"), (0.5, hx(c, 120, 180)), (1.0, hx(c)))
    return ((0.0, hx(c, 200, 50)), (0.5, hx(c, 230, 160)), (1.0, hx(c, 90, 255)))



def _hsl_section(sliders: list) -> tuple[QWidget, SectionHeader, dict, QWidget]:
    """HSL like Lightroom: Amount + separator + tabs HUE · SAT · LUM, below one slider per
    colour band, all inline (user 2026-10-08: "integrieren, Pop-ups nicht gut").

    Returns (widget, header, {band: (hue_slider, sat_slider, light_slider)}, content). All 18
    rows live in the layout; the active tab decides which 6 are visible.
    """
    by_name = {s.name: s for s in sliders}
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    box = QVBoxLayout(w)
    box.setContentsMargins(0, 0, 0, 2)
    box.setSpacing(1)
    header = SectionHeader("hsl", "HSL", on="hsl" in DEFAULT_ON)
    box.addWidget(header)
    content = QWidget()
    cbox = QVBoxLayout(content)
    cbox.setContentsMargins(0, 0, 0, 0)
    cbox.setSpacing(1)
    # Amount slider as the first content row
    amount = SliderRow("amount_hsl", 0.0, 1.0, 1.0, 0.01, "Amount")
    amount.setObjectName("amount_hsl")
    cbox.addWidget(amount)
    header.set_amount_slider(amount)
    cbox.addWidget(_separator())
    tabs = QHBoxLayout()
    tabs.setSpacing(2)
    _hsl_tips = {"hue": "Show Hue sliders for each colour band",
                 "sat": "Show Saturation sliders for each colour band",
                 "light": "Show Lightness sliders for each colour band"}
    for mode, text in HSL_MODES:
        b = QPushButton(text)
        b.setObjectName(f"hsl_mode_{mode}")
        b.setCheckable(True)
        b.setChecked(mode == "hue")
        b.setFont(style.font("small"))
        b.setProperty("font_role", "small")
        b.setFixedHeight(scale.row_h(style.current_s()))
        b.setProperty("small", True)
        b.setToolTip(_hsl_tips[mode])
        tabs.addWidget(b)
    cbox.addLayout(tabs)
    rows: dict[str, list[SliderRow]] = {}
    for band in HSL_BANDS:
        trio = []
        for mode, _text in HSL_MODES:
            sd = by_name[f"sel_{band}_{mode}"]
            row = SliderRow(sd.name, sd.min, sd.max, sd.default, sd.step,
                            band.capitalize(), track=_hsl_track(band, mode))
            row.setVisible(mode == "hue")
            cbox.addWidget(row)                    # always parented: no stray windows
            trio.append(row)
        rows[band] = tuple(trio)
    box.addWidget(content)
    return w, header, rows, content




class SectionsMixin:
    """Mixin for RefGradeDialog: section building, state, reset, wheel, slider helpers."""

    def _build_wheels_section(self) -> QWidget:
        """K25b: WHEELS section in the panel (after CURVES, before LUT).
        Contains: header + Amount + separator + Blending + Balance."""
        by_name = {s.name: s for s in self.sliders}
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 2)
        box.setSpacing(1)
        self._wheel_header = SectionHeader("wheels", "Wheels", on="wheels" in DEFAULT_ON,
                                           auto=("wheels" in AUTO_SECTIONS))
        self._wheel_header.triangle.hide()
        box.addWidget(self._wheel_header)
        content = QWidget()
        cbox = QVBoxLayout(content)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(1)
        self._wheel_amount = SliderRow("amount_wheels", 0.0, 1.0, 1.0, 0.01, "Amount")
        self._wheel_amount.setObjectName("amount_wheels")
        cbox.addWidget(self._wheel_amount)
        self._wheel_header.set_amount_slider(self._wheel_amount)
        cbox.addWidget(_separator())
        # Blending default 0.5 like Lightroom; Intensity scales the wheels' reach (2026-10-11)
        self._wheel_blending_slider = SliderRow("wheel_blending", 0.0, 1.0, 0.5, 0.01, "Blending")
        self._wheel_balance_slider = SliderRow("wheel_balance", -1.0, 1.0, -0.3, 0.01, "Balance")   # = core sliders.py default
        self._wheel_intensity_slider = SliderRow("wheel_intensity", 0.0, 2.0, 1.0, 0.01, "Intensity")
        self._wheel_intensity_slider.setObjectName("wheel_intensity")
        cbox.addWidget(self._wheel_intensity_slider)
        cbox.addWidget(self._wheel_blending_slider)
        cbox.addWidget(self._wheel_balance_slider)
        box.addWidget(content)
        self._wheel_content = content
        return w


    def _build_rgb_horizontal(self) -> QWidget:
        """K25d: QGridLayout — row names (Gamma/Offset/Gain) × column headers
        (Master/Red/Green/Blue); each cell has slider+value only. Zones row below."""
        by_name = {s.name: s for s in self.sliders}
        w = QWidget()
        w.setObjectName("rgb_h")
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        self._rgb_header = SectionHeader("rgb", "RGB", on="rgb" in DEFAULT_ON, auto=("rgb" in AUTO_SECTIONS))
        box.addWidget(self._rgb_header)
        content = QWidget()
        cbox = QVBoxLayout(content)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(2)
        rows: list[SliderRow] = []
        amount = SliderRow("amount_rgb", 0.0, 1.0, 1.0, 0.01, "Amount")
        amount.setObjectName("amount_rgb")
        rows.append(amount)
        cbox.addWidget(amount)
        self._rgb_header.set_amount_slider(amount)
        cbox.addWidget(_separator())
        # Grid: row 0 = headers, rows 1-3 = Gamma/Offset/Gain × Master/Red/Green/Blue
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        headers = ("", "Master", "Red", "Green", "Blue")
        for col, h in enumerate(headers):
            if h:
                lbl = QLabel(h)
                lbl.setFont(style.font("base"))
                lbl.setProperty("font_role", "base")
                grid.addWidget(lbl, 0, col, Qt.AlignmentFlag.AlignCenter)
        # (shown name, slider key part); "Offset" is the DaVinci name of the pedestal (a shift of
        # every tone) — user 2026-10-11; the key stays "pedestal" so saved values keep working
        row_defs = (("Gamma", "gamma"), ("Offset", "pedestal"), ("Gain", "gain"))
        for row_i, (rn, comp) in enumerate(row_defs):
            lbl = QLabel(rn)
            lbl.setFont(style.font("base"))
            lbl.setProperty("font_role", "base")
            lbl.setProperty("muted", True)
            grid.addWidget(lbl, row_i + 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for col_i, ch in enumerate(("m", "r", "g", "b")):
                n = f"rgb_{ch}_{comp}"
                s = by_name[n]
                r = SliderRow(s.name, s.min, s.max, s.default, s.step, "",
                              label_width=0, track=style.TRACKS.get(n))
                rows.append(r)
                grid.addWidget(r, row_i + 1, col_i + 1)
        for col in range(1, 5):
            grid.setColumnStretch(col, 1)
        cbox.addLayout(grid)
        # Zones row: "Channels act on" + Shadows / Mids / Highs as percent
        cbox.addWidget(_separator())
        zones_row = QHBoxLayout()
        zones_row.setSpacing(style.GAP)
        zones_lbl = QLabel("Channels act on")
        zones_lbl.setFont(style.font("base"))
        zones_lbl.setProperty("font_role", "base")
        zones_lbl.setProperty("muted", True)
        zones_lbl.setToolTip("Where the Red/Green/Blue controls act. Master always acts everywhere.")
        zones_row.addWidget(zones_lbl)
        for n in ("rgb_use_shadows", "rgb_use_midtones", "rgb_use_highlights"):
            s = by_name[n]
            r = SliderRow(s.name, s.min, s.max, s.default, s.step, LABELS[n],
                          label_width=56, track=style.TRACKS.get(n))
            r.percent = True
            r._on_change()
            rows.append(r)
            zones_row.addWidget(r)
        cbox.addLayout(zones_row)
        box.addWidget(content)
        self._rgb_rows = rows
        self._rgb_content = content
        return w


    def _show_section_state(self, key: str) -> None:
        """OFF → content dimmed via property (stays enabled); ON → normal."""
        on = self.sections[key][0].is_on()
        self._dim_content(key, not on)
        if key == "reference":
            for t in self.slots.tiles:
                t.dimmed = not on
                t.update()
        if key == "wheels":
            for wh in self.wheels.values():
                wh.setProperty("dim", not on)
                wh.style().unpolish(wh)
                wh.style().polish(wh)
                wh.update()
        if key == "lut":
            self._lut_dim_state()


    def _mark_changed(self, key: str) -> None:
        """A value in section `key` changed → switch the section ON (user 2026-10-11). Silent
        while values are set programmatically for a reset or while the dialog is built."""
        if getattr(self, "_quiet_changes", 0) or not getattr(self, "_built", False):
            return
        header = self.sections.get(key, (None, None))[0]
        if header is not None and not header.is_on():
            header.on.setChecked(True)


    def _section_switched(self, key: str, on: bool) -> None:
        """ON/OFF: only dim/enable the content; no collapse (K25c)."""
        self._touch_section(key)
        self._show_section_state(key)
        self._mode_changed()


    def _dim_content(self, key: str, dim: bool) -> None:
        """Dim a section's content via property (stays enabled, interactive)."""
        content = self._contents.get(key)
        if content is None:
            return
        content.setProperty("dim", dim)
        content.style().unpolish(content)
        content.style().polish(content)
        content.update()
        widgets = [content] + content.findChildren(QWidget)
        guard = getattr(self, "_dim_guards", None)
        if guard is None:
            guard = self._dim_guards = {}
        if dim:
            g = guard.get(key)
            if g is None:
                g = _DimGuard(content, self.sections[key][0])
                guard[key] = g
            for w in widgets:
                w.installEventFilter(g)
        else:
            g = guard.pop(key, None)
            if g is not None:
                for w in widgets:
                    w.removeEventFilter(g)


    def _set_section_collapsed(self, key: str, collapsed: bool) -> None:
        """Collapse or expand a section's content area."""
        content = self._contents.get(key)
        if content is not None:
            content.setVisible(not collapsed)
            content.parent().updateGeometry()
        header = self.sections[key][0]
        header.set_collapsed(collapsed, emit=False)
        self._settings.setValue(f"collapse_{key}", int(collapsed))


    def _section_collapsed_changed(self, key: str, collapsed: bool) -> None:
        """K25c: Dreieck-Klick → Inhalt ein-/ausblenden; bei Öffnen Platz prüfen."""
        self._set_section_collapsed(key, collapsed)
        auto = getattr(self, "_auto_collapsed", None)
        if auto is not None and key in auto:
            auto.remove(key)            # the user decided: no longer ours to reopen
        if not collapsed:
            self._touch_section(key)
            self._enforce_no_scroll()


    def _hsl_mode_selected(self, mode: str) -> None:
        """HUE / SAT / LUM tab: show that slider of every colour band (values all kept)."""
        for m, btn in self._hsl_mode_btns.items():
            btn.setChecked(m == mode)
        k = [m for m, _t in HSL_MODES].index(mode)
        for trio in self._hsl_sliders.values():
            for i, row in enumerate(trio):
                row.setVisible(i == k)


    def _curves_channel_selected(self, ch: str) -> None:
        """Switch the active Curves channel (RGB or Hue)."""
        for c, btn in self._curves_ch_btns.items():
            btn.setChecked(c == ch)
        self.curves_editor.set_active(ch)
        self._mode_changed()


    def _curves_channel_reset(self) -> None:
        """Reset only the active curves channel to identity."""
        self.curves_editor.reset_channel()
        if self._client is not None:
            self._schedule_preview()


    def _curves_changed(self, ch: str, pts: list) -> None:
        """Curves editor emitted a change."""
        self._mark_changed("curves")
        self._mode_changed()


    def _reset_section(self, key: str) -> None:
        """↺: values back to default and the section OFF — nothing in it is changed any more."""
        self._quiet_changes = getattr(self, "_quiet_changes", 0) + 1
        try:
            self._reset_section_values(key)
        finally:
            self._quiet_changes -= 1
        self.sections[key][0].on.setChecked(False)
        if self._client is not None:
            self._schedule_preview()


    def _reset_section_values(self, key: str) -> None:
        header, rows = self.sections[key]
        if header.amount_slider is not None:
            header.amount_slider.set_value(1.0)
        for r in rows:
            r.set_value(r.default)
        if key == "wheels":
            for k in self.wheels:
                self._reset_wheel(k)
        if key == "lut":
            self._lut_current_path = ""
            self._lut_combo.setText("— none —")
            header.on.setChecked(False)
            self._lut_dim_state()
        if key == "hsl":
            for band, (hue, sat, light) in self._hsl_sliders.items():
                hue.set_value(hue.default)
                sat.set_value(sat.default)
                light.set_value(light.default)
        if key == "curves":
            for ch in CHANNELS:
                self.curves_editor.set_curve(ch, [[0.0, 0.0], [1.0, 1.0]])
            for ch in HUE_CHANNELS:
                self.curves_editor.set_curve(ch, [[0.0, 0.5], [1.0, 0.5]])


    def _reset_wheel(self, kind: str) -> None:
        self.wheels[kind].set_point(0.0, 0.0)
        self.masters[kind].set_value(self.masters[kind].default)
        self._wheel_rest[kind] = (0.0, 0.0, 0.0)


    def _reset_all(self) -> None:
        for key, (h, _r) in self.sections.items():
            self._reset_section(key)
            h.on.setChecked(key in DEFAULT_ON)
        self._lut_dim_state()
        self.master.set_value(1.0)
        if self._client is None:
            self._show_mode()
        else:
            self._schedule_preview()


    def _wheel_changed(self, kind: str, u: float, v: float) -> None:
        self._mark_changed("wheels")
        if self._client is None:
            self._set_status(f"LAYOUT DUMMY · {kind}: " + ", ".join(
                f"{k} {v:.2f}" for k, v in self.wheel_values(kind).items()))
        else:
            self._schedule_preview()


    def _get_slider_value(self, name: str) -> float:
        """Get a slider value by name from any section."""
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                if r.name == name:
                    return r.value()
        return 0.0


    def _set_slider_value(self, name: str, value: float) -> None:
        """Set a slider value by name in any section."""
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                if r.name == name:
                    r.set_value(value)
                    return


    def _add_pop_button(self, sections: dict) -> None:
        """Add a POP button to the Tone section header."""
        if "tone" not in sections:
            return
        header, rows = sections["tone"]
        self._pop_btn = QPushButton("POP")
        self._pop_btn.setObjectName("tone_pop")
        self._pop_btn.setFont(style.font("small"))
        self._pop_btn.setProperty("font_role", "small")
        self._pop_btn.setFixedHeight(scale.row_h(style.current_s()))
        self._pop_btn.setProperty("small", True)
        self._pop_btn.setToolTip("Apply POP recipe: preset contrast, vibrance, highlights, shadows and clarity")
        self._pop_btn.clicked.connect(self._on_pop)
        # Insert into the header's layout (after the title, before stretch)
        layout = header.layout()
        layout.insertWidget(2, self._pop_btn)


    _NO_LUT_SLIDERS = ("clarity", "texture", "dehaze")

    def _add_not_in_lut_label(self, sections: dict) -> None:
        """'(no LUT!)' in the clarity/texture/dehaze rows, right after the name."""
        if "tone" not in sections:
            return
        from .ui_help import UI_HELP
        self._not_in_lut_labels: list[QLabel] = []
        self._not_in_lut_rows = list(sections["tone"][1])
        for row in sections["tone"][1]:
            if row.name not in self._NO_LUT_SLIDERS:
                continue
            lbl = QLabel("(no LUT!)")
            lbl.setObjectName("not_in_lut")
            lbl.setToolTip(UI_HELP.get("not_in_lut", ""))
            lbl.setFont(style.font("small"))
            lbl.setProperty("font_role", "small")
            lbl.setStyleSheet(f"color: {style.MUTED};")
            row.layout().insertWidget(1, lbl)
            row.slider.valueChanged.connect(lambda _v: self._update_not_in_lut())
            self._not_in_lut_labels.append(lbl)
        self._layout_not_in_lut()
        self._update_not_in_lut()

    def _layout_not_in_lut(self) -> None:
        """Name + '(no LUT!)' both whole: widen the TONE label column to the longest pair.
        Re-run after every label-width pass, which used to give the name the whole column and
        squeeze the hint to '(no L' (2026-10-11)."""
        labels = getattr(self, "_not_in_lut_labels", [])
        if not labels:
            return
        rows = self._not_in_lut_rows
        pairs = []
        for lbl in labels:
            row = lbl.parentWidget()
            lay = row.layout()
            name = lay.itemAt(0).widget()
            name_w = name.fontMetrics().horizontalAdvance(name.text()) + 2
            lbl_w = lbl.fontMetrics().horizontalAdvance(lbl.text()) + 2
            pairs.append((name, name_w, lbl, lbl_w, lay.spacing()))
        column = max(name_w + sp + lbl_w for _n, name_w, _l, lbl_w, sp in pairs)
        fm = QFontMetrics(style.font("base"))
        pad = scale.row_h(style.current_s()) // 2
        column = max(column, max((fm.horizontalAdvance(r.label_text()) + pad for r in rows
                                  if r.label_width > 0), default=0))
        for r in rows:
            if r.label_width > 0:
                r.set_label_width(column)
        for name, name_w, lbl, _lbl_w, sp in pairs:
            name.setFixedWidth(name_w)
            lbl.setFixedWidth(column - name_w - sp)

    def _update_not_in_lut(self) -> None:
        """Muted while all spatial sliders are 0, red when any would be left out of a LUT."""
        if not hasattr(self, "_not_in_lut_labels") or not hasattr(self, "sections"):
            return
        any_active = any(self._get_slider_value(n) != 0.0 for n in self._NO_LUT_SLIDERS)
        colour = style.ACCENT_HI if any_active else style.MUTED
        for lbl in self._not_in_lut_labels:
            lbl.setStyleSheet(f"color: {colour};")


    def _on_pop(self) -> None:
        """Apply the POP recipe: set sliders, turn Tone and Color ON."""
        for name, val in POP_RECIPE.items():
            self._set_slider_value(name, val)
        # Turn Tone and Color ON
        for key in ("tone", "color"):
            if key in self.sections:
                header, _ = self.sections[key]
                if not header.is_on():
                    header.on.click()
        self._schedule_preview()


    def _touch_section(self, key: str) -> None:
        """Mark a section as most recently touched."""
        if hasattr(self, '_touch_order'):
            if key in self._touch_order:
                self._touch_order.remove(key)
            self._touch_order.append(key)


    def _enforce_no_scroll(self) -> None:
        """Collapse the least recently touched scrolling-list section until the panel fits.
        WHEELS is never auto-collapsed and the last-touched one is kept. CURVES is collapsed
        LAST (K29e): the editor keeps its 2:1 minimum, so collapsing another section is
        preferred over squashing the curve; only if the panel still overflows with every
        other section closed does CURVES itself collapse."""
        if getattr(self, "_enforcing_no_scroll", False):
            return
        self._enforcing_no_scroll = True
        try:
            if not hasattr(self, "_touch_order"):
                self._touch_order: list[str] = list(getattr(self, "_scroll_keys", ()))
            lay = self.panel.layout()
            limit = self.panel.height()
            order = [k for k in self._touch_order[:-1]
                     if k in self._scroll_keys and k != "wheels" and k != "curves"]
            for key in order:
                if self._panel_min_height(lay) <= limit:
                    return
                content = self._contents.get(key)
                if content is not None and content.isVisible():
                    self._set_section_collapsed(key, True)
                    self.__dict__.setdefault("_auto_collapsed", []).append(key)
            if self._panel_min_height(lay) > limit:
                content = self._contents.get("curves")
                if content is not None and content.isVisible():
                    self._set_section_collapsed("curves", True)
                    self.__dict__.setdefault("_auto_collapsed", []).append("curves")
        finally:
            self._enforcing_no_scroll = False


    def _fit_sections(self) -> None:
        """After a size change: collapse what no longer fits, then reopen sections the
        no-scroll rule closed earlier, latest first, as long as they fit (2026-10-11: shown
        small first and then enlarged, TONE stayed closed above ~240 px of free column)."""
        if not hasattr(self, "panel") or getattr(self, "_enforcing_no_scroll", False):
            return
        self._enforce_no_scroll()
        auto = getattr(self, "_auto_collapsed", [])
        lay = self.panel.layout()
        limit = self.panel.height()
        while auto:
            key = auto[-1]
            self._set_section_collapsed(key, False)
            if self._panel_min_height(lay) > limit:
                self._set_section_collapsed(key, True)
                break
            auto.pop()


    def _panel_min_height(self, lay) -> int:
        """Minimum panel height measured now: a hidden content only reaches the cached
        layouts through a posted LayoutRequest, so deliver those before measuring."""
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QWidget
        # only this column's widgets: delivering to every object (receiver None) also hit
        # other dialogs alive at the time and crashed the suite (2026-10-11)
        for w in (self.panel, *self.panel.findChildren(QWidget)):
            if w.layout() is not None:
                QCoreApplication.sendPostedEvents(w, QEvent.Type.LayoutRequest.value)
        lay.activate()
        # sizeHint, not minimumSize: "fits" means every row at its natural height; the
        # minimum let buttons squash below row_h (2026-10-11: wb_auto 17 px at 2560x1392)
        need = max(lay.minimumSize().height(), lay.sizeHint().height())
        # K29e: the Curves editor carries its 2:1 minimum itself (CurvesEditor.resizeEvent)
        return need

