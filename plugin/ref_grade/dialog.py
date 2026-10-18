"""The REF Grade window: control column left (~1/3), image right (~2/3).

Without `client` the window is a layout dummy (no processing). With `client` and `source`
it is wired: every change triggers a debounced /preview, zoom triggers /crop, FIT / LUT /
OK call the worker, and tile hover calls /prerender. No krita import, no krita_bridge.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import (QEvent, QRectF, QSizeF, QObject, QRunnable, QSettings, QThreadPool, QTimer,
                          Qt, pyqtSignal)
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog,
                             QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget, QLayout,
                             QMenu, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from . import config, cursors, scale, schemes, style
from .version import __version__
from . import win_chrome
from . import wheels
from .curves_ui import CurvesEditor, CHANNELS, HUE_CHANNELS, ALL_CHANNELS
from .widgets import (IMAGE_SUFFIXES, ColorWheel, Histogram, ImageView, RefSlots, ScopeButton,
                      SectionHeader, SliderRow, Vectorscope, small_button,
                      _HelpPopupTracker, _HelpPressFilter)
from .dlg_common import (COLUMN_1, COLUMN_2, LABELS, POP_RECIPE, WHEELS, VISIBLE,
                         DEFAULT_ON, METHODS, CHAIN_ORDER, _DEBOUNCE_MS, _CROP_IDLE_MS,
                         load_sliders, load_help, load_auto_help)
from .dlg_sections import (SectionsMixin, _separator, _group, _apply_label_width,
                           _curves_section, _hsl_track, _hsl_section,
                           HSL_BANDS, HSL_HUE, HSL_MODES)
from .dlg_chain import ChainOrderPopup
from .dlg_help_popup import HelpPopup, HelpMixin, _help_html  # noqa: F401  (Tests importieren HelpPopup aus dialog)
from .dlg_layout import LayoutMixin
from .dlg_lut import LutMixin
from .dlg_panel_order import PanelOrderMixin
from .dlg_color_replace import ColorReplaceSection
from .dlg_reference import ReferenceMixin
from .dlg_worker import WorkerMixin, _ResultSignal, _WorkerTask, _histogram_of
from .dlg_wb import WhiteBalanceMixin, _srgb_hue


def _fmt_num(v: float) -> str:
    """Format a number with period as decimal separator, at least one decimal place."""
    s = f"{v:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s



class RefGradeDialog(SectionsMixin, LayoutMixin, LutMixin, ReferenceMixin,
                     WorkerMixin, WhiteBalanceMixin, PanelOrderMixin, HelpMixin, QDialog):
    """Maximised black window; layout per KONZEPT.md §3 (user sketch, second version)."""

    def __init__(self, image: QImage | None = None, parent: QWidget | None = None,
                 open_image: Callable[[], str | None] | None = None, *,
                 client: Callable | None = None, source: dict | None = None,
                 save_lut: Callable | None = None, on_ok: Callable | None = None,
                 on_update: Callable | None = None,
                 choose_folder: Callable[[str], str | None] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"REFgrade {__version__}")
        self.setStyleSheet(style.STYLESHEET)
        self.setFont(style.font("base"))
        self.open_image = open_image or self._ask_image
        self.sliders = load_sliders()
        # Tests run with REFGRADE_SETTINGS set, so they never read or write the user's slots.
        # Offscreen (test env): isolate QSettings per instance to prevent cross-test pollution.
        if os.environ.get("REFGRADE_SETTINGS"):
            _app = os.environ["REFGRADE_SETTINGS"]
        elif os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            _app = f"RefGrade_test_{id(self)}"
        else:
            _app = "RefGrade"
        self._settings = QSettings("RefGrade", _app)
        self._scheme_name = schemes.resolve_name(
            self._settings.value("scheme", schemes.DEFAULT, type=str))
        # Panel order (= chain order): load from QSettings; validate
        _default_panel = list(CHAIN_ORDER) + ["color_replace"]
        _co = self._settings.value("chain_order", None, type=list)
        if _co is not None and set(_co) == set(_default_panel) and len(_co) == len(_default_panel):
            self._panel_order = list(_co)
        elif _co is not None and set(_co) == set(CHAIN_ORDER) and len(_co) == len(CHAIN_ORDER):
            # Old 8-item format: insert color_replace at end
            self._panel_order = list(_co) + ["color_replace"]
        else:
            self._panel_order = list(_default_panel)
        self._chain_order = [k for k in self._panel_order if k in CHAIN_ORDER]
        self._building = True
        try:
            self._build()
        finally:
            self._building = False
        schemes.apply_scheme(self, schemes.SCHEMES[self._scheme_name])
        # K22: color is most recently touched so _enforce_no_scroll won't collapse it first.
        # K35a: the collapse *priority* is the pre-K35a scroll order, fixed and independent of
        # the new panel (chain) order — otherwise the reopen pass (latest-collapsed first) would
        # try color_replace before lut and leave lut collapsed when it still fits (test_plugin_k25c).
        self._touch_order = ["tone", "rgb", "hsl", "color_replace", "curves", "lut",
                             "reference", "color"]

        # ── wiring state ──────────────────────────────────────────────────────
        self._client = client
        self._source = source
        if source and source.get("path"):
            self._btn_load.setEnabled(True)
            _sc = QShortcut(QKeySequence("Ctrl+V"), self)
            _sc.activated.connect(self._load_from_clipboard)
        self._save_lut = save_lut
        self._on_ok = on_ok
        self._on_update = on_update
        self._choose_folder = choose_folder
        self._session_id: str | None = source.get("session") if source else None
        self._port = config.PORT
        self._req_counter = 0
        self._req_path: dict[int, str] = {}
        self._latest: dict[str, int] = {}
        self._crop_rect = None
        self._saveas_out: str | None = None
        # No parent: a worker reply may arrive after the window is gone; it must reach a live
        # object (else a crash — closing during a running preview). done() disconnects it.
        self._result_sig = _ResultSignal()
        self._result_sig.result.connect(self._on_result)
        self._original_image = image
        self._worker_original_requested = False
        self._showing_original = False
        self._preview_scale = 1.0
        self._orig_w = source.get("width", 0) if source else 0
        self._orig_h = source.get("height", 0) if source else 0
        # Re-resolve AUTO now that the image dimensions are known (build used 0×0).
        self._apply_layout_mode()
        self._wheel_rest: dict[str, tuple] = {k: (0.0, 0.0, 0.0) for k in
                                              ("lift", "gamma", "gain")}
        self._lut_dir_last: str | None = None
        self._busy = False
        self._preview_dirty = False

        # Debounce timer for preview
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._trigger_preview)

        # Idle timer for crop
        self._crop_timer = QTimer(self)
        self._crop_timer.setSingleShot(True)
        self._crop_timer.setInterval(_CROP_IDLE_MS)
        self._crop_timer.timeout.connect(self._trigger_crop)

        # K27e: LUT hover debounce timer
        self._lut_hover_timer = QTimer(self)
        self._lut_hover_timer.setSingleShot(True)
        self._lut_hover_timer.timeout.connect(self._lut_hover_timeout)
        self._lut_hover_pending = -1

        # K27e: loading bar timer (updates progress every 100 ms)
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(100)
        self._loading_timer.timeout.connect(self._loading_tick)

        # K27f: two-stage preview state
        self._dragging = False
        self._full_pending = False
        self._full_timer = QTimer(self)
        self._full_timer.setSingleShot(True)
        self._full_timer.setInterval(120)
        self._full_timer.timeout.connect(self._send_full_preview)
        self._loading_start = 0.0
        self._loading_est_ms = 0.0

        # Event filter for tile hover and left-button original
        self.slots.tile_added.connect(self._wire_tile)
        for t in self.slots.tiles:
            self._wire_tile(t)
        self.view.installEventFilter(self)
        self.view.split_button_clicked.connect(self._toggle_split)
        self.view.clip_button_clicked.connect(self._toggle_clip_view)
        self._last_preview: QImage | None = None
        self._hover_tile = None

        # ── initial content ───────────────────────────────────────────────────
        if client is None:
            self._fill_slots_from_pool()
            if image is not None and not image.isNull():
                self.view.set_image(image)
                hist = _histogram_of(image)
                self.histogram.set_data({"source": hist, "result": hist})
            self.view.set_status("LAYOUT DUMMY", "LAYOUT DUMMY · no processing")
        else:
            if image is not None and not image.isNull():
                self.view.set_image(image, QSizeF(self._orig_w or image.width(),
                                                  self._orig_h or image.height()))
                hist = _histogram_of(image)
                self.histogram.set_data({"source": hist, "result": hist})
            self._load_manual_tiles()
            self._fill_empty_slots()
            if not any(t.active for t in self.slots.tiles):
                first = next((t for t in self.slots.tiles if t.path), None)
                if first is not None:            # all slots already set: still start grading
                    first.set_active(True)
            self._schedule_preview()
        self._built = True        # from now on a changed value switches its section ON

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # K25b: no top-level layout — all 4 blocks are direct children positioned
        # by setGeometry in resizeEvent (no reparenting, no crash on mode switch).

        # ── Panel (control column, 3 squares wide) ───────────────────────────
        self.panel = QWidget(self)
        left = QVBoxLayout(self.panel)
        left.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(style.GAP)

        # K25i: MASTER row with general "?" button (first row of column)
        self._general_help_btn = QPushButton("?")
        self._general_help_btn.setObjectName("general_help")
        self._general_help_btn.setFont(style.font("title"))
        self._general_help_btn.setProperty("font_role", "title")
        _rh = scale.row_h(style.current_s())
        self._general_help_btn.setFixedSize(_rh, _rh)
        self._general_help_btn.setProperty("help", True)
        self._general_help_btn.setToolTip("Help: Hover over any control for its function")
        self._general_help_btn.clicked.connect(lambda: self._on_help_requested("general"))
        self._scheme_combo = QComboBox()
        self._scheme_combo.setObjectName("scheme_combo")
        self._scheme_combo.addItems(list(schemes.SCHEMES))
        self._scheme_combo.setCurrentText(self._scheme_name)
        _cf = style.font("small")
        self._scheme_combo.setFont(_cf)
        self._scheme_combo.setProperty("font_role", "small")
        _cfm = QFontMetrics(_cf)
        _cw = max(_cfm.horizontalAdvance(n) for n in schemes.SCHEMES) + 30
        self._scheme_combo.setFixedSize(_cw, _rh)
        self._scheme_combo.setToolTip("UI colour scheme")
        self._scheme_combo.currentTextChanged.connect(self._set_scheme)
        self._chain_order_btn = QPushButton("⇅")
        self._chain_order_btn.setObjectName("chain_order_btn")
        self._chain_order_btn.setFont(style.font("title"))
        self._chain_order_btn.setProperty("font_role", "title")
        self._chain_order_btn.setFixedSize(_rh, _rh)
        self._chain_order_btn.setProperty("help", True)
        self._chain_order_btn.setToolTip(
            "Processing order: sections are applied top to bottom, each on the result of the one above. Drag to reorder.")
        self._chain_order_btn.clicked.connect(self._on_chain_order_clicked)
        # no label: the wordmark block already says MASTER and the slider sits right next to it
        self.master = SliderRow("master", 0.0, 1.0, 1.0, 0.01, label_width=0)
        self.master.setObjectName("master")
        self.master.changed.connect(self._mode_changed)
        self._master_row = QWidget()
        self._master_row.setFixedHeight(scale.row_h(style.current_s()))
        master_row_lay = QHBoxLayout(self._master_row)
        master_row_lay.setContentsMargins(0, 0, 0, 0)
        master_row_lay.setSpacing(4)
        master_row_lay.addWidget(self._build_wordmark())
        master_row_lay.addWidget(self.master, 1)
        master_row_lay.addWidget(self._chain_order_btn)
        master_row_lay.addWidget(self._general_help_btn)
        master_row_lay.addWidget(self._scheme_combo)
        left.addWidget(self._master_row)

        # Scroll list: WB·COLOR, TONE, HSL, CURVES, LUT, REFERENCE
        # (K24: RGB moved to the bottom strip, wheels moved to the wheel strip)
        self._scroll = QWidget(self)
        scroll_layout = QVBoxLayout(self._scroll)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(style.GAP)

        # Color (WB · COLOR) — first section
        color_w, color_h, color_rows, color_content = _group(
            "WB · COLOR", COLUMN_2[0][1], self.sliders, key="color")
        scroll_layout.addWidget(color_w)

        # Tone
        tone_w, tone_h, tone_rows, tone_content = _group(
            "Tone", COLUMN_1[1][1], self.sliders)
        scroll_layout.addWidget(tone_w)

        # RGB (normal chapter in the panel)
        self._rgb_h = self._build_rgb_horizontal()
        scroll_layout.addWidget(self._rgb_h)

        # HSL
        hsl_w, hsl_h, self._hsl_sliders, hsl_content = _hsl_section(self.sliders)
        scroll_layout.addWidget(hsl_w)

        # Color Replace
        self._cr_section = ColorReplaceSection(self._scroll)
        self._cr_section.changed.connect(lambda *_a: self._mark_changed("color_replace"))
        self._cr_section.changed.connect(self._mode_changed)
        self._cr_section.pipette_requested.connect(self._cr_pipette_mode)
        self._cr_picking = False
        self._cr_pick_index = 0
        scroll_layout.addWidget(self._cr_section)

        # Curves
        curves_w, curves_h, self.curves_editor, curves_content = _curves_section(self.sliders)
        scroll_layout.addWidget(curves_w)

        # Wheels: chapter lives in the wheel strip beside/under the wheels (user 2026-10-11,
        # reverses K25b); parented to the strip once that exists, see below
        self._wheel_section = self._build_wheels_section()

        # LUT
        lut_w = self._build_lut_section()
        scroll_layout.addWidget(lut_w)

        # Reference (with tiles + method)
        ref_w, ref_h, ref_rows, ref_content = _group(
            "Reference", COLUMN_1[0][1], self.sliders)
        scroll_layout.addWidget(ref_w)
        # K25c: free space collects at the bottom of the scroll area
        scroll_layout.addStretch(1)
        left.addWidget(self._scroll, 1)
        self._load_lut_folder()

        # K25i: apply label width per section (no truncation)
        _apply_label_width(color_rows)
        _apply_label_width(tone_rows)
        for _band_rows in self._hsl_sliders.values():
            _apply_label_width(_band_rows)
        _apply_label_width(ref_rows)

        # Wire Tone extras (POP button, no-LUT label)
        self._add_pop_button({"tone": (tone_h, tone_rows)})
        self._add_not_in_lut_label({"tone": (tone_h, tone_rows)})
        # Tiles + method row inside Reference content
        self.slots = RefSlots()
        self.slots.dropped_multi.connect(self._on_dropped_multi)
        method_row = self._method_row()
        ref_cbox = ref_content.layout()
        ref_cbox.insertWidget(2, self.slots)
        ref_cbox.insertWidget(3, method_row)

        # K25c: LOAD / CANCEL / RESET / UPDATE / SAVE AS / SAVE at the bottom
        _btn_h = round(scale.button_h(style.current_s()) * 1.5)
        self._btn_load = QPushButton("LOAD")
        self._btn_load.setObjectName("btn_load")
        self._btn_load.setProperty("main", "reset")
        self._btn_load.setToolTip("Load a new source image (standalone only)")
        self._btn_load.setEnabled(False)
        self._btn_cancel = QPushButton("CANCEL")
        self._btn_cancel.setObjectName("btn_cancel")
        self._btn_cancel.setProperty("main", "cancel")
        self._btn_cancel.setToolTip("Close the dialog without applying any changes")
        self._btn_reset = QPushButton("RESET")
        self._btn_reset.setObjectName("btn_reset")
        self._btn_reset.setProperty("main", "reset")
        self._btn_reset.setToolTip("Reset all sections to their default values")
        self._btn_update = QPushButton("UPDATE")
        self._btn_update.setObjectName("btn_update")
        self._btn_update.setProperty("main", "reset")
        self._btn_update.setToolTip("Check GitHub for a newer version and update if available.")
        self._btn_saveas = QPushButton("SAVE AS")
        self._btn_saveas.setObjectName("btn_saveas")
        self._btn_saveas.setProperty("main", "cancel")
        self._btn_saveas.setToolTip("Save the graded result to a chosen folder")
        self._btn_ok = QPushButton("SAVE")
        self._btn_ok.setObjectName("btn_ok")
        self._btn_ok.setProperty("main", "cancel")
        self._btn_ok.setToolTip("Save the grading as a new layer above the active one (the original layer stays untouched).")
        for b in (self._btn_load, self._btn_cancel, self._btn_reset, self._btn_update,
                  self._btn_saveas, self._btn_ok):
            b.setFont(style.font("title", QFont.Weight.Normal))
            b.setProperty("font_role", "title")
            b.setFixedHeight(_btn_h)
        self._btn_ok.setDefault(True)
        self._btn_load.clicked.connect(self._do_load_menu)
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_reset.clicked.connect(self._reset_all)
        self._btn_update.clicked.connect(self._do_update)
        self._btn_saveas.clicked.connect(self._do_saveas)
        self._btn_ok.clicked.connect(self._do_ok)
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        btn_row.addWidget(self._btn_load, 1)
        btn_row.addWidget(self._btn_cancel, 1)
        btn_row.addWidget(self._btn_reset, 1)
        btn_row.addWidget(self._btn_update, 1)
        btn_row.addWidget(self._btn_saveas, 1)
        btn_row.addWidget(self._btn_ok, 2)
        # status line (was a strip under the image) sits in the gap above the main buttons
        self._status_label = QLabel("")
        self._status_label.setObjectName("status_label")
        self._status_label.setFont(style.font("small"))
        self._status_label.setProperty("font_role", "small")
        self._status_label.setProperty("muted", True)
        self._status_label.setFixedHeight(scale.row_h(style.current_s()))
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        left.addWidget(self._status_label)
        left.addLayout(btn_row)

        # ── Wheel strip (direct child, only 4 squares) ──
        self._wheel_strip = self._build_wheel_strip()
        # a direct child of the dialog, not of the strip: on small screens it may be wider
        # than the V wheel column and must not be clipped by it (2026-10-11)
        self._wheel_section.setParent(self)

        # ── Scope block (direct child, buttons + histogram) ──
        self._scope_block = self._build_scope_block()

        # ── Image view (direct child) ──
        self.view = ImageView(self)
        # clipping warning box removed (user 2026-10-12: ugly, red even with every section off);
        # replaced by the "C" clipping view in K34a (spec 88)
        self.view.setObjectName("image_view")
        self.view.set_status_h(0)          # image gets the full height (user 2026-10-11)
        self.view.status_changed.connect(self._show_status)

        # K28f: one-time reset of layout_mode to AUTO (marker in QSettings)
        if not self._settings.value("layout_mode_reset_2026_10_10", False, type=bool):
            self._settings.setValue("layout_mode", "AUTO")
            self._settings.setValue("layout_mode_reset_2026_10_10", True)
        self._layout_mode = self._settings.value("layout_mode", "AUTO", type=str)
        self._apply_layout_mode()

        # Section registry
        self.sections = {
            "color": (color_h, color_rows),
            "tone": (tone_h, tone_rows),
            "rgb": (self._rgb_header, self._rgb_rows),
            "hsl": (hsl_h, []),
            "color_replace": (self._cr_section.header, []),
            "curves": (curves_h, []),
            "wheels": (self._wheel_header,
                       [self._wheel_amount, self._wheel_intensity_slider,
                        self._wheel_gain_slider, self._wheel_falloff_slider]),
            "lut": (self._lut_header, []),
            "reference": (ref_h, ref_rows),
        }
        self._contents = {
            "color": color_content, "tone": tone_content, "rgb": self._rgb_content,
            "hsl": hsl_content, "color_replace": self._cr_section._content,
            "curves": curves_content,
            "wheels": self._wheel_content,
            "lut": self._lut_content, "reference": ref_content,
        }
        # Sections in the scrolling list (wheels is in the wheel strip, not the scroll)
        self._scroll_keys = tuple(k for k in self._panel_order if k != "wheels")
        self._scroll_widgets = {
            "color": color_w, "tone": tone_w, "rgb": self._rgb_h,
            "hsl": hsl_w, "color_replace": self._cr_section, "curves": curves_w,
            "lut": lut_w, "reference": ref_w,
        }
        self._drag_key: str | None = None
        self._drag_indicator: QWidget | None = None
        self._apply_panel_order()

        self.curves_editor.changed.connect(self._curves_changed)
        # Wire all section signals
        self._help_popup: HelpPopup | None = None
        self._chain_order_popup: ChainOrderPopup | None = None
        # Frozen-help fix (2026-10-17): open state lives in a flag the popup's auto-close
        # does not clear; _HelpPressFilter clears it on a genuine click-away.
        self._help_tracker = _HelpPopupTracker(
            [self._general_help_btn] + [h.help_btn for h, _r in self.sections.values()],
            lambda: self._help_popup)
        self._help_press_filter = _HelpPressFilter(self._help_trackers)
        for header, rows in self.sections.values():
            header.weighted.connect(self._mode_changed)
            header.switched.connect(self._section_switched)
            header.reset.connect(self._reset_section)
            header.collapsed_changed.connect(self._section_collapsed_changed)
            header.help_requested.connect(self._on_help_requested)
            header.auto_requested.connect(self._on_auto_requested)
            header.drag_started.connect(self._on_section_drag_started)
            header.drag_moved.connect(self._on_section_drag_moved)
            header.drag_ended.connect(self._on_section_drag_ended)
            header.reset_order_requested.connect(self._on_reset_chain_order)
            for r in rows:
                r.changed.connect(lambda *_a, k=header.key: self._mark_changed(k))
                r.changed.connect(self._mode_changed)
            header.weighted.connect(lambda *_a, k=header.key: self._mark_changed(k))
        # Wire HSL slider signals
        for band, (hue, sat, light) in self._hsl_sliders.items():
            for r in (hue, sat, light):
                r.changed.connect(lambda *_a: self._mark_changed("hsl"))
                r.changed.connect(self._mode_changed)
        # K27f: wire slider drag signals for two-stage preview
        from .widgets import _RelativeSlider
        for sl in self.findChildren(_RelativeSlider):
            sl.drag_started.connect(self._on_drag_started)
            sl.drag_stopped.connect(self._on_drag_stopped)
        # K31b-F: drag signals from ColorWheel and CurvesEditor
        for w in self.findChildren(ColorWheel):
            w.drag_started.connect(self._on_drag_started)
            w.drag_stopped.connect(self._on_drag_stopped)
        if hasattr(self, "curves_editor"):
            self.curves_editor.drag_started.connect(self._on_drag_started)
            self.curves_editor.drag_stopped.connect(self._on_drag_stopped)
        for key in self.sections:
            self._show_section_state(key)
        # K25c: all sections expanded except reference
        for key in self.sections:
            self._set_section_collapsed(key, key == "reference")

        # Split view: the button is wired once, above (a second connect here toggled every
        # click twice, so split could not be switched off — 2026-10-11)
        # Restore split mode from QSettings
        if self._settings.value("split_mode", False, type=bool):
            self.view.split_mode = True
        # Restore clip view from QSettings
        self._clip_view = self._settings.value("clip_view", False, type=bool)
        self.view._clip_view = self._clip_view
        # Wire HSL band buttons and Curves channel buttons (after the panel is in the
        # widget hierarchy, so findChild can locate them)
        self._hsl_mode_btns: dict[str, QPushButton] = {}
        for mode, _text in HSL_MODES:
            btn = self.findChild(QPushButton, f"hsl_mode_{mode}")
            if btn:
                btn.clicked.connect(lambda _c=False, m=mode: self._hsl_mode_selected(m))
                self._hsl_mode_btns[mode] = btn
        self._curves_ch_btns: dict[str, QPushButton] = {}
        for ch in ALL_CHANNELS:
            btn = self.findChild(QPushButton, f"curves_ch_{ch}")
            if btn:
                btn.clicked.connect(lambda _c=False, c=ch: self._curves_channel_selected(c))
                self._curves_ch_btns[ch] = btn
        reset_btn = self.findChild(QPushButton, "curves_ch_reset")
        if reset_btn:
            reset_btn.clicked.connect(self._curves_channel_reset)

        self._build_wb_row()
        self._apply_tooltips()


    def _apply_tooltips(self) -> None:
        """Set objectName and tooltip on every widget from ui_help + HELP + SLIDERS."""
        from .ui_help import UI_HELP, slider_tooltip
        help_data = load_help()
        by_name = {s.name: s for s in self.sliders}

        # Section headers: set objectName on ON/OFF, undo (before findChild)
        for key, (header, _rows) in self.sections.items():
            header.on.setObjectName(f"sec_{key}_on")
            header.undo.setObjectName(f"sec_{key}_undo")

        # Slider rows: tooltip = short (bold) + tip
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                r.setToolTip(slider_tooltip(help_data.get(r.name, {})))
        # HSL sliders (stored separately, not in sections rows)
        for band, (hue, sat, light) in self._hsl_sliders.items():
            for r in (hue, sat, light):
                r.setToolTip(slider_tooltip(help_data.get(r.name, {})))

        # Non-slider widgets from UI_HELP table
        for obj_name, text in UI_HELP.items():
            w = self.findChild(QWidget, obj_name)
            if w is not None:
                w.setToolTip(text)


    def mode(self) -> dict:
        """Sections ON are applied one after another in the chain order; Amount blends each step in."""
        weights = {k: h.value() for k, (h, _r) in self.sections.items()
                   if h.is_on() and h.value() > 0}
        m: dict = {"solo": None, "weights": weights, "master": self.master.value(),
                   "order": list(self._chain_order)}
        lut_header = self.sections.get("lut", (None, None))[0]
        if lut_header is not None and lut_header.is_on() and self._lut_current_path:
            m["lut"] = self._lut_current_path
        # Curves: include when ON and non-identity
        curves_h = self.sections.get("curves", (None, None))[0]
        if curves_h is not None and curves_h.is_on():
            curves = {ch: self.curves_editor.get_curve(ch) for ch in ALL_CHANNELS}
            # Filter out identity curves
            non_id = {}
            for ch, pts in curves.items():
                if ch in HUE_CHANNELS:
                    if not all(abs(p[1] - 0.5) < 1e-6 for p in pts):
                        non_id[ch] = pts
                else:
                    if not (len(pts) == 2 and pts[0] == [0.0, 0.0] and pts[1] == [1.0, 1.0]):
                        non_id[ch] = pts
            if non_id:
                m["curves"] = non_id
        return m


    def _show_mode(self) -> None:
        m = self.mode()
        mix = " + ".join(f"{k.upper()} {round(w * 100)}%" for k, w in m["weights"].items()) or "NOTHING ON"
        self._set_status(f"LAYOUT DUMMY · MASTER {round(m['master'] * 100)}% · {mix}")


    def _mode_changed(self, *_args) -> None:
        if self._client is None:
            self._show_mode()
        else:
            self._schedule_preview()


    def _show_status(self, text: str) -> None:
        """View status (text · zoom) in the label above the main buttons, elided to fit."""
        try:
            lbl = self._status_label
            lbl.setToolTip(text)
            lbl.setText(lbl.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight,
                                                     max(lbl.width(), 40)))
        except RuntimeError:          # dialog torn down while a status was still queued
            pass


    # ── LOAD / SAVE AS ─────────────────────────────────────────────────────────

    def _do_load_menu(self) -> None:
        m = QMenu(self)
        m.addAction("Open file…", self._load_from_file)
        m.addAction("Paste screenshot  (Ctrl+V)", self._load_from_clipboard)
        m.addAction("Folder…", self._load_from_folder)
        m.exec(self._btn_load.mapToGlobal(self._btn_load.rect().bottomLeft()))

    def _load_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Bild öffnen", str(config.ROOT),
            "Bilder (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
        if path:
            self._load_path(Path(path))

    def _load_from_clipboard(self) -> None:
        from PyQt6.QtGui import QClipboard
        cb = QApplication.clipboard()
        if cb is None:
            return
        img = cb.image()
        if img is None or img.isNull():
            self._set_status("Kein Bild in der Zwischenablage")
            return
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_paste_"))
        p = tmp / "pasted.png"
        img.save(str(p), "PNG")
        self._load_path(p)

    def _load_from_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Ordner wählen", str(config.ROOT))
        if not d:
            return
        d = Path(d)
        files = sorted(
            f for f in d.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
        )
        if not files:
            self._set_status("Keine Bilddateien im Ordner")
            return
        self._load_path(files[0])

    def _load_path(self, path: Path) -> None:
        if not path.is_file():
            return
        if self._source is None or self._on_update is None:
            return
        self._source["path"] = str(path.resolve())
        self._on_update()

    def _set_status(self, text: str) -> None:
        """K25i: status in the bottom strip; full text as tooltip."""
        self.view.set_status(text, text)

    # ── mix / on-off / reset ──────────────────────────────────────────────────

    def wheel_values(self, kind: str) -> dict[str, float]:
        """The four core slider values of one wheel (point + master + rest)."""
        wh = self.wheels[kind]
        return wheels.to_values(kind, wh.u, wh.v, self.masters[kind].value(),
                                self._wheel_rest[kind])


    def _on_help_requested(self, key: str) -> None:
        """Toggle the help popup for `key`. Reads the tracker flag, not isVisible():
        the '?' click auto-closes the Qt.Popup during the press (before clicked() runs),
        so a naive isVisible() toggle would re-open it (frozen help, 2026-10-17)."""
        if self._help_tracker.open_value() == key:
            self._help_tracker.set_open(None)
            self._close_help_popup()
            self._set_help_open(key, False)
            return
        self._close_help_popup()
        # black square centred over the picture, always (user 2026-10-11): the help sits where
        # the eye is, never over the controls being explained
        vr = self.view.rect()
        side = max(320, int(min(vr.width(), vr.height()) * 0.92))
        popup = HelpPopup(key, side, self)
        popup.setFixedHeight(side)
        popup.closed.connect(self._on_help_closed)
        centre = self.view.mapToGlobal(vr.center())
        popup.move(centre.x() - side // 2, centre.y() - side // 2)
        popup.show()
        popup.raise_()
        self._help_popup = popup
        self._help_tracker.set_open(key)
        self._set_help_open(key, True)


    def _on_help_closed(self, key: str) -> None:
        """Popup closed (Esc/click-away): reset the border. Does NOT clear the tracker
        flag (it is what makes the next '?' click read as a close); _HelpPressFilter
        clears it on click-away."""
        self._set_help_open(key, False)
        self._help_popup = None


    def _set_scheme(self, name: str) -> None:
        """Switch the UI colour scheme, persist it, and re-apply palette + stylesheet."""
        if name not in schemes.SCHEMES:
            name = schemes.DEFAULT
        self._scheme_name = name
        self._settings.setValue("scheme", name)
        schemes.apply_scheme(self, schemes.SCHEMES[name])
        self._restyle_chrome()
        self._restyle_wordmark()
        win_chrome.apply(self, schemes.SCHEMES[name])


    def _set_help_open(self, key: str, open: bool) -> None:
        """Set the red-border state on the help button."""
        btn = (self.sections[key][0].help_btn if key in self.sections
               else self._general_help_btn)
        btn.setProperty("open", open)
        btn.style().unpolish(btn)
        btn.style().polish(btn)


    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
                self.show_screen()
            else:
                self.showFullScreen()
        elif e.key() == Qt.Key.Key_S and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._toggle_split()
        elif e.key() == Qt.Key.Key_C and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._toggle_clip_view()
        else:
            super().keyPressEvent(e)          # Esc → reject (QDialog)


    def done(self, r: int) -> None:  # noqa: D401
        """Close (OK / Cancel / Esc): stop listening to replies still in flight."""
        try:
            self._result_sig.result.disconnect(self._on_result)
        except (TypeError, RuntimeError):
            pass
        self._remove_help_press_filter()
        super().done(r)


    def _cr_pipette_mode(self, index: int, which: str) -> None:
        """Enable image sampling for Color Replace instance `index`, target src or tgt."""
        self._cr_dbg(f"_cr_pipette_mode called: index={index} which={which}")
        self._cr_picking = True
        self._cr_pick_index = index
        self._cr_pick_which = which
        self.setCursor(cursors.pipette() if which == "src" else cursors.crosshair())
        label = "SOURCE" if which == "src" else "TARGET"
        self._set_status(f"Color Replace · click to pick {label} color (instance {index + 1})")

    def _cr_dbg(self, msg: str) -> None:
        try:
            import datetime
            log_path = Path(os.environ.get("TEMP", ".")) / "refgrade_cr_debug.log"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now().isoformat()} {msg}\n")
        except Exception:
            pass

    def _cr_pick_at(self, pos) -> None:
        """Click on the image in CR-pick mode → read pixel → set src or tgt wheel."""
        self._cr_picking = False
        self.unsetCursor()
        img = self._last_preview
        if img is None or img.isNull():
            img = self._original_image
        if img is None or img.isNull():
            self._set_status("Color Replace · kein Bild zum Abtasten")
            return
        rgb = self._sample_view(pos, img)          # zoom/pan aware (dlg_wb)
        if rgb is None:
            return
        r, g, b = (c / 255.0 for c in rgb)
        self._cr_section.set_pipette_color(self._cr_pick_index, r, g, b, getattr(self, "_cr_pick_which", "src"))
        self._set_status(f"Color Replace · picked ({r:.2f}, {g:.2f}, {b:.2f})")

    def closeEvent(self, e) -> None:  # noqa: N802
        try:
            self._result_sig.result.disconnect(self._on_result)
        except (TypeError, RuntimeError):
            pass
        self._remove_help_press_filter()
        super().closeEvent(e)


    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.view:
            if getattr(self, "_cr_picking", False) and event.type() == QEvent.Type.MouseButtonPress:
                self._cr_dbg(f"eventFilter: CR-pick active, MouseButtonPress btn={event.button()}")
            if (event.type() == QEvent.Type.MouseButtonPress and getattr(self, "_cr_picking", False)
                    and event.button() == Qt.MouseButton.LeftButton):
                self._cr_pick_at(event.position())
                return True
            if (event.type() == QEvent.Type.MouseButtonPress and getattr(self, "_wb_picking", False)
                    and event.button() == Qt.MouseButton.LeftButton):
                self._wb_pick_at(event.position())
                return True
            if event.type() == QEvent.Type.MouseButtonPress:
                # Pipette: Ctrl+Click sets a point in the active Hue tab
                if (event.button() == Qt.MouseButton.LeftButton
                        and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self._pipette(event.position())
                    return True
                # In split mode, left click does NOT toggle original
                if (event.button() == Qt.MouseButton.LeftButton
                        and not self.view.split_mode
                        and self._original_image is not None
                        and not self._original_image.isNull()):
                    self._showing_original = True
                    self.view.show_overlay = False
                    self.view.replace_image(self._original_image, keep_overlay=True)
                    self.view.set_status("ORIGINAL", "ORIGINAL · original image shown")
            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton and self._showing_original:
                    self._showing_original = False
                    self._restore_computed()
            elif event.type() in (QEvent.Type.MouseMove, QEvent.Type.Wheel,
                                  QEvent.Type.KeyPress):
                self._crop_timer.stop()
                if self._client is not None:
                    self._maybe_schedule_crop()
        elif hasattr(obj, "index") and hasattr(obj, "path"):
            # RefTile hover
            if event.type() == QEvent.Type.Enter:
                self._tile_hover(obj)
            elif event.type() == QEvent.Type.Leave:
                self._tile_leave()
        return super().eventFilter(obj, event)


    def _block_signals(self) -> None:
        """Block all slider/wheel signals (for FIT)."""
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                r.slider.blockSignals(True)
        for band, (hue, sat, light) in self._hsl_sliders.items():
            for r in (hue, sat, light):
                r.slider.blockSignals(True)
        for kind in ("lift", "gamma", "gain"):
            self.wheels[kind].changed.disconnect(self._wheel_changed)
            self.masters[kind].slider.blockSignals(True)


    def _unblock_signals(self) -> None:
        """Unblock all slider/wheel signals (after FIT)."""
        for key, (_h, rows) in self.sections.items():
            for r in rows:
                r.slider.blockSignals(False)
        for band, (hue, sat, light) in self._hsl_sliders.items():
            for r in (hue, sat, light):
                r.slider.blockSignals(False)
        for kind in ("lift", "gamma", "gain"):
            self.wheels[kind].changed.connect(self._wheel_changed)
            self.masters[kind].slider.blockSignals(False)

    # ── mark ──────────────────────────────────────────────────────────────────

    def _update_mark(self) -> None:
        """Set the status mark based on current state."""
        if self._client is None:
            return
        on = " + ".join(k.upper() for k in self.mode()["weights"]) or "NOTHING ON"
        base = f"GRADING · {self._method().upper()} · {self._first_ref_name()} · {on}"
        self.view.set_status(base, base)


    def _set_mark_error(self, msg: str) -> None:
        """Show a worker error in the status strip (grey mark, K27e)."""
        self.view.set_status(f"ERROR · {msg}", f"ERROR · {msg}")
        self.view.mark_color = QColor(style.MARK_ERROR)

    # ── FIT ───────────────────────────────────────────────────────────────────
