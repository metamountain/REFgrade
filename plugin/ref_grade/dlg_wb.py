"""WhiteBalanceMixin: WB row, auto, pick, apply, pipette (K26)."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from . import scale, style
from .curves_ui import HUE_CHANNELS
from .widgets import ImageView, SliderRow, small_button

def _srgb_hue(r: float, g: float, b: float) -> float:
    """LCh-Hue aus sRGB (0..1), dieselbe Formel wie core (kornia rgb_to_lab → atan2).

    Returns:
        Hue in 0..1 (0 = Rot, 0.25 = Gelb, 0.5 = Grün, …).
    """
    # sRGB → linear
    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    rl, gl, bl = _lin(r), _lin(g), _lin(b)
    # Linear RGB → XYZ (sRGB D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl
    # XYZ → Lab (D65 white point)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    _EPS_LAB = (6.0 / 29.0) ** 3
    _K_LAB = 3.0 * (6.0 / 29.0) ** 2 + 4.0 / 29.0
    def _f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > _EPS_LAB else t / _K_LAB + 4.0 / 29.0
    fx, fy, fz = _f(X / Xn), _f(Y / Yn), _f(Z / Zn)
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    # LCh: h = atan2(b, a) / 2π, mapped to [0, 1)
    h = math.atan2(bb, a) / (2.0 * math.pi)
    return (h + 1.0) % 1.0




class WhiteBalanceMixin:
    """Mixin for RefGradeDialog: WB row, auto, pick, apply, pipette."""

    def _build_wb_row(self) -> None:
        """AUTO and PICK at the top of the Color section."""
        self._wb_picking = False
        row = QHBoxLayout()
        row.setSpacing(2)
        self._wb_auto_btn = small_button("AUTO WB", tip="")
        self._wb_auto_btn.setObjectName("wb_auto")
        self._wb_pick_btn = small_button("PICK WB", "pick", checkable=True)
        self._wb_pick_btn.setObjectName("wb_pick")
        for b in (self._wb_auto_btn, self._wb_pick_btn):
            b.setFixedHeight(scale.row_h(style.current_s()))                     # (handlers check for a worker)
            row.addWidget(b)
        self._contents["color"].layout().insertLayout(0, row)
        self._wb_auto_btn.clicked.connect(self._wb_auto)
        self._wb_pick_btn.toggled.connect(self._wb_pick_mode)


    def _wb_color_on(self) -> None:
        header = self.sections["color"][0]
        if not header.is_on():
            header.on.setChecked(True)


    def _wb_auto(self) -> None:
        if self._client is None or self._session_id is None:
            return
        self._wb_color_on()
        self._run_task("/whitebalance", {"session": self._session_id, "mode": "auto"},
                       self._next_req(), route="wb")


    def _wb_pick_mode(self, on: bool) -> None:
        self._wb_picking = on
        if on:
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            self._set_status("PICK WB · click a spot that should be neutral")
        else:
            self.view.unsetCursor()
            self._update_mark()


    def _wb_pick_at(self, pos) -> None:
        """Click on the image in pick mode → full-image point → /whitebalance (pick)."""
        v = self.view
        x, y = v.offset.x() + pos.x() / v.zoom, v.offset.y() + pos.y() / v.zoom
        self._wb_pick_btn.setChecked(False)              # one click, then off
        if not (0 <= x < v.logical.width() and 0 <= y < v.logical.height()):
            return
        if self._client is None or self._session_id is None:
            return
        self._wb_color_on()
        self._run_task("/whitebalance", {"session": self._session_id, "mode": "pick",
                                         "point": [x, y]}, self._next_req(), route="wb")


    def _apply_wb(self, result: dict) -> None:
        rows = {r.name: r for r in self.sections["color"][1]}
        for name in ("temperature", "tint"):
            if name in result and name in rows:
                rows[name].set_value(float(result[name]))
        self._set_status(f"WB · temperature {result.get('temperature', 0):+.2f} · "
                         f"tint {result.get('tint', 0):+.2f}")


    def _sample_view(self, pos, img) -> tuple[int, int, int] | None:
        """Pixel of `img` (the whole picture at preview size) under widget point `pos`, through
        the view's zoom/pan like _wb_pick_at. The old letterbox maths assumed the fitted view:
        zoomed in or panned, the pipettes read another pixel (debug round 2026-10-17)."""
        v = self.view
        if img is None or img.isNull() or v.logical.isEmpty() or v.zoom <= 0:
            return None
        x = v.offset.x() + pos.x() / v.zoom
        y = v.offset.y() + pos.y() / v.zoom
        if not (0 <= x < v.logical.width() and 0 <= y < v.logical.height()):
            return None
        px = min(int(x * img.width() / v.logical.width()), img.width() - 1)
        py = min(int(y * img.height() / v.logical.height()), img.height() - 1)
        c = img.pixel(px, py)
        return (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF

    def _pipette(self, pos) -> None:
        """Ctrl+Click: liest Pixel aus _last_preview, setzt Punkt im Hue-Tab."""
        rgb = self._sample_view(pos, self._last_preview)
        if rgb is None:
            return
        r, g, b = rgb
        hue = _srgb_hue(r / 255.0, g / 255.0, b / 255.0)
        # Nur in Hue-Tabs
        if self.curves_editor.active_channel in HUE_CHANNELS:
            self.curves_editor.add_point_at(hue, 0.5)

    # ── event filter: tile hover + left-button original ──────────────────────
