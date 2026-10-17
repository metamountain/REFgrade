"""Widgets of the REF Grade window: reference slots, slider columns, histogram, image view.

No krita import. In the layout dummy nothing here computes a grade; the widgets only show
images and emit signals.
"""

from __future__ import annotations

import ctypes
import html as _html
import math
import time
from pathlib import Path

from PyQt6.QtCore import QElapsedTimer, QEvent, QPointF, QPoint, QRectF, QSize, QSizeF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QConicalGradient, QFont, QFontMetrics, QImage, QPainter,
                          QLinearGradient, QPainterPath, QPen, QPixmap, QRadialGradient)
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy,
                              QSlider, QStyle, QStyleOptionSlider, QToolTip, QVBoxLayout, QWidget)

from . import scale, style, wheels
from .dlg_common import load_help

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")

SHORT_TRACK_PX = 120
VS_EXAGGERATE = 3


def _build_vs_color_lut() -> tuple[tuple[int, int, int], ...]:
    """360-entry LUT: angle (degrees) → (R, G, B) at S=1, V=1."""
    lut = []
    for i in range(360):
        hue = ((math.radians(i) - math.pi / 2) / (2.0 * math.pi)) % 1.0
        c = QColor.fromHsvF(hue, 1.0, 1.0)
        lut.append((c.red(), c.green(), c.blue()))
    return tuple(lut)


VS_COLOR_LUT = _build_vs_color_lut()


# ── Reference slots ───────────────────────────────────────────────────────────

class RefTile(QWidget):
    """One square slot: image centre-cropped to fill, name below; red frame when active."""

    clicked = pyqtSignal(int, bool)          # index, ctrl held
    menu_requested = pyqtSignal(int, str)    # index, action ("replace"|"reload"|"clear")
    dropped = pyqtSignal(int, str)           # index, file path

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index
        self.path: str | None = None
        self.label = ""
        self.active = False
        self.pair_label = ""
        self.dimmed = False
        self._pix: QPixmap | None = None
        self._drag_hint = ""
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        _rh = scale.row_h(style.current_s())
        self.setMinimumSize(round(2.5 * _rh), round(2.5 * _rh))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 (Qt API)
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return w                              # always square

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(120, 120)

    def set_image(self, path: str | None, label: str = "") -> None:
        """Show `path` (or nothing); the thumbnail is display-only."""
        self.path, self.label = path, label
        self._pix = QPixmap(path) if path else None
        self.update()

    def set_active(self, on: bool) -> None:
        self.active = on
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        side = min(self.width(), self.height())
        r = QRectF(0, 0, side, side)
        if self._pix and not self._pix.isNull():
            src = self._pix.rect()
            s = min(src.width(), src.height())   # centre crop to a square
            crop = QRectF((src.width() - s) / 2, (src.height() - s) / 2, s, s)
            p.drawPixmap(r, self._pix, crop)
            p.fillRect(QRectF(0, side - 18, side, 18), style.TILE_SCRIM)
            p.setPen(QColor(style.TEXT))
            p.setFont(style.font("base"))
            p.drawText(r.adjusted(6, 0, -6, -3), Qt.AlignmentFlag.AlignBottom |
                       Qt.AlignmentFlag.AlignLeft, self.label)
            if self.pair_label:
                p.setPen(style.TILE_PAIR_LABEL)
                p.setFont(style.font("small"))
                p.drawText(r.adjusted(6, 0, -6, -20), Qt.AlignmentFlag.AlignBottom |
                           Qt.AlignmentFlag.AlignLeft, self.pair_label)
        else:
            p.setPen(QPen(QColor(style.LINE_STRONG), 1, Qt.PenStyle.DashLine))
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
            p.setPen(QColor(style.MUTED))
            p.setFont(style.font("big", QFont.Weight.Light))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "+")
        if self.active or self._drag_hint:
            p.setPen(QPen(QColor(style.ACCENT_HI), 1))
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
        if self._drag_hint:
            p.setPen(QColor(style.ACCENT_HI))
            p.setFont(style.font("base"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._drag_hint)
        if self.dimmed:
            p.fillRect(r, style.TILE_DIM)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self.clicked.emit(self.index, ctrl)

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        menu = QMenu(self)
        for text, key in (("Replace…", "replace"), ("Reload", "reload"), ("Clear", "clear")):
            act = menu.addAction(text)
            act.setData(key)
            act.setEnabled(key == "replace" or self.path is not None)
        chosen = menu.exec(e.globalPos())
        if chosen:
            self.menu_requested.emit(self.index, chosen.data())

    @staticmethod
    def _image_url(e) -> str | None:
        for url in e.mimeData().urls() if e.mimeData().hasUrls() else ():
            if url.isLocalFile() and url.toLocalFile().lower().endswith(IMAGE_SUFFIXES):
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if self._image_url(e):
            self._drag_hint = "REPLACE REFERENCE" if self.path else "LOAD REFERENCE"
            self.update()
            e.acceptProposedAction()

    def dragLeaveEvent(self, _e) -> None:  # noqa: N802
        self._drag_hint = ""
        self.update()

    def dropEvent(self, e) -> None:  # noqa: N802
        path = self._image_url(e)
        self._drag_hint = ""
        self.update()
        if path:
            self.dropped.emit(self.index, path)
            e.acceptProposedAction()


class RefSlots(QWidget):
    """Dynamic grid of square tiles; the edge follows the column width (flex)."""

    MAX_REFS = 12

    @property
    def TILE_TARGET(self) -> int:  # noqa: N802
        return round(style.current_s() * 0.45)

    dropped_multi = pyqtSignal(list)  # list of file paths dropped on the grid / "+" tile
    tile_added = pyqtSignal(object)   # emitted with the new RefTile after creation

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tiles: list[RefTile] = []
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(style.GAP)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @property
    def tiles(self) -> list[RefTile]:
        return self._tiles

    def set_count(self, n: int) -> None:
        """Resize the tile list to exactly n tiles (add or remove from the end)."""
        while len(self._tiles) < n:
            self._add_tile()
        while len(self._tiles) > n:
            self._remove_tile(len(self._tiles) - 1)
        self._relayout()

    def insert_tile(self, i: int) -> RefTile:
        """Insert a new empty tile at position i; reindex all tiles."""
        tile = RefTile(i, self)
        self._tiles.insert(i, tile)
        for j, t in enumerate(self._tiles):
            t.index = j
        self._relayout()
        self.tile_added.emit(tile)
        return tile

    def remove_tile(self, i: int) -> RefTile:
        """Remove the tile at position i; reindex the rest."""
        tile = self._tiles[i]
        self._grid.removeWidget(tile)
        tile.deleteLater()
        self._tiles.pop(i)
        for j, t in enumerate(self._tiles):
            t.index = j
        self._relayout()
        return tile

    def _add_tile(self) -> RefTile:
        tile = RefTile(len(self._tiles), self)
        self._tiles.append(tile)
        self.tile_added.emit(tile)
        return tile

    def _remove_tile(self, i: int) -> RefTile:
        tile = self._tiles[i]
        self._grid.removeWidget(tile)
        tile.deleteLater()
        self._tiles.pop(i)
        for j, t in enumerate(self._tiles):
            t.index = j
        return tile

    def cols_for_width(self, width: int) -> int:
        """Number of columns for the given width: clamp(floor((w+gap)/(TARGET+gap)), 2, 6)."""
        cols = (width + style.GAP) // (self.TILE_TARGET + style.GAP)
        return max(2, min(6, cols))

    def tile_side(self, width: int) -> int:
        """Tile edge: fill the width exactly across all columns."""
        cols = self.cols_for_width(width)
        return (width - (cols - 1) * style.GAP) // cols

    def _relayout(self) -> None:
        """Re-place tiles in the grid at the current width."""
        n = len(self._tiles)
        if n == 0:
            self._grid.setRowStretch(0, 1)
            self.setFixedHeight(0)
            return
        for t in self._tiles:
            self._grid.removeWidget(t)
        cols = self.cols_for_width(self.width() if self.width() > 0 else 320)
        for i, t in enumerate(self._tiles):
            self._grid.addWidget(t, i // cols, i % cols)
        self._apply(self.width() if self.width() > 0 else 320)

    def set_pair_labels(self, on: bool) -> None:
        """Show BEFORE/AFTER labels on the first two tiles when `on` is True."""
        for i, t in enumerate(self._tiles):
            if on and i == 0:
                t.pair_label = "BEFORE"
            elif on and i == 1:
                t.pair_label = "AFTER"
            else:
                t.pair_label = ""
            t.update()

    def _apply(self, width: int) -> None:
        if not self._tiles:
            self.setFixedHeight(0)
            return
        cols = self.cols_for_width(width)
        side = self.tile_side(width)
        rows = (len(self._tiles) + cols - 1) // cols
        for t in self._tiles:
            t.setFixedSize(side, side)
        self.setFixedHeight(rows * side + (rows - 1) * style.GAP)

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._apply(e.size().width())
        super().resizeEvent(e)

    @staticmethod
    def _image_urls(e) -> list[str]:
        """All valid image file paths from a drag event."""
        urls = e.mimeData().urls() if e.mimeData().hasUrls() else []
        return [u.toLocalFile() for u in urls
                if u.isLocalFile() and u.toLocalFile().lower().endswith(IMAGE_SUFFIXES)]

    def dragEnterEvent(self, e) -> None:  # noqa: N802
        if self._image_urls(e):
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:  # noqa: N802
        paths = self._image_urls(e)
        if paths:
            self.dropped_multi.emit(paths)
            e.acceptProposedAction()


# ── Sliders ───────────────────────────────────────────────────────────────────

class _RelativeSlider(QSlider):
    """QSlider with relative drag: click does not jump; drag changes value by
    Δx / track_width × range × gain.  Shift × 0.25.  Wheel: ±step (Shift: step/10)."""

    drag_started = pyqtSignal()
    drag_stopped = pyqtSignal()

    def __init__(self, lo: float, hi: float, step: float, parent: QWidget | None = None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._lo, self._hi, self._step = lo, hi, step
        self._dragging = False
        self._drag_start_x = 0.0
        self._drag_start_sv = 0
        self._wheel_tenths = 0

    def _track_width(self) -> float:
        opt = QStyleOptionSlider()
        opt.initFrom(self)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt,
            QStyle.SubControl.SC_SliderGroove, self)
        return max(1.0, groove.width())

    def _gain(self) -> float:
        return 0.5 if self._track_width() < SHORT_TRACK_PX else 1.0

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_x = e.position().x()
            self._drag_start_sv = self.value()
            self.drag_started.emit()
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._dragging and (e.buttons() & Qt.MouseButton.LeftButton):
            dx = e.position().x() - self._drag_start_x
            gain = self._gain()
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                gain *= 0.25
            track_w = self._track_width()
            range_ = self._hi - self._lo
            delta_slider = (dx / track_w) * (range_ / self._step) * gain
            new_sv = self._drag_start_sv + delta_slider
            new_sv = max(0.0, min(float(self.maximum()), new_sv))
            self.setValue(round(new_sv))
            e.accept()
        else:
            super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.drag_stopped.emit()
            e.accept()
        else:
            super().mouseReleaseEvent(e)

    def wheelEvent(self, e) -> None:  # noqa: N802
        steps = e.angleDelta().y() // 120
        if steps == 0:
            super().wheelEvent(e)
            return
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._wheel_tenths += steps
            if abs(self._wheel_tenths) >= 10:
                delta = self._wheel_tenths // 10
                self._wheel_tenths -= delta * 10
                self.setValue(max(self.minimum(), min(self.maximum(), self.value() + delta)))
        else:
            self.setValue(max(self.minimum(), min(self.maximum(), self.value() + steps)))
        e.accept()


class SliderRow(QWidget):
    """Name, slider, value; double-click on the value resets to the default.

    K40a: a small "?" button sits directly left of the value label (only when the slider
    name has a HELP entry); clicking it opens a popup with short (bold), what, tip (italic).
    """

    changed = pyqtSignal(str, float)

    def __init__(self, name: str, lo: float, hi: float, default: float, step: float,
                 label: str = "", label_width: int = 100, track=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.name, self.lo, self.step, self.default = name, lo, step, default
        self.decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        self.percent = False
        self.label_width = label_width
        self._label_text = label or name.replace("_", " ")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        _rh = scale.row_h(style.current_s())
        self.setFixedHeight(_rh)
        self.slider = _RelativeSlider(lo, hi, step)
        self.slider.setRange(0, round((hi - lo) / step))
        self.slider.setMinimumWidth(round(6 * _rh))
        if track:                                # superslim colour track instead of grey
            self.slider.setStyleSheet(style.track_sheet(track))
        self.value_label = QLabel()
        self.value_label.setFont(style.font("base"))
        self.value_label.setProperty("font_role", "base")
        self.value_label.setFixedWidth(scale.value_w(style.current_s()))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.mouseDoubleClickEvent = lambda _e: self.set_value(self.default)
        if label_width > 0:
            lbl = QLabel(self._label_text)
            lbl.setFont(style.font("base"))
            lbl.setProperty("font_role", "base")
            lbl.setProperty("muted", True)
            lbl.setFixedWidth(label_width)
            row.addWidget(lbl)
        row.addWidget(self.slider, 1)
        # K40a: "?" button directly left of the value label (the reset double-click spot)
        self.help_btn: QPushButton | None = None
        self.help_popup: QLabel | None = None
        entry = load_help().get(self.name)
        if entry:
            self.help_btn = small_button("?", tip=entry.get("short", ""))
            self.help_btn.setObjectName(f"help_{self.name}")
            _hs = scale.row_h(style.current_s())
            self.help_btn.setFixedSize(_hs, _hs)   # square, like the wheel ↺ and general "?" buttons
            self.help_btn.clicked.connect(self._show_help_popup)
            row.addWidget(self.help_btn)
        row.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_change)
        self.set_value(default)

    def _show_help_popup(self, *_args) -> None:
        """K40a: popup at the "?" button: short (bold), what, tip (italic); closes on click away."""
        entry = load_help().get(self.name)
        if not entry or self.help_btn is None:
            return
        if self.help_popup is None:
            self.help_popup = QLabel(self)
            self.help_popup.setWindowFlags(Qt.WindowType.Popup)
            self.help_popup.setTextFormat(Qt.TextFormat.RichText)
            self.help_popup.setWordWrap(True)
            self.help_popup.setFont(style.font("small"))
            self.help_popup.setStyleSheet(
                f"background: {style.BLACK}; color: {style.TEXT}; border: 1px solid {style.LINE};")
        rich = (f"<b>{_html.escape(entry.get('short', ''))}</b><br>"
                f"{_html.escape(entry.get('what', ''))}<br>"
                f"<i>{_html.escape(entry.get('tip', ''))}</i>")
        self.help_popup.setText(rich)
        self.help_popup.setFixedWidth(round(2.2 * style.current_s()))
        self.help_popup.adjustSize()
        self.help_popup.move(self.help_btn.mapToGlobal(self.help_btn.rect().bottomLeft()))
        self.help_popup.show()

    def label_text(self) -> str:
        """The label text shown to the left of the slider."""
        return self._label_text

    def set_label_width(self, w: int) -> None:
        """Set the label's fixed width (call after all rows in a section are created)."""
        if w <= 0:
            return
        lay = self.layout()
        for i in range(lay.count()):
            item = lay.itemAt(i)
            widget = item.widget() if item else None
            if isinstance(widget, QLabel) and widget is not self.value_label:
                widget.setFixedWidth(w)
                break

    def set_value_width(self, w: int) -> None:
        """Update the value label's fixed width (call when the scale changes)."""
        if w > 0:
            self.value_label.setFixedWidth(w)
            if self.help_btn is not None:      # K40a: keep the "?" button square at row height
                _hs = scale.row_h(style.current_s())
                self.help_btn.setFixedSize(_hs, _hs)

    def value(self) -> float:
        return self.lo + self.slider.value() * self.step

    def set_value(self, v: float) -> None:
        if not math.isfinite(v):                 # never let NaN/inf from the worker crash the UI
            return
        self.slider.setValue(round((v - self.lo) / self.step))
        self._on_change()

    def _on_change(self, *_args) -> None:
        if self.percent or self.name.startswith("amount_"):
            self.value_label.setText(f"{round(self.value() * 100)} %")
        else:
            self.value_label.setText(f"{self.value():.{self.decimals}f}".replace(".", ","))
        self.changed.emit(self.name, self.value())


def small_button(text: str, role: str = "", checkable: bool = False, tip: str = "") -> QPushButton:
    """Compact header button (ON / SOLO / ↺)."""
    b = QPushButton(text)
    b.setProperty("small", True)
    if role:
        b.setProperty("role", role)
    b.setCheckable(checkable)
    b.setFont(style.font("small"))
    b.setProperty("font_role", "small")
    b.setFixedHeight(scale.row_h(style.current_s()))
    b.setToolTip(tip)
    return b


class CollapseTriangle(QWidget):
    """Filled triangle painted via QPainter: ▾ open (TEXT), ▸ closed (MUTED)."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._collapsed = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        _rh = scale.row_h(style.current_s())
        self.setFixedSize(_rh, _rh)

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsed != collapsed:
            self._collapsed = collapsed
            self.update()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            e.accept()
        else:
            super().mousePressEvent(e)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = self.width()
        edge = side * 0.45
        cx, cy = side / 2, side / 2
        if self._collapsed:
            # ▸ pointing right, MUTED
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.MUTED))
            path = QPainterPath()
            path.moveTo(cx - edge / 2, cy - edge / 2)
            path.lineTo(cx + edge / 2, cy)
            path.lineTo(cx - edge / 2, cy + edge / 2)
            path.closeSubpath()
        else:
            # ▾ pointing down, TEXT
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.TEXT))
            path = QPainterPath()
            path.moveTo(cx - edge / 2, cy - edge / 2)
            path.lineTo(cx + edge / 2, cy - edge / 2)
            path.lineTo(cx, cy + edge / 2)
            path.closeSubpath()
        p.drawPath(path)


class SectionHeader(QWidget):
    """Section title with collapse triangle; right-aligned AUTO · ↺ (reset) · ON/OFF · ? (help).

    The weight is controlled by an Amount SliderRow in the content (set via
    `set_amount_slider`).  When collapsed and weight < 100 % a muted label
    (e.g. "60 %") appears next to the title.
    """

    weighted = pyqtSignal(str, float)
    switched = pyqtSignal(str, bool)
    reset = pyqtSignal(str)
    collapsed_changed = pyqtSignal(str, bool)  # key, collapsed
    help_requested = pyqtSignal(str)           # key
    auto_requested = pyqtSignal(str)           # key
    drag_started = pyqtSignal(str)             # key
    drag_moved = pyqtSignal(str, object)       # key, QPoint (global)
    drag_ended = pyqtSignal(str, bool)         # key, was_drag
    reset_order_requested = pyqtSignal()

    def __init__(self, key: str, title: str, on: bool = True, parent: QWidget | None = None,
                 auto: bool = False):
        super().__init__(parent)
        self.key = key
        self._collapsed = False
        self.amount_slider: SliderRow | None = None
        _rh = scale.row_h(style.current_s())
        self.setFixedHeight(_rh)
        # full column width: the buttons sit right-aligned and line up across all sections
        # (user 2026-10-11: "rechtsbündig, einfacher zu finden")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._drag_start_pos = None
        self._is_dragging = False
        self.triangle = CollapseTriangle()
        self.triangle.clicked.connect(self._title_click)
        self._title = group_title(title)
        self._title.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title.installEventFilter(self)
        self._title.setToolTip(
            "Drag to change the processing order. Sections are applied top to bottom, "
            "each one works on the result of the one above.")
        # K25i: the title keeps its natural width (minimum = sizeHint) so it is never
        # truncated at small resolutions; the header buttons absorb the deficit instead.
        self._title.setMinimumWidth(self._title.sizeHint().width())
        row.addWidget(self.triangle)
        row.addWidget(self._title)
        self._collapsed_label = QLabel("")
        self._collapsed_label.setFont(style.font("small"))
        self._collapsed_label.setProperty("font_role", "small")
        self._collapsed_label.setProperty("muted", True)
        row.addWidget(self._collapsed_label)
        row.addStretch(1)
        # K25i: header buttons are compressible (max = natural width, min = rh-1) so that in a
        # narrow chapter (e.g. WHEELS in the wheel strip) they absorb the deficit instead of
        # the title, which keeps its minimum width. The minimum stays at rh-1 (test k25i).
        _bmin = max(1, _rh - 1)
        # AUTO always exists so the columns line up; sections without AUTO keep its space empty
        self.auto_btn = small_button("AUTO", "auto")
        self.auto_btn.setObjectName(f"auto_{key}")
        self.auto_btn.setFixedHeight(_rh)
        self.auto_btn.setMinimumWidth(_bmin)
        self.auto_btn.setMaximumWidth(self._auto_w())
        self.auto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_btn.clicked.connect(lambda: self.auto_requested.emit(self.key))
        keep = self.auto_btn.sizePolicy()
        keep.setRetainSizeWhenHidden(True)
        self.auto_btn.setSizePolicy(keep)
        self.auto_btn.setVisible(auto)
        self.auto_btn.setToolTip(f"AUTO: set {title} from the picture as it arrives here in the chain.\nA picture that is already in range is left alone; ↺ undoes it.")
        self.has_auto = auto
        self.help_btn = QPushButton("?")
        self.help_btn.setObjectName(f"help_{key}")
        self.help_btn.setProperty("help", True)
        self.help_btn.setFont(style.font("small"))
        self.help_btn.setProperty("font_role", "small")
        self.help_btn.setFixedHeight(_rh)
        self.help_btn.setMinimumWidth(_bmin)
        self.help_btn.setMaximumWidth(_rh)
        self.help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_btn.clicked.connect(lambda: self.help_requested.emit(self.key))
        self.help_btn.setToolTip(f"Show help for the {title} section")
        self.on = small_button("ON" if on else "OFF", "on", True)
        self.on.setChecked(on)
        self._show_on(on)
        self.on.setMinimumWidth(_bmin)
        self.on.setMaximumWidth(scale.value_w(style.current_s()))
        self.on.setToolTip(f"Turn {title} ON or OFF")
        self.undo = small_button("↺")
        self.undo.setMinimumWidth(_bmin)
        self.undo.setMaximumWidth(_rh)
        self.undo.setToolTip(f"Reset all {title} settings to defaults")
        # order, most prominent on the right (user 2026-10-11): AUTO · ↺ · ON · ?
        for b in (self.auto_btn, self.undo, self.on, self.help_btn):
            row.addWidget(b)
        self.on.toggled.connect(self._on_switch)
        self.undo.clicked.connect(lambda: self.reset.emit(self.key))

    def _auto_w(self) -> int:
        """Wide enough for "AUTO" in the button font (value_w clipped it to "UTC", 2026-10-11)."""
        fm = self.auto_btn.fontMetrics()
        return max(scale.value_w(style.current_s()),
                   fm.horizontalAdvance("AUTO") + 2 * max(4, scale.row_h(style.current_s()) // 3))

    def value(self) -> float:
        if self.amount_slider is not None:
            return self.amount_slider.value()
        return 1.0

    def is_on(self) -> bool:
        return self.on.isChecked()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def update_scale(self) -> None:
        """K25i: update button sizes for the current scale."""
        _rh = scale.row_h(style.current_s())
        _bmin = max(1, _rh - 1)
        self._title.setMinimumWidth(self._title.sizeHint().width())
        self.triangle.setFixedSize(_rh, _rh)
        self.auto_btn.setFixedHeight(_rh)
        self.auto_btn.setMinimumWidth(_bmin)
        self.auto_btn.setMaximumWidth(self._auto_w())
        self.help_btn.setFixedHeight(_rh)
        self.help_btn.setMinimumWidth(_bmin)
        self.help_btn.setMaximumWidth(_rh)
        self.on.setFixedHeight(_rh)
        self.on.setMinimumWidth(_bmin)
        self.on.setMaximumWidth(scale.value_w(style.current_s()))
        self.undo.setFixedHeight(_rh)
        self.undo.setMinimumWidth(_bmin)
        self.undo.setMaximumWidth(_rh)

    def set_collapsed(self, collapsed: bool, emit: bool = True) -> None:
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self.triangle.set_collapsed(collapsed)
        self._update_collapsed_label()
        if emit:
            self.collapsed_changed.emit(self.key, collapsed)

    def toggle(self) -> None:
        """Toggle the ON state (and expand/collapse via the switched signal)."""
        self.on.setChecked(not self.on.isChecked())

    def _title_click(self, _e=None) -> None:
        """Title/triangle click: only collapse/expand, do not change the ON state."""
        self.set_collapsed(not self._collapsed)

    def eventFilter(self, obj, event):
        """Drag & drop on the title; right-click context menu."""
        if obj is self._title:
            et = event.type()
            if et == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._drag_start_pos = event.globalPosition().toPoint()
                    self._is_dragging = False
                    return True
                elif event.button() == Qt.MouseButton.RightButton:
                    self._show_context_menu(event.globalPosition().toPoint())
                    return True
            elif et == QEvent.Type.MouseMove:
                if self._drag_start_pos is not None:
                    pos = event.globalPosition().toPoint()
                    if not self._is_dragging:
                        if (pos - self._drag_start_pos).manhattanLength() >= 6:
                            self._is_dragging = True
                            self.drag_started.emit(self.key)
                    if self._is_dragging:
                        self.drag_moved.emit(self.key, pos)
                    return True
            elif et == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    if self._is_dragging:
                        self.drag_ended.emit(self.key, True)
                    else:
                        self._title_click()
                    self._drag_start_pos = None
                    self._is_dragging = False
                    return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, pos) -> None:
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        reset_action = menu.addAction("Reset chain order")
        action = menu.exec(pos)
        if action == reset_action:
            self.reset_order_requested.emit()

    def set_amount_slider(self, slider: SliderRow) -> None:
        """Link the Amount SliderRow; its `changed` signal drives `weighted`."""
        self.amount_slider = slider
        slider.changed.connect(self._on_amount_changed)
        self._update_collapsed_label()
        self._show_on(self.is_on())

    def _on_amount_changed(self, _name: str, value: float) -> None:
        self._update_collapsed_label()
        self.weighted.emit(self.key, value)

    def _update_collapsed_label(self) -> None:
        """Show 'NN %' next to the title only when collapsed and weight < 100 %."""
        if self._collapsed and self.amount_slider is not None:
            pct = round(self.amount_slider.value() * 100)
            self._collapsed_label.setText(f"{pct} %" if pct < 100 else "")
        else:
            self._collapsed_label.setText("")

    def _show_on(self, on: bool) -> None:
        """Title white when ON, dark when OFF — also while collapsed (user 2026-10-08)."""
        self._title.setProperty("state", "on" if on else "off")
        self._title.style().unpolish(self._title)
        self._title.style().polish(self._title)
        if self.amount_slider is not None:
            self.amount_slider.setEnabled(on)
        self._collapsed_label.setEnabled(on)

    def _on_switch(self, on: bool) -> None:
        self.on.setText("ON" if on else "OFF")
        self._show_on(on)
        self.switched.emit(self.key, on)


def group_title(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setFont(style.font("title", QFont.Weight.DemiBold))
    label.setProperty("font_role", "title")
    label.setProperty("font_weight", QFont.Weight.DemiBold)
    label.setProperty("section", True)
    return label


# ── Colour wheels ─────────────────────────────────────────────────────────────

class ColorWheel(QWidget):
    """Hue disk with a draggable point (tint) — the point maps to the core sliders via
    `wheels.to_values`. Left drag moves the point, double-click resets it."""

    changed = pyqtSignal(str, float, float)       # kind, u, v
    reset_requested = pyqtSignal(str)             # kind — double-click or ↺ button
    drag_started = pyqtSignal()                   # K31b-F
    drag_stopped = pyqtSignal()                   # K31b-F

    _LABELS = {"gain": "Highlights", "gamma": "Midtones", "lift": "Shadows"}
    _TOOLTIPS = {
        "gain": "Highlights (Gain): tint and brightness of the bright tones.",
        "gamma": "Mids (Gamma): tint and brightness of the midtones.",
        "lift": "Shadows (Lift): tint and brightness of the dark tones.",
    }

    def __init__(self, kind: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.kind = kind
        self.u = self.v = 0.0
        self._drag = False
        self._label = self._LABELS.get(kind, "")
        self.setToolTip(self._TOOLTIPS.get(kind, ""))
        _rh = scale.row_h(style.current_s())
        self.setMinimumSize(round(4 * _rh), round(4 * _rh))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_point(self, u: float, v: float) -> None:
        self.u, self.v = u, v
        self.update()

    def _label_h(self) -> int:
        """Height of the label text (0 if no label)."""
        if not self._label:
            return 0
        fm = QFontMetrics(style.font("small"))
        return fm.height()

    def _geometry(self) -> tuple[QPointF, float]:
        # the name is no longer drawn here but by the dialog above the master slider
        # (user 2026-10-11), so the disc is simply centred
        side = min(self.width(), self.height())
        r = side / 2 - 10   # room for rim dot
        return QPointF(self.width() / 2, self.height() / 2), max(r, 4)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._geometry()
        hue = QConicalGradient(c, 0)
        for i in range(13):
            hue.setColorAt(i / 12, QColor(*wheels.hue_rgb(math.radians(i * 30))))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hue)
        p.drawEllipse(c, r, r)
        fade = QRadialGradient(c, r)                 # dark centre: muted, black-first look
        fade.setColorAt(0.0, style.WHEEL_FADE_CENTER)
        fade.setColorAt(1.0, style.WHEEL_FADE_EDGE)
        p.setBrush(fade)
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(style.LINE_STRONG), 1))
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        if not self.isEnabled():                     # section off: 80 % black over the disk
            p.setBrush(style.WHEEL_DISABLED)
            p.drawEllipse(c, r + 1, r + 1)
        dot = QPointF(c.x() + self.u * r, c.y() - self.v * r)
        strength = math.hypot(self.u, self.v)
        if strength > 0.02:                          # the tint as a visible vector (user, LR):
            a = math.atan2(self.v, self.u)           # faint line centre → puck + colour dot
            tint = QColor(*wheels.hue_rgb(a))        # on the rim in the colour it produces
            tint.setHsv(tint.hsvHue(), 255, 255, 255 if self.isEnabled() else 70)
            line = QColor(tint)
            line.setAlpha(110 if self.isEnabled() else 40)
            p.setPen(QPen(line, 1))
            p.drawLine(c, dot)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tint)
            p.drawEllipse(QPointF(c.x() + math.cos(a) * (r + 6), c.y() - math.sin(a) * (r + 6)),
                          3.5, 3.5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(style.ACCENT_HI if self._drag or self.u or self.v else style.TEXT), 2))
        p.drawEllipse(dot, 5, 5)

    def _set_from(self, pos: QPointF) -> None:
        c, r = self._geometry()
        u, v = (pos.x() - c.x()) / r, (c.y() - pos.y()) / r
        n = math.hypot(u, v)
        if n > 1.0:
            u, v = u / n, v / n
        self.set_point(u, v)
        self.changed.emit(self.kind, u, v)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self.drag_started.emit()
            self._set_from(e.position())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag:
            self._set_from(e.position())

    def mouseReleaseEvent(self, _e) -> None:  # noqa: N802
        if self._drag:
            self._drag = False
            self.drag_stopped.emit()
        self.update()

    def mouseDoubleClickEvent(self, _e) -> None:  # noqa: N802
        self.reset_requested.emit(self.kind)

    # mouse wheel: strength along the current direction, gently (user 2026-10-11)
    WHEEL_STEP, WHEEL_STEP_FINE = 0.02, 0.005

    def wheelEvent(self, e) -> None:  # noqa: N802
        notches = e.angleDelta().y() / 120.0
        r = math.hypot(self.u, self.v)
        if notches == 0 or r < 1e-6:          # no direction yet: nothing to scale along
            e.ignore()
            return
        fine = e.modifiers() & Qt.KeyboardModifier.ShiftModifier
        step = (self.WHEEL_STEP_FINE if fine else self.WHEEL_STEP) * notches
        r_new = min(1.0, max(0.0, r + step))
        self.set_point(self.u * r_new / r, self.v * r_new / r)
        self.changed.emit(self.kind, self.u, self.v)
        e.accept()


# ── Histogram ─────────────────────────────────────────────────────────────────

class Histogram(QWidget):
    """RGB histogram: additive R/G/B composition, single channel, clipping, grid."""

    CHANNELS = ("r", "g", "b", "l")
    _CHANNEL_COLORS = style.HIST_CHANNEL_COLORS
    _CLIP_THRESHOLD = 0.001  # 0.1 % of total
    _CLIP_SHOW = 0.005  # 0.5 % – minimum to display text

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.visible_channels: set[str] = {"r", "g", "b"}
        self.show_luma = True            # L switch: luminance (waveform + ghost) behind the curves
        self._wave_img: QImage | None = None
        self._rgb_data: dict[str, dict[str, list[float]]] = {}
        self._clip: dict[str, float] = {}
        self.setMinimumHeight(round(3 * scale.row_h(style.current_s())))

    def set_rgb_data(self, data: dict[str, dict[str, list[float]]]) -> None:
        """data = {"source": {"r":[...], "g":[...], "b":[...], "l":[...]}, "result": {...}}"""
        self._rgb_data = data
        self.visible_channels = {ch for ch in ("r", "g", "b") if data.get("result", {}).get(ch)}
        self.update()

    def set_data(self, data: dict[str, list[float]]) -> None:
        """Backward compat: treat as L* histogram under 'l' key."""
        self._rgb_data = {
            "source": {"l": data.get("source", [])},
            "result": {"l": data.get("result", [])},
        }
        self.visible_channels = {"l"}
        self.update()

    def set_clip_data(self, clip: dict[str, float] | None) -> None:
        """Set per-channel clipping fractions for display."""
        self._clip = clip or {}
        self.update()

    WAVE_OPACITY = 0.35               # luma waveform behind the histogram (user 2026-10-11: "B")
    LINE_GHOST = True                 # also the luminance ghost line

    def set_wave(self, path: str | None) -> None:
        """Luma waveform image from the worker, drawn behind the curves while L is on."""
        img = QImage(path or "")
        self._wave_img = None if img.isNull() else img
        self.update()

    def _peak(self) -> float:
        """Height scale without the three end bins of each side: a clipping spike at pure black or
        white used to set the scale and flatten every curve (2026-10-11); it is cut at the top."""
        peak = 0.0
        for series in self._rgb_data.values():
            for ch in self.visible_channels:
                vals = series.get(ch)
                if vals:
                    inner = vals[3:-3] if len(vals) > 12 else vals
                    peak = max(peak, max(inner))
        return peak * 1.05 or 1.0

    # Neon look like the parade (user 2026-10-11): curves unfilled with a small glow; the white
    # area of the additive fills is replaced by a soft white "ghost" glow of the luminance
    GHOST = 1.0                      # strength of the white ghost glow (0 = off)
    _GLOW_STROKES = ((7.0, 0.10), (4.0, 0.18), (2.2, 0.35))     # (width px, alpha) of a curve
    _GHOST_STROKES = ((22.0, 0.035), (14.0, 0.05), (8.0, 0.07), (4.0, 0.09))

    @staticmethod
    def _curve(vals: list[float], x0: float, y0: float, w: float, h: float,
               peak: float) -> QPainterPath:
        n = len(vals)
        path = QPainterPath()
        for i, v in enumerate(vals):
            pt = QPointF(x0 + i * w / (n - 1), y0 + h - v / peak * (h - 2))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        return path

    def _draw_histogram(self, p: QPainter, x0: int, y0: int, w: int, h: int,
                        series: dict[str, list[float]], peak: float,
                        source: dict[str, list[float]] | None) -> None:
        """Ghost glow (white, luminance) behind neon RGB curves; source as dashed outline."""
        plus = QPainter.CompositionMode.CompositionMode_Plus
        round_pen = (Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        # 1) ghost: luminance (or the mean of the visible channels) as a soft white haze
        ghost_vals = series.get("l")
        rgb_vis = [series[c] for c in ("r", "g", "b") if c in self.visible_channels and series.get(c)]
        if not ghost_vals and rgb_vis:
            ghost_vals = [sum(t) / len(rgb_vis) for t in zip(*rgb_vis)]
        if (self.show_luma and self.LINE_GHOST and self.GHOST > 0 and ghost_vals and len(ghost_vals) > 1
                and self.visible_channels != {"l"}):
            gpath = self._curve(ghost_vals, x0, y0, w, h, peak)
            p.setCompositionMode(plus)
            for width, alpha in self._GHOST_STROKES:
                c = QColor(style.HIST_GHOST)
                c.setAlphaF(min(1.0, alpha * self.GHOST))
                p.setPen(QPen(c, width, *round_pen))
                p.drawPath(gpath)
            # faint haze under the luminance curve, fading to the baseline
            haze = QPainterPath(gpath)
            haze.lineTo(QPointF(x0 + w, y0 + h))
            haze.lineTo(QPointF(x0, y0 + h))
            haze.closeSubpath()
            grad = QLinearGradient(0, y0, 0, y0 + h)
            top, bottom = QColor(style.HIST_GHOST), QColor(style.HIST_GHOST)
            top.setAlphaF(min(1.0, 0.10 * self.GHOST))
            bottom.setAlphaF(0.0)
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bottom)
            p.fillPath(haze, QBrush(grad))
        # 2) curves: unfilled, crisp line on a small neon glow of the channel colour
        for ch in ("r", "g", "b", "l"):
            if ch not in self.visible_channels:
                continue
            vals = series.get(ch)
            if not vals or len(vals) < 2:
                continue
            colour = self._CHANNEL_COLORS[ch]
            path = self._curve(vals, x0, y0, w, h, peak)
            p.setCompositionMode(plus)
            for width, alpha in self._GLOW_STROKES:
                c = QColor(colour)
                c.setAlphaF(alpha)
                p.setPen(QPen(c, width, *round_pen))
                p.drawPath(path)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(colour.lighter(130), 1.2, *round_pen))
            p.drawPath(path)
        # Source as grey outline
        if source:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            for ch in ("r", "g", "b", "l"):
                if ch not in self.visible_channels:
                    continue
                vals = source.get(ch)
                if not vals or len(vals) < 2:
                    continue
                n = len(vals)
                p.setPen(QPen(style.HIST_SOURCE, 1, Qt.PenStyle.DashLine))
                line_path = QPainterPath()
                for i, v in enumerate(vals):
                    px = x0 + i * w / (n - 1)
                    py = y0 + h - v / peak * (h - 2)
                    if i == 0:
                        line_path.moveTo(QPointF(px, py))
                    else:
                        line_path.lineTo(QPointF(px, py))
                p.drawPath(line_path)

    def _draw_clipping(self, p: QPainter, x0: int, y0: int, w: int, h: int,
                       series: dict[str, list[float]]) -> None:
        """Draw clipping triangles at left (black) and right (white) edges."""
        for ch in ("r", "g", "b", "l"):
            if ch not in self.visible_channels:
                continue
            vals = series.get(ch)
            if not vals:
                continue
            if vals[0] > self._CLIP_THRESHOLD:
                tri = QPainterPath()
                tri.moveTo(QPointF(x0, y0))
                tri.lineTo(QPointF(x0, y0 + 10))
                tri.lineTo(QPointF(x0 + 10, y0))
                tri.closeSubpath()
                p.fillPath(tri, style.HIST_CLIP)
            if vals[-1] > self._CLIP_THRESHOLD:
                tri = QPainterPath()
                tri.moveTo(QPointF(x0 + w, y0))
                tri.lineTo(QPointF(x0 + w, y0 + 10))
                tri.lineTo(QPointF(x0 + w - 10, y0))
                tri.closeSubpath()
                p.fillPath(tri, style.HIST_CLIP)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # Background
        p.fillRect(self.rect(), style.HIST_BG)
        if self.show_luma and self._wave_img is not None and self.WAVE_OPACITY > 0:
            p.save()
            p.setOpacity(self.WAVE_OPACITY)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(QRectF(0, 0, w, h), self._wave_img)
            p.restore()
        # Grid lines at 25/50/75%
        p.setPen(QPen(style.HIST_GRID, 1))
        for frac in (0.25, 0.5, 0.75):
            x = int(w * frac)
            p.drawLine(x, 0, x, h)
        peak = self._peak()
        src = self._rgb_data.get("source", {})
        res = self._rgb_data.get("result", {})
        self._draw_histogram(p, 0, 0, w, h, res, peak, src if src else None)
        self._draw_clipping(p, 0, 0, w, h, res)
        self._draw_clip_text(p, w)

    def _draw_clip_text(self, p: QPainter, w: int) -> None:
        """Draw per-channel clipping warnings in the top-right corner."""
        if not self._clip:
            return
        p.setFont(style.font("small"))
        y = 2
        for ch in ("r", "g", "b"):
            hi = self._clip.get(f"{ch}_hi", 0.0)
            lo = self._clip.get(f"{ch}_lo", 0.0)
            col = self._CHANNEL_COLORS[ch]
            col.setAlpha(160)
            p.setPen(QPen(col))
            if hi >= self._CLIP_SHOW:
                txt = f"{ch.upper()} ▲ {hi * 100:.0f} %"
                p.drawText(w - p.fontMetrics().horizontalAdvance(txt) - 4, y + 10, txt)
                y += 12
            if lo >= self._CLIP_SHOW:
                txt = f"{ch.upper()} ▼ {lo * 100:.0f} %"
                p.drawText(w - p.fontMetrics().horizontalAdvance(txt) - 4, y + 10, txt)
                y += 12


# ── Scope channel buttons ─────────────────────────────────────────────────────

class ScopeButton(QWidget):
    """20×20 scope button: channel fill/border, RGB stripes, SPLIT symbol, ? text."""

    clicked = pyqtSignal(str)

    CH_COLORS = style.SCOPE_CH_COLORS

    def __init__(self, channel: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.channel = channel
        self._active = False
        _rh = scale.row_h(style.current_s())
        self.setFixedSize(_rh, _rh)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, on: bool) -> None:
        if self._active != on:
            self._active = on
            self.update()

    def is_active(self) -> bool:
        return self._active

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.channel)
            e.accept()
        else:
            super().mousePressEvent(e)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(style.BLACK))
        if self.channel == "rgb":
            self._paint_rgb(p, w, h)
        elif self.channel == "split":
            self._paint_split(p, w, h)
        elif self.channel == "help":
            self._paint_help(p, w, h)
        else:
            self._paint_channel(p, w, h, self.channel)

    def _paint_channel(self, p: QPainter, w: int, h: int, ch: str) -> None:
        color = self.CH_COLORS[ch]
        if ch == "l" and not self._active:
            color = style.SCOPE_L_OFF    # L off: dark grey, on: white (user 2026-10-11)
        if self._active:
            p.fillRect(self.rect(), color)
            p.setPen(style.SCOPE_ACTIVE_TEXT)
        else:
            p.setPen(QPen(color, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(0, 0, w - 1, h - 1)
            p.setPen(color)
        p.setFont(style.font("small"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, ch.upper())

    def _paint_rgb(self, p: QPainter, w: int, h: int) -> None:
        # Boxed like R/G/B, letters in their channel colours (2026-10-11: the old stripes ran
        # through the text and "RGB" did not fit the square). Active: the three colours filled.
        colors = [self.CH_COLORS[c] for c in ("r", "g", "b")]
        if self._active:
            sw = w // 3
            for i, color in enumerate(colors):
                x0 = i * sw
                x1 = x0 + sw if i < 2 else w
                p.fillRect(x0, 0, x1 - x0, h, color)
        else:
            p.setPen(QPen(QColor(style.TEXT), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(0, 0, w - 1, h - 1)
        font = QFont(style.font("small"))
        fm = QFontMetrics(font)
        while fm.horizontalAdvance("RGB") > w - 4 and font.pixelSize() > 6:
            font.setPixelSize(font.pixelSize() - 1)
            fm = QFontMetrics(font)
        p.setFont(font)
        x = (w - fm.horizontalAdvance("RGB")) / 2
        y = (h + fm.ascent() - fm.descent()) / 2
        for letter, color in zip("RGB", colors):
            p.setPen(style.SCOPE_ACTIVE_TEXT if self._active else color)
            p.drawText(QPointF(x, y), letter)
            x += fm.horizontalAdvance(letter)

    def _paint_split(self, p: QPainter, w: int, h: int) -> None:
        accent = QColor(style.ACCENT)
        if self._active:
            p.fillRect(self.rect(), accent)
            pen_col = style.SCOPE_ACTIVE_TEXT
        else:
            p.setPen(QPen(accent, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(0, 0, w - 1, h - 1)
            pen_col = accent
        # Symbol: two stacked half-boxes
        p.setPen(QPen(pen_col, 1))
        p.drawLine(5, 5, 15, 5)
        p.drawLine(5, 5, 5, 10)
        p.drawLine(15, 5, 15, 10)
        p.drawLine(5, 10, 5, 15)
        p.drawLine(15, 10, 15, 15)
        p.drawLine(5, 15, 15, 15)

    def _paint_help(self, p: QPainter, w: int, h: int) -> None:
        accent = QColor(style.ACCENT)
        if self._active:
            p.fillRect(self.rect(), accent)
            p.setPen(style.SCOPE_ACTIVE_TEXT)
        else:
            p.setPen(QPen(accent, 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(0, 0, w - 1, h - 1)
            p.setPen(accent)
        p.setFont(style.font("small"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "?")


# ── Image view ────────────────────────────────────────────────────────────────

class ImageView(QWidget):
    """Whole image fitted at start; wheel zooms about the cursor, middle button pans,
    F/0 = fit, 1 = 100 %, S = split view.

    Everything is in FULL-IMAGE coordinates (`logical` = size of the original layer): zoom is
    screen pixels per original pixel, so 100 % is pixel-perfect, and swapping the shown image
    (preview, original, reference preview — all at different resolutions) never moves the view.
    A full-resolution crop can be laid over the area it was computed for (`set_overlay`).

    Split view: left = original, right = grading, separated by a draggable vertical line."""

    ZOOM_STEP, ZOOM_MAX = 1.25, 16.0
    SPLIT_GRAB_ROWS = 1.5  # half-width of the split grab area in row heights (≈ 36 px at 2560; user 2026-10-12: 6 px was too fiddly)
    split_button_clicked = pyqtSignal()
    clip_button_clicked = pyqtSignal()
    # status text + zoom for a label outside the view (user 2026-10-11: no strip under the image)
    status_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.image = QImage()
        self.original_image = QImage()      # for split view left side
        self.logical = QSizeF()
        self.overlay = QImage()
        self.overlay_rect = QRectF()
        self.show_overlay = True
        self.zoom = 1.0
        self.offset = QPointF(0, 0)
        self.mark = ""
        self.mark_color: QColor | None = None
        self._pan_from: QPointF | None = None
        self._fitted = True
        # Split view state
        self.split_mode = False
        self.split_pos = 0.5               # 0..1 fraction of widget width
        self._split_dragging = False
        self._split_btn_rect = QRectF()    # hit area for the split toggle
        self._clip_btn_rect = QRectF()     # hit area for the clip toggle
        self._clip_view = False
        self._clip_text = ""
        # K25i: status strip at the bottom (outside the drawn image)
        self._status_h = 0
        self._status_tooltip = ""
        # K27e: loading bar state
        self._loading = False
        self._loading_progress = 0.0
        # K27f: fade-in state
        self._fade_prev: QImage = QImage()
        self._fade_alpha: float = 0.0
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(10)
        self._fade_timer.timeout.connect(self._fade_tick)
        self._fade_elapsed = QElapsedTimer()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # hover cursor over the split line
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip("Split view: graded left, original right (drag the line).")

    def set_status_h(self, h: int) -> None:
        """Height of the status strip at the bottom (0 = no strip)."""
        if h == self._status_h:
            return
        self._status_h = h
        if self._fitted:
            self.fit()
        else:
            self._clamp()

    def _emit_status(self) -> None:
        try:
            self.status_changed.emit(self._last_status)
        except RuntimeError:          # view already deleted when the timer fires
            pass

    def set_status(self, text: str, tooltip: str = "") -> None:
        """Set the status text shown in the bottom strip; `tooltip` for the full text."""
        self.mark = text
        self._status_tooltip = tooltip or text
        self.setToolTip(self._status_tooltip)
        self.update()

    def set_clip_text(self, text: str) -> None:
        """Set the clip percentage text shown left of the C button."""
        if text == self._clip_text:
            return
        self._clip_text = text
        self.update()

    def set_loading(self, active: bool, progress: float = 0.0) -> None:
        """Show/hide the loading bar; `progress` is 0..0.95."""
        if active == self._loading and not active:
            return
        self._loading = active
        self._loading_progress = progress
        self.update()

    def set_image(self, image: QImage, logical: QSizeF | None = None) -> None:
        """New document: show `image` as an image of `logical` size (default: its own) and fit."""
        self.logical = QSizeF(logical) if logical else QSizeF(image.size())
        self.image = image
        self.overlay = QImage()
        self.mark_color = None
        self.fit()

    def set_original(self, image: QImage) -> None:
        """Set the original image for split view left side."""
        self.original_image = image
        self.update()

    def replace_image(self, image: QImage, keep_overlay: bool = False) -> None:
        """Same document, other rendering (preview / original / reference preview): view stays."""
        if not self.image.isNull():
            self._fade_prev = self.image
            self._fade_alpha = 0.0
            self._fade_elapsed.start()
            self._fade_timer.start()
        else:
            self._fade_prev = QImage()
        self.image = image
        if not keep_overlay:
            self.overlay = QImage()
        self.update()

    def _fade_tick(self) -> None:
        """Advance the fade-in; alpha is the real elapsed time over the 120 ms span."""
        self._fade_alpha = min(1.0, self._fade_elapsed.elapsed() / 120.0)
        if self._fade_alpha >= 1.0:
            self._fade_timer.stop()
            self._fade_prev = QImage()
        self.update()

    def set_overlay(self, image: QImage, rect: QRectF) -> None:
        """Sharper pixels for `rect` (logical coordinates); cleared by the next new rendering."""
        self.overlay, self.overlay_rect = image, QRectF(rect)
        self.update()

    def fit_zoom(self) -> float:
        if self.logical.isEmpty() or self.width() < 2 or self._img_h() < 2:
            return 1.0
        return min(self.width() / self.logical.width(), self._img_h() / self.logical.height())

    def fit(self) -> None:
        self._fitted = True
        self.zoom = self.fit_zoom()
        self._centre()
        self.update()

    def _centre(self) -> None:
        self.offset = QPointF((self.logical.width() - self.width() / self.zoom) / 2,
                              (self.logical.height() - self._img_h() / self.zoom) / 2)

    def _clamp(self) -> None:
        """Keep the image inside the view: centred when smaller, edge-bound when larger."""
        vw, vh = self.width() / self.zoom, self._img_h() / self.zoom
        iw, ih = self.logical.width(), self.logical.height()
        x = (iw - vw) / 2 if vw >= iw else min(max(self.offset.x(), 0), iw - vw)
        y = (ih - vh) / 2 if vh >= ih else min(max(self.offset.y(), 0), ih - vh)
        self.offset = QPointF(x, y)

    def zoom_at(self, factor: float, pos: QPointF) -> None:
        """Zoom by `factor` keeping the image point under `pos` fixed."""
        new = min(max(self.zoom * factor, min(self.fit_zoom(), 1.0)), self.ZOOM_MAX)
        anchor = self.offset + pos / self.zoom
        self.zoom = new
        self.offset = anchor - pos / self.zoom
        self._fitted = False
        self._clamp()
        self.update()

    def _img_h(self) -> int:
        """Height of the image drawing area (excludes status strip)."""
        return self.height() - self._status_h

    def _image_rect(self) -> QRectF:
        """The image drawing area (full widget minus the status strip)."""
        return QRectF(0, 0, self.width(), self._img_h())

    def _status_rect(self) -> QRectF:
        """The status strip rect at the bottom (0 height if no strip)."""
        if self._status_h <= 0:
            return QRectF()
        return QRectF(0, self._img_h(), self.width(), self._status_h)

    def visible_rect(self) -> QRectF:
        """Visible part of the image in full-image (logical) coordinates."""
        r = QRectF(self.offset.x(), self.offset.y(), self.width() / self.zoom, self._img_h() / self.zoom)
        return r.intersected(QRectF(0, 0, self.logical.width(), self.logical.height()))

    def _to_screen(self, r: QRectF) -> QRectF:
        return QRectF((r.x() - self.offset.x()) * self.zoom, (r.y() - self.offset.y()) * self.zoom,
                      r.width() * self.zoom, r.height() * self.zoom)

    def _split_x(self) -> float:
        """Screen x of the split line."""
        return self.split_pos * self.width()

    def _is_near_split(self, pos: QPointF) -> bool:
        return abs(pos.x() - self._split_x()) <= self._split_grab_half()

    def _split_grab_half(self) -> float:
        return max(20.0, scale.row_h(style.current_s()) * self.SPLIT_GRAB_ROWS)

    def toggle_split(self) -> None:
        self.split_mode = not self.split_mode
        self.update()

    def resizeEvent(self, _e) -> None:  # noqa: N802
        if self._fitted:
            self.fit()
        else:
            self._clamp()

    def wheelEvent(self, e) -> None:  # noqa: N802
        steps = e.angleDelta().y() / 120
        if steps:
            self.zoom_at(self.ZOOM_STEP ** steps, e.position())

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if e.button() == Qt.MouseButton.LeftButton:
            if self._clip_btn_rect.contains(e.position()):
                self.clip_button_clicked.emit()
                return
            if self._split_btn_rect.contains(e.position()):
                self.split_button_clicked.emit()
                return
            if self.split_mode and self._is_near_split(e.position()):
                self._split_dragging = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            # In split mode, left click does NOT toggle original

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._split_dragging:
            self.split_pos = max(0.0, min(1.0, e.position().x() / self.width()))
            self.update()
            return
        # hover: show the resize cursor wherever the split line can be grabbed
        if (self.split_mode and self._pan_from is None and not e.buttons()
                and not self._split_btn_rect.contains(e.position())):
            if self._is_near_split(e.position()):
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.unsetCursor()
        if self._pan_from is not None:
            self.offset -= (e.position() - self._pan_from) / self.zoom
            self._pan_from = e.position()
            self._fitted = False
            self._clamp()
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.MiddleButton:
            self._pan_from = None
            self.unsetCursor()
        if e.button() == Qt.MouseButton.LeftButton and self._split_dragging:
            self._split_dragging = False
            self.unsetCursor()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_F, Qt.Key.Key_0):
            self.fit()
        elif e.key() == Qt.Key.Key_1:
            self.zoom_at(1.0 / self.zoom, QPointF(self.width() / 2, self._img_h() / 2))
        elif e.key() == Qt.Key.Key_S:
            # one toggle path: the dialog switches and remembers it (2026-10-11: view and
            # dialog both toggled, so split could not be switched off)
            self.split_button_clicked.emit()
        elif e.key() == Qt.Key.Key_C and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.clip_button_clicked.emit()
        else:
            super().keyPressEvent(e)

    _CLIP_TIP = ("Clipping view: red = blown highlights, blue = crushed shadows. "
                 "Numbers: share of clipped pixels. Key: C")

    def event(self, e) -> bool:  # noqa: N802
        if e.type() == QEvent.Type.ToolTip:
            # QHelpEvent hat pos(), kein position() — sonst Ausnahme bei jedem Überfahren (2026-10-13)
            if self._clip_btn_rect.contains(QPointF(e.pos())):
                QToolTip.showText(self.mapToGlobal(e.pos()), self._CLIP_TIP, self)
                return True
            QToolTip.hideText()
            e.ignore()
            return True
        return super().event(e)

    def _draw_image(self, p: QPainter, img: QImage, clip: QRectF) -> None:
        """Draw `img` (in logical coords) clipped to `clip` (screen coords)."""
        if img.isNull() or self.logical.isEmpty():
            return
        kx = img.width() / self.logical.width()
        ky = img.height() / self.logical.height()
        vis = self.visible_rect()
        src = QRectF(vis.x() * kx, vis.y() * ky, vis.width() * kx, vis.height() * ky)
        dst = self._to_screen(vis)
        p.setClipRect(clip)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2.0)
        p.drawImage(dst, img, src)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(style.BLACK))
        # K25i: image area excludes the status strip at the bottom
        img_h = self.height() - self._status_h
        if self.split_mode and not self.original_image.isNull():
            sx = self._split_x()
            left_clip = QRectF(0, 0, sx, img_h)
            right_clip = QRectF(sx, 0, self.width() - sx, img_h)
            # Left: grading (self.image) — user 2026-10-10: "mirror split, graded left"
            self._draw_image(p, self.image, left_clip)
            # Right: original
            self._draw_image(p, self.original_image, right_clip)
            if self.show_overlay and not self.overlay.isNull():
                odst = self._to_screen(self.overlay_rect)
                p.setClipRect(left_clip)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                odst.width() < self.overlay.width())
                p.drawImage(odst, self.overlay)
            # Split line: white, 1 px black shadow, no handle (user 2026-10-10/12). The clip
            # left by _draw_image must go first: the line sits exactly on its edge and was cut
            # away (2026-10-12 "split line disappearing"); x snapped to a pixel centre.
            p.setClipping(False)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            lx = math.floor(sx) + 0.5
            p.setPen(QPen(style.SPLIT_LINE_SHADOW, 1))
            p.drawLine(QPointF(lx + 1, 0), QPointF(lx + 1, img_h))
            p.setPen(QPen(style.SPLIT_LINE, 1))
            p.drawLine(QPointF(lx, 0), QPointF(lx, img_h))
        elif not self.image.isNull() and not self.logical.isEmpty():
            kx = self.image.width() / self.logical.width()
            ky = self.image.height() / self.logical.height()
            vis = self.visible_rect()
            src = QRectF(vis.x() * kx, vis.y() * ky, vis.width() * kx, vis.height() * ky)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2.0)
            # K27f: fade-in — draw previous image, then new image with alpha
            if not self._fade_prev.isNull():
                kpx = self._fade_prev.width() / self.logical.width()
                kpy = self._fade_prev.height() / self.logical.height()
                src_p = QRectF(vis.x() * kpx, vis.y() * kpy, vis.width() * kpx, vis.height() * kpy)
                p.drawImage(self._to_screen(vis), self._fade_prev, src_p)
                p.setOpacity(self._fade_alpha)
            p.drawImage(self._to_screen(vis), self.image, src)
            p.setOpacity(1.0)
            if self.show_overlay and not self.overlay.isNull():
                odst = self._to_screen(self.overlay_rect)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                odst.width() < self.overlay.width())
                p.drawImage(odst, self.overlay)
        # K27e/K27f: loading bar (top-right inside the image area)
        if self._loading:
            rh = scale.row_h(style.current_s())
            bar_w = 6 * rh
            bar_h = 2
            margin = round(rh / 2)
            bx = self.width() - bar_w - margin
            by = margin
            alpha, hexcol = style.overlay_style("busy")
            r, g, b = int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)
            # Pulsing: ±15 % brightness, 1 s period
            pulse = 1.0 + 0.15 * math.sin(2.0 * math.pi * time.time())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(style.LOAD_BAR_BG)
            p.drawRect(QRectF(bx, by, bar_w, bar_h))
            fill_w = bar_w * self._loading_progress
            if fill_w > 0:
                p.setBrush(QColor(min(255, int(r * pulse)), min(255, int(g * pulse)),
                                  min(255, int(b * pulse)), alpha))
                p.drawRect(QRectF(bx, by, fill_w, bar_h))
            p.setFont(style.font("small"))
            txt = f"{self._loading_progress * 100:.0f}%"
            tx = bx + bar_w - p.fontMetrics().horizontalAdvance(txt) - 4
            ty = by + bar_h + p.fontMetrics().ascent() + 2
            p.setPen(style.LOAD_TEXT_SHADOW)
            p.drawText(tx + 1, ty + 1, txt)
            p.setPen(QColor(r, g, b))
            p.drawText(tx, ty, txt)
        # status: text + zoom; drawn in a strip only if one is set, otherwise handed to the
        # dialog's label above the main buttons (user 2026-10-11)
        zoom_pct = round(self.zoom * 100)
        mark_text = f"{self.mark} · {zoom_pct} %" if self.mark else f"{zoom_pct} %"
        if mark_text != getattr(self, "_last_status", None):
            self._last_status = mark_text
            # never emit from inside paintEvent: an exception in a slot there aborts PyQt
            QTimer.singleShot(0, self._emit_status)
        if self._status_h > 0:
            strip = QRectF(0, img_h, self.width(), self._status_h)
            p.fillRect(strip, style.STATUS_STRIP_BG)
            p.setPen(QPen(QColor(style.LINE), 1))
            p.drawLine(0, img_h, self.width(), img_h)
            p.setFont(style.font("small"))
            p.setPen(self.mark_color or QColor(style.MUTED))
            p.drawText(strip.adjusted(8, 0, -8, 0),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, mark_text)
        # Split toggle "S", bottom-right corner (user 2026-10-11; the top-right symbol is gone)
        rh = scale.row_h(style.current_s())
        bs = round(rh * 1.2)
        margin = round(rh / 2)
        self._split_btn_rect = QRectF(self.width() - bs - margin, img_h - bs - margin, bs, bs)
        on = self.split_mode
        p.setPen(QPen(QColor(style.ACCENT_HI if on else style.LINE_STRONG), 1))
        p.setBrush(QColor(style.ACCENT) if on else style.TOGGLE_OFF_BG)
        p.drawRoundedRect(self._split_btn_rect, 3, 3)
        p.setFont(style.font("small"))
        p.setPen(QColor(style.TEXT if on else style.MUTED))
        p.drawText(self._split_btn_rect, Qt.AlignmentFlag.AlignCenter, "S")
        # Clip toggle "C", left of S
        self._clip_btn_rect = QRectF(self._split_btn_rect.x() - bs - margin, img_h - bs - margin, bs, bs)
        con = self._clip_view
        p.setPen(QPen(QColor(style.ACCENT_HI if con else style.LINE_STRONG), 1))
        p.setBrush(QColor(style.ACCENT) if con else style.TOGGLE_OFF_BG)
        p.drawRoundedRect(self._clip_btn_rect, 3, 3)
        p.setPen(QColor(style.TEXT if con else style.MUTED))
        p.drawText(self._clip_btn_rect, Qt.AlignmentFlag.AlignCenter, "C")
        # Clip text, left of C
        if self._clip_text:
            p.setPen(QColor(style.MUTED))
            p.drawText(QRectF(0, img_h - bs - margin, self._clip_btn_rect.x() - margin, bs),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, self._clip_text)


# ── Vectorscope ───────────────────────────────────────────────────────────────

class Vectorscope(QWidget):
    """Vectorscope: Cb/Cr-Punktwolke in echten Pixel-Farben, Hue-Ring, Zielmarken."""

    # Rec.709 Cb/Cr für reine Primärfarbton bei 75 % Sättigung.
    _MARKS = style.VS_MARKS
    _SKIN_ANGLE_DEG = 123.0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.data: list[list[float]] = []
        self.data_source: list[list[float]] = []
        self._result: list[list[float]] | None = None
        self._cloud_cache: tuple[QImage, list[list[float]], int, int] | None = None
        _rh = scale.row_h(style.current_s())
        self.setMinimumSize(round(4 * _rh), round(4 * _rh))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_data(self, result: list[list[float]], source: list[list[float]] | None = None) -> None:
        self.data = result
        self.data_source = source or []
        self._result = result
        self.update()

    def _scope_rect(self) -> QRectF:
        side = min(self.width(), self.height())
        ox = (self.width() - side) / 2
        oy = (self.height() - side) / 2
        return QRectF(ox, oy, side, side)

    def _to_scope(self, cb: float, cr: float, r: QRectF) -> QPointF:
        """Cb, Cr in 0..1 → Screen. cb=0.5 Mitte, cr=0.5 Mitte. cr=1 oben."""
        x = r.x() + cb * r.width()
        y = r.y() + (1.0 - cr) * r.height()
        return QPointF(x, y)

    @staticmethod
    def _cbcr_to_rgb(cb: float, cr: float, y: float = 0.5) -> tuple[int, int, int]:
        """Cb, Cr → sRGB 0..255: Farbe nur aus dem Farbwinkel, S=1 V=1, kein Grau."""
        angle = math.atan2(cr - 0.5, cb - 0.5)
        # Map: VS angle 90° (top=R) → HSV 0 (red)
        hue = (angle - math.pi / 2) / (2.0 * math.pi)
        hue = hue % 1.0
        c = QColor.fromHsvF(hue, 1.0, 1.0)
        return (c.red(), c.green(), c.blue())

    @staticmethod
    def _exaggerate(d: float, power: int = VS_EXAGGERATE) -> float:
        """Radial push from center: d' = min(0.95, 1-(1-d)^power)."""
        return min(0.95, 1.0 - (1.0 - d) ** power)

    def _draw_hue_ring(self, p: QPainter, r: QRectF) -> None:
        """Draw a thin colored ring around the scope perimeter."""
        cx, cy = r.center().x(), r.center().y()
        radius = r.width() / 2 - 2
        steps = 360
        for i in range(steps):
            angle = math.radians(i)
            # Hue → Cb/Cr direction (simplified: use hue angle directly)
            cb = 0.5 + 0.4 * math.cos(angle)
            cr = 0.5 + 0.4 * math.sin(angle)
            rgb = self._cbcr_to_rgb(cb, cr)
            x1 = cx + radius * math.cos(angle)
            y1 = cy - radius * math.sin(angle)
            x2 = cx + radius * math.cos(math.radians(i + 2))
            y2 = cy - radius * math.sin(math.radians(i + 2))
            p.setPen(QPen(QColor(*rgb), 1))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    @staticmethod
    def _compute_cloud_image(data: list[list[float]], side: int,
                             alpha_max: int) -> QImage:
        """Render the cloud as a QImage (ARGB32_Premultiplied), bilinear, Neon."""
        n = len(data)
        if n == 0 or side <= 0:
            return QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
        img = QImage(side, side, QImage.Format.Format_ARGB32_Premultiplied)
        buf = (ctypes.c_ubyte * (side * side * 4)).from_address(
            int(img.scanLine(0)))
        buf[:] = b'\x00' * (side * side * 4)
        cx = side / 2.0
        cy = side / 2.0
        radius = side / 2.0
        inv_3 = 1.0 / 3.0
        log21 = math.log(21.0)
        for py in range(side):
            row_off = py * side * 4
            dy = py + 0.5 - cy
            for px in range(side):
                dx = px + 0.5 - cx
                dist = math.sqrt(dx * dx + dy * dy) / radius
                if dist >= 1.0:
                    continue
                d_data = 1.0 - (1.0 - dist) ** inv_3
                if d_data < 0.001:
                    angle = 0.0
                else:
                    angle = math.atan2(dy, dx)
                cb = 0.5 + d_data * math.cos(angle) * 0.5
                cr = 0.5 - d_data * math.sin(angle) * 0.5
                col_f = cb * n - 0.5
                row_f = (1.0 - cr) * n - 0.5
                if col_f < 0 or col_f >= n - 1 or row_f < 0 or row_f >= n - 1:
                    continue
                c0 = int(col_f)
                r0 = int(row_f)
                fc = col_f - c0
                fr = row_f - r0
                v = (data[r0][c0] * (1 - fc) * (1 - fr)
                     + data[r0][c0 + 1] * fc * (1 - fr)
                     + data[r0 + 1][c0] * (1 - fc) * fr
                     + data[r0 + 1][c0 + 1] * fc * fr)
                if v < 0.01:
                    continue
                alpha = int(math.log1p(v * 20.0) / log21 * alpha_max)
                if alpha > 255:
                    alpha = 255
                if dist < 0.03:
                    alpha //= 2
                if alpha <= 0:
                    continue
                deg = int(math.degrees(angle)) % 360
                r_c, g_c, b_c = VS_COLOR_LUT[deg]
                off = row_off + px * 4
                buf[off] = b_c * alpha // 255
                buf[off + 1] = g_c * alpha // 255
                buf[off + 2] = r_c * alpha // 255
                buf[off + 3] = alpha
        return img

    def _draw_cloud(self, p: QPainter, r: QRectF, data: list[list[float]],
                    alpha_max: int) -> None:
        """Wolke als QImage mit Glow, bilinear, Neon-Farbe."""
        n = len(data)
        if n == 0:
            return
        side = int(r.width())
        if side <= 0:
            return
        if (self._cloud_cache is not None
                and self._cloud_cache[1] is data
                and self._cloud_cache[2] == side
                and self._cloud_cache[3] == alpha_max):
            img = self._cloud_cache[0]
        else:
            img = self._compute_cloud_image(data, side, alpha_max)
            self._cloud_cache = (img, data, side, alpha_max)
        # Glow: 1/4 down, smooth up, alpha 0.5, below
        if side > 8:
            small = img.scaled(side // 4, side // 4,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            glow = small.scaled(side, side,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            p.setOpacity(0.5)
            p.drawImage(int(r.x()), int(r.y()), glow)
            p.setOpacity(1.0)
        # Main image
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(int(r.x()), int(r.y()), img)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._scope_rect()
        cx, cy = r.center().x(), r.center().y()
        radius = r.width() / 2

        # Background: black like the wheels beside it (user 2026-10-11)
        p.fillRect(self.rect(), style.VS_BG)

        self._draw_scope_circle(p, r, self.data, alpha_max=255,
                                source_data=self.data_source)

    def _draw_scope_circle(self, p: QPainter, r: QRectF,
                           data: list[list[float]], alpha_max: int,
                           source_data: list[list[float]] | None = None) -> None:
        """Draw one scope circle: rainbow contour, weak crosshair, skin line, cloud."""
        cx, cy = r.center().x(), r.center().y()
        radius = r.width() / 2

        # Background: black inside circle
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(style.VS_BG)
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        # Very weak crosshair
        p.setPen(QPen(style.VS_CROSSHAIR, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(r.x(), cy), QPointF(r.x() + r.width(), cy))
        p.drawLine(QPointF(cx, r.y()), QPointF(cx, r.y() + r.height()))

        # Skin tone line (123°): center outward only (red/yellow upper-left)
        angle = math.radians(self._SKIN_ANGLE_DEG)
        dx = math.cos(angle) * radius
        dy = math.sin(angle) * radius
        p.setPen(QPen(style.VS_SKIN_LINE, 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cx, cy), QPointF(cx + dx, cy - dy))

        # Rainbow contour: QConicalGradient matching _cbcr_to_rgb hue mapping
        stroke = max(1.0, min(r.width(), r.height()) / 200.0)
        grad = QConicalGradient(QPointF(cx, cy), 0.0)
        for i in range(13):
            f = i / 12.0
            hue = (f - 0.25) % 1.0
            c = QColor.fromHsvF(hue, 1.0, 1.0)
            grad.setColorAt(f, c)
        p.setPen(QPen(QBrush(grad), stroke))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), radius - stroke / 2, radius - stroke / 2)

        # Cloud: source first (dimmer), then result
        if source_data:
            self._draw_cloud(p, r, source_data, 100)
        if data:
            self._draw_cloud(p, r, data, alpha_max)


# ── RGB Parade ────────────────────────────────────────────────────────────────

class Waveform(QWidget):
    """Luma waveform (user 2026-10-11, like Kdenlive "Paint mode: White"): one neon image from the
    worker (`scope.luma_waveform` + `parade_neon_all`), drawn over the full width on black."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._image: QImage | None = None
        self.setMinimumHeight(round(3 * scale.row_h(style.current_s())))

    def set_data(self, path: str | None) -> None:
        img = QImage(path or "")
        self._image = None if img.isNull() else img
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), style.WAVE_BG)
        if self._image is not None and w > 1 and h > 1:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(QRectF(0, 0, w, h), self._image)
        p.setPen(QPen(style.WAVE_GRID, 1))
        for frac in (0.25, 0.5, 0.75):
            y = int(h * (1.0 - frac))
            p.drawLine(0, y, w, y)


class Parade(QWidget):
    """RGB Parade: three channel strips side by side, smooth glow on black.

    Data: {"r": path, "g": path, "b": path} — 8-bit grayscale PNGs (256×256),
    row 0 top = value 1.
    """

    _CHANNEL_COLORS = style.PARADE_CHANNEL_COLORS
    _CLIP_SHOW = 0.005

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._images: dict[str, QImage] = {}
        self._glows: dict[str, QImage] = {}
        self.visible_channels: set[str] = {"r", "g", "b"}
        self._clip: dict[str, float] = {}
        self.setMinimumHeight(round(3 * scale.row_h(style.current_s())))

    def set_data(self, data: dict[str, str]) -> None:
        """Load parade PNGs and prepare colored images + glow (once per response)."""
        self._images = {}
        self._glows = {}
        for ch in ("r", "g", "b"):
            path = data.get(ch)
            if not path:
                continue
            gray = QImage(path)   # format from content: neon BMP from the worker, grey PNG in tests
            if gray.isNull():
                continue
            if not gray.isGrayscale():
                # worker sends finished neon colour images (scope.parade_neon): glow baked in
                self._images[ch] = gray.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
                continue
            w, h = gray.width(), gray.height()
            # Channel colour as a flat fill, then the grey value as its alpha mask.
            img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(self._CHANNEL_COLORS[ch])
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawImage(0, 0, self._gray_as_alpha(gray))
            p.end()
            self._images[ch] = img
            # Glow: 1/8 down, smooth up — a wide halo, the DaVinci "radioactive" look
            # (1/4 spread only ~4 px and was barely visible, 2026-10-11)
            small = img.scaled(max(1, w // 8), max(1, h // 8),
                               Qt.AspectRatioMode.IgnoreAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            glow = small.scaled(w, h,
                                Qt.AspectRatioMode.IgnoreAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
            self._glows[ch] = glow
        self.update()

    @staticmethod
    def _gray_as_alpha(gray: QImage) -> QImage:
        """Grey values become the alpha channel. convertToFormat(Format_Alpha8) takes the
        grey image's own alpha (opaque everywhere) and painted solid colour blocks
        (user 2026-10-11: "Parade kaputt") — reinterpret the grey bytes instead."""
        g = gray.convertToFormat(QImage.Format.Format_Grayscale8)
        bits = g.constBits()
        bits.setsize(g.sizeInBytes())
        return QImage(bytes(bits), g.width(), g.height(), g.bytesPerLine(),
                      QImage.Format.Format_Alpha8).copy()

    def set_clip_data(self, clip: dict[str, float] | None) -> None:
        """Set per-channel clipping fractions for display."""
        self._clip = clip or {}
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), style.PARADE_BG)
        if w < 2 or h < 2:
            return

        gap = max(1, int(w * 0.02))
        panel_w = (w - 2 * gap) // 3

        for i, ch in enumerate(("r", "g", "b")):
            if ch not in self.visible_channels:
                continue
            img = self._images.get(ch)
            if img is None:
                continue
            x0 = i * (panel_w + gap)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            # Glow first (90 % opacity, additive)
            glow = self._glows.get(ch)
            if glow is not None:
                p.setOpacity(0.9)
                p.drawImage(x0, 0, glow.scaled(panel_w, h,
                           Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation))
                p.setOpacity(1.0)
            # Main image
            p.drawImage(x0, 0, img.scaled(panel_w, h,
                       Qt.AspectRatioMode.IgnoreAspectRatio,
                       Qt.TransformationMode.SmoothTransformation))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(style.LINE), 1))
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                y = int(h * (1.0 - frac))
                p.drawLine(x0, y, x0 + panel_w, y)

        # Clipping text top-right
        if self._clip:
            p.setFont(style.font("small"))
            ty = 2
            for ch in ("r", "g", "b"):
                hi = self._clip.get(f"{ch}_hi", 0.0)
                lo = self._clip.get(f"{ch}_lo", 0.0)
                col = self._CHANNEL_COLORS[ch]
                col.setAlpha(160)
                p.setPen(QPen(col))
                if hi >= self._CLIP_SHOW:
                    txt = f"{ch.upper()} ▲ {hi * 100:.0f} %"
                    p.drawText(w - p.fontMetrics().horizontalAdvance(txt) - 4, ty + 10, txt)
                    ty += 12
                if lo >= self._CLIP_SHOW:
                    txt = f"{ch.upper()} ▼ {lo * 100:.0f} %"
                    p.drawText(w - p.fontMetrics().horizontalAdvance(txt) - 4, ty + 10, txt)
                    ty += 12
