"""Color Replace section: multiple instances in a 2-column grid, each with 2 small wheels + pipette + reset + 2 sliders."""

from __future__ import annotations

import math

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (QGridLayout, QLabel, QPushButton,
                              QSizePolicy, QToolButton, QVBoxLayout, QWidget)

from . import cursors, scale, style
from .widgets import ColorWheel, SectionHeader, SliderRow

MAX_INSTANCES = 8


class BrightBar(QWidget):
    """Thin brightness bar: black→white gradient with a draggable handle (value 0..1).

    Not a normal slider — a solid gradient line you drag across (user 2026-10-15:
    "not normal slider but a gradient white black line").
    """

    changed = pyqtSignal(str, float)

    def __init__(self, name: str, default: float = 0.5, tip: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self._value = default
        self._drag = False
        self.setFixedHeight(8)  # slim strip: a compact gradient line, not a full slider row
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            self.setToolTip(tip)

    def value(self) -> float:
        return self._value

    def set_value(self, v: float) -> None:
        v = max(0.0, min(1.0, v))
        if abs(v - self._value) > 1e-4:
            self._value = v
            self.update()
            self.changed.emit(self.name, v)

    def _set_from_x(self, x: float) -> None:
        w = max(1, self.width())
        self.set_value(x / w)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = True
            self._set_from_x(e.position().x())
            e.accept()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag and (e.buttons() & Qt.MouseButton.LeftButton):
            self._set_from_x(e.position().x())
            e.accept()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = False
            e.accept()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(255, 255, 255))
        p.fillRect(0, 0, w, h, QBrush(grad))
        hx = int(round(self._value * (w - 1)))
        # handle: light core with dark edges → visible on both the black and white ends
        p.setPen(QPen(QColor(0, 0, 0), 4))
        p.drawLine(hx, 0, hx, h)
        p.setPen(QPen(QColor(style.ACCENT_HI), 2))
        p.drawLine(hx, 0, hx, h)
        p.setPen(QPen(QColor(style.LINE_STRONG), 1))
        p.drawRect(QRectF(0.5, 0.5, w - 1, h - 1))


class _ColorSquare(QWidget):
    """Small colored rect: the only pick trigger for src/tgt. Shows a loupe cursor on hover."""

    clicked = pyqtSignal()

    def __init__(self, tip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._color = "#808080"
        self.setCursor(cursors.loupe())
        if tip:
            self.setToolTip(tip)

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color
        self.update()

    def enterEvent(self, e) -> None:  # noqa: N802
        self.setCursor(cursors.loupe())
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self.unsetCursor()
        super().leaveEvent(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            e.accept()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QBrush(QColor(self._color)))
        p.setPen(QPen(QColor("#2A2A2A"), 1))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)


class _WheelResetBtn(QWidget):
    """Tiny 16×16 reset button (↺) overlaid on a colour wheel's top-right corner.

    Subtle: semi-transparent dark rounded background, gray glyph that turns white on hover.
    """

    clicked = pyqtSignal()

    def __init__(self, tip: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._hover = False
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tip:
            self.setToolTip(tip)

    def enterEvent(self, e) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            e.accept()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 3, 3)
        p.setPen(QColor(255, 255, 255) if self._hover else QColor(0xA0, 0xA0, 0xA0))
        p.setFont(style.font("small"))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "↺")


class _CRInstance(QWidget):
    """One Color Replace instance: 2 small wheels, pipette, reset, intensity, range."""

    changed = pyqtSignal()
    pipette_requested = pyqtSignal(int, str)  # instance_index, "src"|"tgt"
    removed = pyqtSignal(int)  # instance index

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index

        _s = style.current_s()
        wheel_size = round(0.85 * _s * 0.7)   # wheels scaled down 30 % (user 2026-10-15)

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setSpacing(4)
        sq_size = max(16, round(wheel_size * 0.25))

        # ── Row 0: 2 wheels side by side, color square overlay bottom-left ──
        # Col 0: SRC
        src_box = QVBoxLayout()
        src_box.setSpacing(2)
        lbl_src = QLabel("SRC")
        lbl_src.setFont(style.font("small"))
        lbl_src.setProperty("font_role", "small")
        lbl_src.setAlignment(Qt.AlignmentFlag.AlignLeft)
        src_box.addWidget(lbl_src)
        self._src_container = QWidget()
        self._src_container.setFixedSize(wheel_size, wheel_size)
        self.src_wheel = ColorWheel(f"cr_{index}_src", self._src_container)
        self.src_wheel.setGeometry(0, 0, wheel_size, wheel_size)
        self.src_wheel.setToolTip("Source color: the hue to replace")
        self.src_wheel.set_point(0.5, 0.0)
        self._src_color_sq = _ColorSquare("Click to pick source color", self._src_container)
        self._src_color_sq.setFixedSize(sq_size, sq_size)
        self._src_color_sq.move(0, wheel_size - sq_size)
        self._src_color_sq.clicked.connect(lambda: self.pipette_requested.emit(self.index, "src"))
        self._src_reset = _WheelResetBtn("Reset source wheel (hue, saturation, brightness)",
                                         self._src_container)
        self._src_reset.move(wheel_size - 16, 0)
        self._src_reset.clicked.connect(self._on_reset_src)
        src_box.addWidget(self._src_container, 0, Qt.AlignmentFlag.AlignHCenter)
        grid.addLayout(src_box, 0, 0)

        # Col 1: TGT
        tgt_box = QVBoxLayout()
        tgt_box.setSpacing(2)
        lbl_tgt = QLabel("TGT")
        lbl_tgt.setFont(style.font("small"))
        lbl_tgt.setProperty("font_role", "small")
        lbl_tgt.setAlignment(Qt.AlignmentFlag.AlignLeft)
        tgt_box.addWidget(lbl_tgt)
        self._tgt_container = QWidget()
        self._tgt_container.setFixedSize(wheel_size, wheel_size)
        self.tgt_wheel = ColorWheel(f"cr_{index}_tgt", self._tgt_container)
        self.tgt_wheel.setGeometry(0, 0, wheel_size, wheel_size)
        self.tgt_wheel.setToolTip("Target color: the replacement hue")
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        self._tgt_color_sq = _ColorSquare("Click to pick target color", self._tgt_container)
        self._tgt_color_sq.setFixedSize(sq_size, sq_size)
        self._tgt_color_sq.move(0, wheel_size - sq_size)
        self._tgt_color_sq.clicked.connect(lambda: self.pipette_requested.emit(self.index, "tgt"))
        self._tgt_reset = _WheelResetBtn("Reset target wheel (hue, saturation, brightness)",
                                         self._tgt_container)
        self._tgt_reset.move(wheel_size - 16, 0)
        self._tgt_reset.clicked.connect(self._on_reset_tgt)
        tgt_box.addWidget(self._tgt_container, 0, Qt.AlignmentFlag.AlignHCenter)
        grid.addLayout(tgt_box, 0, 1)

        # ── Row 1: brightness gradient bars under the wheels (half width, one line) ──
        self.src_bright = BrightBar(
            f"cr_{index}_src_bright", 1.0,
            tip="Source luminance window (black=shadows, white=highlights)")
        self.tgt_bright = BrightBar(
            f"cr_{index}_tgt_bright", 1.0,
            tip="Target brightness offset (black=darker, white=brighter)")
        grid.addWidget(self.src_bright, 1, 0)
        grid.addWidget(self.tgt_bright, 1, 1)

        # ── Row 2: intensity + range sliders, spanning both columns ──
        slider_box = QVBoxLayout()
        slider_box.setSpacing(2)
        self.intensity = SliderRow(
            f"cr_{index}_intensity", 0.0, 2.0, 0.5, 0.01, "Intensity", label_width=65)
        self.range = SliderRow(
            f"cr_{index}_range", 1.0, 180.0, 30.0, 1.0, "Range °", label_width=65)
        slider_box.addWidget(self.intensity)
        slider_box.addWidget(self.range)
        grid.addLayout(slider_box, 2, 0, 1, 2)

        # ── Close button (×) top-right corner: instance #1 has none ──
        if index > 0:
            self._close_btn = QToolButton(self)
            self._close_btn.setText("×")
            self._close_btn.setFixedSize(16, 16)
            self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._close_btn.setToolTip("Disable this instance (intensity → 0)")
            self._close_btn.clicked.connect(self._on_close)

        # Wire signals
        self.src_wheel.changed.connect(self._on_src_wheel_changed)
        self.tgt_wheel.changed.connect(self._on_tgt_wheel_changed)
        self.src_bright.changed.connect(self._on_src_bright_changed)
        self.tgt_bright.changed.connect(self._on_tgt_bright_changed)
        self.intensity.changed.connect(lambda *_a: self.changed.emit())
        self.range.changed.connect(lambda *_a: self.changed.emit())

    def _on_src_wheel_changed(self, *_a) -> None:
        self._update_squares()
        self.changed.emit()

    def _on_tgt_wheel_changed(self, *_a) -> None:
        self._update_squares()
        self.changed.emit()

    def _on_src_bright_changed(self, *_a) -> None:
        """The src bright bar moved: re-derive the src square so its brightness
        always equals the bar position (real-time sync)."""
        self._update_squares()
        self.changed.emit()

    def _on_tgt_bright_changed(self, *_a) -> None:
        """The tgt bright bar moved: re-derive the tgt square so its brightness
        always equals the bar position (real-time sync)."""
        self._update_squares()
        self.changed.emit()

    def _on_reset(self) -> None:
        self.src_wheel.set_point(0.5, 0.0)
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        self.src_bright.set_value(1.0)
        self.tgt_bright.set_value(1.0)
        self.intensity.set_value(0.5)
        self.range.set_value(30.0)
        self._update_squares()
        self.changed.emit()

    def _on_reset_src(self) -> None:
        """Reset only the source wheel: hue, saturation and brightness to defaults."""
        self.src_wheel.set_point(0.5, 0.0)
        self.src_bright.set_value(1.0)
        self._update_squares()
        self.changed.emit()

    def _on_reset_tgt(self) -> None:
        """Reset only the target wheel: hue, saturation and brightness to defaults."""
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        self.tgt_bright.set_value(1.0)
        self._update_squares()
        self.changed.emit()

    def _on_close(self) -> None:
        """Close: intensity → 0 (the core skips intensity ≤ 0) and hide the instance."""
        self.intensity.set_value(0.0)
        self.setVisible(False)
        self.changed.emit()

    def values(self) -> dict[str, float]:
        """Collect all cr_{i}_* values for this instance (only keys known to the worker)."""
        i = self.index
        src_h, src_s = self._wheel_hsv(self.src_wheel)
        tgt_h, tgt_s = self._wheel_hsv(self.tgt_wheel)
        return {
            f"cr_{i}_src_hue": src_h,
            f"cr_{i}_src_sat": src_s,
            f"cr_{i}_tgt_hue": tgt_h,
            f"cr_{i}_tgt_sat": tgt_s,
            f"cr_{i}_src_bright": self.src_bright.value(),
            f"cr_{i}_tgt_bright": self.tgt_bright.value(),
            f"cr_{i}_intensity": self.intensity.value(),
            f"cr_{i}_range": self.range.value(),
        }

    def set_values(self, vals: dict[str, float]) -> None:
        """Restore values (for session restore)."""
        i = self.index
        src_h = vals.get(f"cr_{i}_src_hue", 0.0)
        src_s = vals.get(f"cr_{i}_src_sat", 1.0)
        tgt_h = vals.get(f"cr_{i}_tgt_hue", 240.0)
        tgt_s = vals.get(f"cr_{i}_tgt_sat", 1.0)
        self._set_wheel_hsv(self.src_wheel, src_h, src_s)
        self._set_wheel_hsv(self.tgt_wheel, tgt_h, tgt_s)
        self.src_bright.set_value(vals.get(f"cr_{i}_src_bright", 1.0))
        self.tgt_bright.set_value(vals.get(f"cr_{i}_tgt_bright", 1.0))
        self.intensity.set_value(vals.get(f"cr_{i}_intensity", 0.5))
        self.range.set_value(vals.get(f"cr_{i}_range", 30.0))
        self._update_squares()

    def set_src_from_rgb(self, r: float, g: float, b: float) -> None:
        """Set the source wheel from an RGB pixel (0-1). Stores full RGB anchor.

        Links the picked pixel's brightness (V = max RGB) to the src_bright bar,
        so the picked color matches itself (proximity filter centered on its V).
        """
        hue, sat = _rgb_to_hue_sat(r, g, b)
        self._set_wheel_hsv(self.src_wheel, hue, sat)
        self.src_bright.set_value(max(r, g, b))
        self._update_squares()
        self.changed.emit()

    def set_tgt_from_rgb(self, r: float, g: float, b: float) -> None:
        """Set the target wheel from an RGB pixel (0-1).

        Links the picked pixel's brightness (V = max RGB) to the tgt_bright bar,
        so the replacement gets the picked color's brightness level.
        """
        hue, sat = _rgb_to_hue_sat(r, g, b)
        self._set_wheel_hsv(self.tgt_wheel, hue, sat)
        self.tgt_bright.set_value(max(r, g, b))
        self._update_squares()
        self.changed.emit()

    def _update_squares(self) -> None:
        """Update both color preview squares.

        Each square is the wheel's hue/saturation at the brightness (V) of its bright
        bar, so the square's brightness always equals the bar's handle position —
        moving the bar updates the square in real time.
        """
        sh, ss = self._wheel_hsv(self.src_wheel)
        sr, sg, sb = _hsv_to_rgb(sh, ss, self.src_bright.value())
        self._src_color_sq.set_color(_rgb_hex(sr, sg, sb))
        th, ts = self._wheel_hsv(self.tgt_wheel)
        tr, tg, tb = _hsv_to_rgb(th, ts, self.tgt_bright.value())
        self._tgt_color_sq.set_color(_rgb_hex(tr, tg, tb))

    @staticmethod
    def _wheel_hsv(wheel: ColorWheel) -> tuple[float, float]:
        """ColorWheel u,v → (hue 0-360, sat 0-1)."""
        u, v = wheel.u, wheel.v
        hue = math.degrees(math.atan2(v, u)) % 360.0
        sat = min(1.0, math.hypot(u, v) * 2)
        return round(hue, 1), round(sat, 3)

    @staticmethod
    def _set_wheel_hsv(wheel: ColorWheel, hue: float, sat: float) -> None:
        """(hue 0-360, sat 0-1) → ColorWheel u,v."""
        rad = math.radians(hue)
        r = sat / 2.0
        wheel.set_point(r * math.cos(rad), r * math.sin(rad))


def _rgb_to_hue_sat(r: float, g: float, b: float) -> tuple[float, float]:
    """sRGB 0-1 → (hue 0-360, sat 0-1)."""
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        return 0.0, 0.0
    sat = d / mx if mx > 0 else 0.0
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    return round(h % 360.0, 1), round(sat, 3)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV (h 0-360, s 0-1, v 0-1) → sRGB 0-1."""
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0.0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0.0
    elif 120 <= h < 180:
        r1, g1, b1 = 0.0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0.0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x
    return (r1 + m, g1 + m, b1 + m)


def _rgb_hex(r: float, g: float, b: float) -> str:
    """sRGB 0-1 → '#RRGGBB' (clamped)."""
    return "#{:02X}{:02X}{:02X}".format(
        min(255, max(0, round(r * 255))), min(255, max(0, round(g * 255))),
        min(255, max(0, round(b * 255))))


class ColorReplaceSection(QWidget):
    """The full Color Replace chapter: header + grid of instances (2 per row) + Add button."""

    changed = pyqtSignal()
    pipette_requested = pyqtSignal(int, str)  # instance_index, "src"|"tgt"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 2)
        box.setSpacing(1)

        self.header = SectionHeader("color_replace", "Color Replace", on=False)
        box.addWidget(self.header)

        self._content = QWidget()
        cbox = QVBoxLayout(self._content)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(4)

        # Grid: 2 instances per row, equal width
        self._grid = QGridLayout()
        self._grid.setSpacing(4)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
        cbox.addLayout(self._grid)

        self._instances: list[_CRInstance] = []
        self._add_instance()

        self._add_btn = QPushButton("+ Add color")
        self._add_btn.setFixedHeight(max(28, scale.button_h(style.current_s())))
        self._add_btn.setToolTip("Add another color replacement instance (max 8)")
        self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_btn.clicked.connect(self._on_add)
        cbox.addWidget(self._add_btn)

        box.addWidget(self._content)

        self.header.weighted.connect(lambda *_a: self.changed.emit())
        self.header.switched.connect(self._on_switched)
        self.header.reset.connect(self._on_reset)

    def _grid_pos(self, idx: int) -> tuple[int, int]:
        """Instance index → (row, col) in the 2-column grid."""
        return divmod(idx, 2)

    def _add_instance(self) -> None:
        if len(self._instances) >= MAX_INSTANCES:
            return
        idx = len(self._instances)
        inst = _CRInstance(idx, self._content)
        inst.changed.connect(self.changed)
        inst.pipette_requested.connect(self.pipette_requested)
        inst.removed.connect(self._remove_instance)
        row, col = self._grid_pos(idx)
        self._grid.addWidget(inst, row, col)
        self._instances.append(inst)

    def _remove_instance(self, index: int) -> None:
        if index <= 0 or index >= len(self._instances):
            return
        inst = self._instances[index]
        self._grid.removeWidget(inst)
        inst.deleteLater()
        del self._instances[index]
        # Re-index and re-layout remaining instances
        for i, remaining in enumerate(self._instances):
            remaining.index = i
            row, col = self._grid_pos(i)
            self._grid.removeWidget(remaining)
            self._grid.addWidget(remaining, row, col)
        self.changed.emit()

    def _on_add(self) -> None:
        self._add_instance()
        self.changed.emit()

    def _on_switched(self, _key: str, on: bool) -> None:
        self._content.setEnabled(on)
        self._add_btn.setEnabled(on)

    def _on_reset(self, _key: str) -> None:
        for inst in self._instances:
            inst._on_reset()
        self.changed.emit()

    def set_pipette_color(self, index: int, r: float, g: float, b: float, which: str = "src") -> None:
        """Set the src or tgt wheel of instance `index` from an RGB pixel."""
        if 0 <= index < len(self._instances):
            inst = self._instances[index]
            if which == "tgt":
                inst.set_tgt_from_rgb(r, g, b)
            else:
                inst.set_src_from_rgb(r, g, b)

    def values(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for inst in self._instances:
            out.update(inst.values())
        return out

    def set_values(self, vals: dict[str, float]) -> None:
        for inst in self._instances:
            inst.set_values(vals)

    def is_on(self) -> bool:
        return self.header.is_on()
