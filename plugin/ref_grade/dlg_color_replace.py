"""Color Replace section: multiple instances in a 2-column grid, each with 2 small wheels + pipette + reset + 2 sliders."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from . import scale, style
from .widgets import ColorWheel, SectionHeader, SliderRow, small_button

MAX_INSTANCES = 8


class _CRInstance(QWidget):
    """One Color Replace instance: 2 small wheels, pipette, reset, intensity, range."""

    changed = pyqtSignal()
    pipette_requested = pyqtSignal(int, str)  # instance_index, "src"|"tgt"
    removed = pyqtSignal(int)  # instance index

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index
        self._src_rgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
        self._tgt_rgb: tuple[float, float, float] = (0.25, 0.25, 0.75)

        _s = style.current_s()
        wheel_size = min(65, round(0.70 * _s))
        btn_h = max(28, scale.button_h(_s))

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setSpacing(4)
        sq_size = max(14, round(wheel_size * 0.22))

        # ── Row 0: 2 wheels side by side ──
        # Col 0: SRC
        src_box = QVBoxLayout()
        src_box.setSpacing(2)
        lbl_src = QLabel("SRC")
        lbl_src.setFont(style.font("small"))
        lbl_src.setProperty("font_role", "small")
        lbl_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        src_box.addWidget(lbl_src)
        self.src_wheel = ColorWheel(f"cr_{index}_src", self)
        self.src_wheel.setFixedSize(wheel_size, wheel_size)
        self.src_wheel.setToolTip("Source color: the hue to replace")
        self.src_wheel.set_point(0.5, 0.0)
        src_box.addWidget(self.src_wheel, 0, Qt.AlignmentFlag.AlignHCenter)
        self._src_color_sq = QLabel()
        self._src_color_sq.setFixedSize(sq_size, sq_size)
        self._src_color_sq.setStyleSheet("background: #808080; border: 1px solid #2A2A2A;")
        self._src_color_sq.setToolTip("Sampled source color")
        src_box.addWidget(self._src_color_sq, 0, Qt.AlignmentFlag.AlignHCenter)
        grid.addLayout(src_box, 0, 0)

        # Col 1: TGT
        tgt_box = QVBoxLayout()
        tgt_box.setSpacing(2)
        lbl_tgt = QLabel("TGT")
        lbl_tgt.setFont(style.font("small"))
        lbl_tgt.setProperty("font_role", "small")
        lbl_tgt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tgt_box.addWidget(lbl_tgt)
        self.tgt_wheel = ColorWheel(f"cr_{index}_tgt", self)
        self.tgt_wheel.setFixedSize(wheel_size, wheel_size)
        self.tgt_wheel.setToolTip("Target color: the replacement hue")
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        tgt_box.addWidget(self.tgt_wheel, 0, Qt.AlignmentFlag.AlignHCenter)
        self._tgt_color_sq = QLabel()
        self._tgt_color_sq.setFixedSize(sq_size, sq_size)
        self._tgt_color_sq.setStyleSheet("background: #4040C0; border: 1px solid #2A2A2A;")
        self._tgt_color_sq.setToolTip("Target color")
        tgt_box.addWidget(self._tgt_color_sq, 0, Qt.AlignmentFlag.AlignHCenter)
        grid.addLayout(tgt_box, 0, 1)

        # ── Row 1: 2 sliders stacked, spanning both columns ──
        slider_box = QVBoxLayout()
        slider_box.setSpacing(2)
        self.intensity = SliderRow(
            f"cr_{index}_intensity", 0.0, 1.0, 0.5, 0.01, "Intensity", label_width=65)
        self.range = SliderRow(
            f"cr_{index}_range", 1.0, 180.0, 30.0, 1.0, "Range °", label_width=65)
        slider_box.addWidget(self.intensity)
        slider_box.addWidget(self.range)
        grid.addLayout(slider_box, 1, 0, 1, 2)

        # ── Row 2: buttons (Source, Target, ↺, ×) in a horizontal row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        btn_row.addStretch()
        self._pipette_src = QPushButton("Source")
        self._pipette_src.setFixedHeight(btn_h)
        self._pipette_src.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pipette_src.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pipette_src.setToolTip("Pick a pixel to set the source color")
        self._pipette_src.clicked.connect(lambda: self.pipette_requested.emit(self.index, "src"))
        btn_row.addWidget(self._pipette_src)
        self._pipette_tgt = QPushButton("Target")
        self._pipette_tgt.setFixedHeight(btn_h)
        self._pipette_tgt.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pipette_tgt.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pipette_tgt.setToolTip("Pick a pixel to set the target color")
        self._pipette_tgt.clicked.connect(lambda: self.pipette_requested.emit(self.index, "tgt"))
        btn_row.addWidget(self._pipette_tgt)
        self._reset_btn = small_button("↺", tip=f"Reset instance {index + 1}")
        self._reset_btn.setFixedSize(btn_h, btn_h)
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        self._remove_btn = small_button("×", tip="Remove this instance")
        self._remove_btn.setFixedSize(btn_h, btn_h)
        self._remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self.index))
        self._remove_btn.setVisible(index > 0)
        btn_row.addWidget(self._remove_btn)
        grid.addLayout(btn_row, 2, 0, 1, 2)

        # Wire signals
        self.src_wheel.changed.connect(self._on_src_wheel_changed)
        self.tgt_wheel.changed.connect(self._on_tgt_wheel_changed)
        self.intensity.changed.connect(lambda *_a: self.changed.emit())
        self.range.changed.connect(lambda *_a: self.changed.emit())

    def _on_src_wheel_changed(self, *_a) -> None:
        h, s = self._wheel_hsv(self.src_wheel)
        r, g, b = _hsv_to_rgb(h, s, 1.0)
        self._src_rgb = (r, g, b)
        self._update_squares()
        self.changed.emit()

    def _on_tgt_wheel_changed(self, *_a) -> None:
        h, s = self._wheel_hsv(self.tgt_wheel)
        r, g, b = _hsv_to_rgb(h, s, 1.0)
        self._tgt_rgb = (r, g, b)
        self._update_squares()
        self.changed.emit()

    def _on_reset(self) -> None:
        self.src_wheel.set_point(0.5, 0.0)
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        self.intensity.set_value(0.5)
        self.range.set_value(30.0)
        self._src_rgb = (0.5, 0.5, 0.5)
        self._tgt_rgb = (0.25, 0.25, 0.75)
        self._update_squares()
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
        self.intensity.set_value(vals.get(f"cr_{i}_intensity", 0.5))
        self.range.set_value(vals.get(f"cr_{i}_range", 30.0))

    def set_src_from_rgb(self, r: float, g: float, b: float) -> None:
        """Set the source wheel from an RGB pixel (0-1). Stores full RGB anchor."""
        hue, sat = _rgb_to_hue_sat(r, g, b)
        self._set_wheel_hsv(self.src_wheel, hue, sat)
        self._src_rgb = (r, g, b)
        self._update_squares()
        self.changed.emit()

    def set_tgt_from_rgb(self, r: float, g: float, b: float) -> None:
        """Set the target wheel from an RGB pixel (0-1)."""
        hue, sat = _rgb_to_hue_sat(r, g, b)
        self._set_wheel_hsv(self.tgt_wheel, hue, sat)
        self._tgt_rgb = (r, g, b)
        self._update_squares()
        self.changed.emit()

    def _update_squares(self) -> None:
        """Update both color preview squares."""
        r, g, b = self._src_rgb
        hex_src = "#{:02X}{:02X}{:02X}".format(
            min(255, round(r * 255)), min(255, round(g * 255)), min(255, round(b * 255)))
        self._src_color_sq.setStyleSheet(
            f"background: {hex_src}; border: 1px solid #2A2A2A;")
        tr, tg, tb = self._tgt_rgb
        hex_tgt = "#{:02X}{:02X}{:02X}".format(
            min(255, round(tr * 255)), min(255, round(tg * 255)), min(255, round(tb * 255)))
        self._tgt_color_sq.setStyleSheet(
            f"background: {hex_tgt}; border: 1px solid #2A2A2A;")

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
        self._add_btn.setFixedHeight(scale.row_h(style.current_s()))
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
