"""Color Replace section: multiple instances, each with 2 color wheels (75% size) + pipette + reset + 2 sliders."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from . import scale, style
from .widgets import ColorWheel, SectionHeader, SliderRow, small_button

MAX_INSTANCES = 5


class _CRInstance(QWidget):
    """One Color Replace instance: 2 wheels (75%), pipette, reset, intensity, range."""

    changed = pyqtSignal()
    pipette_requested = pyqtSignal(int, str)  # instance_index, "src"|"tgt"

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 4)
        box.setSpacing(2)

        _rh = scale.row_h(style.current_s())
        # 75% of the WHEELS section wheel size
        wheel_size = round(0.75 * style.current_s())
        btn_size = round(_rh * 1.4)

        # Row 0: labels
        label_row = QHBoxLayout()
        label_row.setSpacing(6)
        lbl_src = QLabel("Source")
        lbl_src.setFont(style.font("small"))
        lbl_src.setProperty("font_role", "small")
        lbl_src.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_tgt = QLabel("Target")
        lbl_tgt.setFont(style.font("small"))
        lbl_tgt.setProperty("font_role", "small")
        lbl_tgt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label_row.addWidget(lbl_src, 1)
        label_row.addWidget(lbl_tgt, 1)
        label_row.addStretch(1)
        box.addLayout(label_row)

        # Row 1: two wheels + pipette + reset
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self.src_wheel = ColorWheel(f"cr_{index}_src", self)
        self.src_wheel.setFixedSize(wheel_size, wheel_size)
        self.src_wheel.setToolTip("Source color: the hue to replace")
        # Default: red at full saturation (right edge)
        self.src_wheel.set_point(0.5, 0.0)

        self.tgt_wheel = ColorWheel(f"cr_{index}_tgt", self)
        self.tgt_wheel.setFixedSize(wheel_size, wheel_size)
        self.tgt_wheel.setToolTip("Target color: the replacement hue")
        # Default: blue at full saturation
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))

        row1.addWidget(self.src_wheel)
        row1.addWidget(self.tgt_wheel)

        # Buttons: pipette + reset
        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)
        self._pipette_btn = small_button("◉", tip="Pick source color from image (click on image)")
        self._pipette_btn.setFixedSize(btn_size, btn_size)
        self._pipette_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pipette_btn.clicked.connect(lambda: self.pipette_requested.emit(self.index, "src"))
        btn_col.addWidget(self._pipette_btn)

        self._reset_btn = small_button("↺", tip=f"Reset instance {index + 1}")
        self._reset_btn.setFixedSize(btn_size, btn_size)
        self._reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reset_btn.clicked.connect(self._on_reset)
        btn_col.addWidget(self._reset_btn)
        btn_col.addStretch(1)

        row1.addLayout(btn_col)
        row1.addStretch(1)
        box.addLayout(row1)

        # Row 2: intensity + range sliders
        self.intensity = SliderRow(
            f"cr_{index}_intensity", 0.0, 1.0, 1.0, 0.01, "Intensity", label_width=70)
        self.range = SliderRow(
            f"cr_{index}_range", 1.0, 180.0, 30.0, 1.0, "Range °", label_width=70)
        self.range.setToolTip("Falloff: how many degrees of neighboring hues are affected")
        box.addWidget(self.intensity)
        box.addWidget(self.range)

        # Wire signals
        self.src_wheel.changed.connect(lambda *_a: self.changed.emit())
        self.tgt_wheel.changed.connect(lambda *_a: self.changed.emit())
        self.intensity.changed.connect(lambda *_a: self.changed.emit())
        self.range.changed.connect(lambda *_a: self.changed.emit())

    def _on_reset(self) -> None:
        self.src_wheel.set_point(0.5, 0.0)
        self.tgt_wheel.set_point(-0.5 * math.cos(math.radians(60)), -0.5 * math.sin(math.radians(60)))
        self.intensity.set_value(1.0)
        self.range.set_value(30.0)
        self.changed.emit()

    def values(self) -> dict[str, float]:
        """Collect all cr_{i}_* values for this instance."""
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
        self.intensity.set_value(vals.get(f"cr_{i}_intensity", 1.0))
        self.range.set_value(vals.get(f"cr_{i}_range", 30.0))

    def set_src_from_rgb(self, r: float, g: float, b: float) -> None:
        """Set the source wheel from an RGB pixel (0-1)."""
        hue, sat = _rgb_to_hue_sat(r, g, b)
        self._set_wheel_hsv(self.src_wheel, hue, sat)
        self.changed.emit()

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


class ColorReplaceSection(QWidget):
    """The full Color Replace chapter: header + N instances + Add button."""

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

        self._instances: list[_CRInstance] = []
        self._add_instance()

        self._add_btn = QPushButton("+ Add color")
        self._add_btn.setFixedHeight(scale.row_h(style.current_s()))
        self._add_btn.setToolTip("Add another color replacement instance (max 5)")
        self._add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_btn.clicked.connect(self._on_add)
        cbox.addWidget(self._add_btn)

        box.addWidget(self._content)

        self.header.weighted.connect(lambda *_a: self.changed.emit())
        self.header.switched.connect(self._on_switched)
        self.header.reset.connect(self._on_reset)

    def _add_instance(self) -> None:
        if len(self._instances) >= MAX_INSTANCES:
            return
        inst = _CRInstance(len(self._instances), self._content)
        inst.changed.connect(self.changed)
        inst.pipette_requested.connect(self.pipette_requested)
        cbox = self._content.layout()
        cbox.insertWidget(cbox.count() - 1, inst)
        self._instances.append(inst)

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

    def set_pipette_color(self, index: int, r: float, g: float, b: float) -> None:
        """Set the source wheel of instance `index` from an RGB pixel."""
        if 0 <= index < len(self._instances):
            self._instances[index].set_src_from_rgb(r, g, b)

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
