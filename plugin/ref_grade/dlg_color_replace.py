"""Color Replace section: multiple instances, each with 2 color wheels + 2 sliders."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QHBoxLayout, QPushButton, QVBoxLayout, QWidget)

from . import scale, style
from .widgets import ColorWheel, SectionHeader, SliderRow

MAX_INSTANCES = 5


class _CRInstance(QWidget):
    """One Color Replace instance: source wheel, target wheel, intensity, range."""

    changed = pyqtSignal()

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.index = index
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 4)
        box.setSpacing(2)

        # Row 1: two color wheels side by side
        wheel_row = QHBoxLayout()
        wheel_row.setSpacing(8)
        _rh = scale.row_h(style.current_s())
        wheel_size = round(3 * _rh)

        self.src_wheel = ColorWheel(f"cr_{index}_src", self)
        self.src_wheel.setFixedSizes(wheel_size, wheel_size) if hasattr(self.src_wheel, 'setFixedSizes') else None
        self.src_wheel.setFixedSize(wheel_size, wheel_size)
        self.src_wheel.setToolTip("Source color: the hue range to replace")

        self.tgt_wheel = ColorWheel(f"cr_{index}_tgt", self)
        self.tgt_wheel.setFixedSize(wheel_size, wheel_size)
        self.tgt_wheel.setToolTip("Target color: the replacement hue")

        wheel_row.addWidget(self.src_wheel)
        wheel_row.addWidget(self.tgt_wheel)
        wheel_row.addStretch(1)
        box.addLayout(wheel_row)

        # Row 2: intensity + range sliders
        self.intensity = SliderRow(
            f"cr_{index}_intensity", 0.0, 1.0, 0.0, 0.01, "Intensity", label_width=70)
        self.range = SliderRow(
            f"cr_{index}_range", 1.0, 180.0, 30.0, 1.0, "Range °", label_width=70)
        box.addWidget(self.intensity)
        box.addWidget(self.range)

        # Wire signals
        self.src_wheel.changed.connect(lambda *_a: self.changed.emit())
        self.tgt_wheel.changed.connect(lambda *_a: self.changed.emit())
        self.intensity.changed.connect(lambda *_a: self.changed.emit())
        self.range.changed.connect(lambda *_a: self.changed.emit())

    def values(self) -> dict[str, float]:
        """Collect all cr_{i}_* values for this instance."""
        i = self.index
        # Wheel: u,v → hue, sat
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
        tgt_h = vals.get(f"cr_{i}_tgt_hue", 0.0)
        tgt_s = vals.get(f"cr_{i}_tgt_sat", 1.0)
        self._set_wheel_hsv(self.src_wheel, src_h, src_s)
        self._set_wheel_hsv(self.tgt_wheel, tgt_h, tgt_s)
        self.intensity.set_value(vals.get(f"cr_{i}_intensity", 0.0))
        self.range.set_value(vals.get(f"cr_{i}_range", 30.0))

    @staticmethod
    def _wheel_hsv(wheel: ColorWheel) -> tuple[float, float]:
        """ColorWheel u,v → (hue 0-360, sat 0-1)."""
        import math
        u, v = wheel.u, wheel.v
        hue = math.degrees(math.atan2(v, u)) % 360.0
        sat = min(1.0, math.hypot(u, v) * 2)  # wheel radius = 0.5 → sat 1.0
        return round(hue, 1), round(sat, 3)

    @staticmethod
    def _set_wheel_hsv(wheel: ColorWheel, hue: float, sat: float) -> None:
        """(hue 0-360, sat 0-1) → ColorWheel u,v."""
        import math
        rad = math.radians(hue)
        r = sat / 2.0  # sat 1.0 → distance 0.5 (rim)
        wheel.set_point(r * math.cos(rad), r * math.sin(rad))


class ColorReplaceSection(QWidget):
    """The full Color Replace chapter: header + N instances + Add button."""

    changed = pyqtSignal()

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

        # Start with one instance
        self._instances: list[_CRInstance] = []
        self._add_instance()

        # Add button
        self._add_btn = QPushButton("+ Add color")
        self._add_btn.setFixedHeight(scale.row_h(style.current_s()))
        self._add_btn.setToolTip("Add another color replacement instance (max 5)")
        self._add_btn.setCursor(__import__('PyQt6.QtCore', fromlist=['Qt']).Qt.CursorShape.PointingHandCursor)
        self._add_btn.clicked.connect(self._on_add)
        cbox.addWidget(self._add_btn)

        box.addWidget(self._content)

        # Wire header
        self.header.weighted.connect(lambda *_a: self.changed.emit())
        self.header.switched.connect(self._on_switched)
        self.header.reset.connect(self._on_reset)

    def _add_instance(self) -> None:
        if len(self._instances) >= MAX_INSTANCES:
            return
        inst = _CRInstance(len(self._instances), self._content)
        inst.changed.connect(self.changed)
        # Insert before the Add button
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
            inst.set_values({})
        self.changed.emit()

    def values(self) -> dict[str, float]:
        """All cr_* values from all instances."""
        out: dict[str, float] = {}
        for inst in self._instances:
            out.update(inst.values())
        return out

    def set_values(self, vals: dict[str, float]) -> None:
        for inst in self._instances:
            inst.set_values(vals)

    def is_on(self) -> bool:
        return self.header.is_on()
