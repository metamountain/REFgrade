"""Zone visualizer — a small B/W bar under the wheels showing the shadow/mid/highlight
zone masks. It mirrors the EXACT mask computation in ``core/refgrade/grade.py::_zone_masks``,
so it shows both Gain (core size) and Falloff (blur width): three white bumps with black
gaps, and how the gaps close as the bumps grow. It re-renders only when Gain or Falloff
change — never on a wheel drag, which is what the bars are not about.

2026-10-18 (user): "a graphic to visualize this? black and white under the wheels …
combined for gain and falloff" — one graphic, both parameters.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QWidget

from . import schemes


def _ss(e0: float, e1: float, x: float) -> float:
    """Smoothstep: 0 below e0, 1 above e1, cubic in between — exactly like the core."""
    if e1 <= e0:
        return 0.0 if x < e0 else 1.0
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


class ZoneVisualizer(QWidget):
    """Three stacked B/W bars (Shadows / Mids / Highlights) over the luminance axis 0→1.

    Each bar shows one zone mask as a white→black gradient: white = the wheel acts fully
    on that tone, black = no effect. The zones are soft bumps (glow): Gain sizes the
    white core, Falloff the blur around it. At the default there are big black gaps
    between the bumps; raising Gain/Falloff closes them.
    """

    BAR_H = 3   # 2026-10-18 (user): schlanke Linien, keine grauen Streifen dazwischen
    GAP = 0

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._gain = 0.1       # = core default (sliders.py)
        self._falloff = 0.1    # = core default
        self.setFixedHeight(self.BAR_H * 3 + self.GAP * 2)
        self.setMinimumWidth(160)
        self.setToolTip(
            "Zone masks Shadows / Mids / Highlights — white = the wheel acts on this tone. "
            "Gain grows the white core areas, Falloff the soft edge (glow). "
            "Gain and Falloff close the black gaps between the zones.")

    def set_gain(self, v: float) -> None:
        self._gain = float(v)
        self.update()

    def set_falloff(self, v: float) -> None:
        self._falloff = float(v)
        self.update()

    def _masks(self, y: float) -> tuple[float, float, float]:
        """The exact S/M/H masks from ``core/refgrade/grade.py::_zone_masks`` for luminance ``y``."""
        g = self._gain
        f = self._falloff

        def _bump(dist: float) -> float:
            if f <= 0.0:
                return 1.0 if dist <= 0.0 else 0.0
            return 1.0 - _ss(0.0, f, min(max(dist, 0.0), f))

        s = _bump(y - g)
        h = _bump((1.0 - g) - y)
        m = _bump(abs(y - 0.5) - g)
        return s, m, h

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        w = self.width()
        p.fillRect(self.rect(), QColor(schemes.role_color("surface")))
        if w >= 2:
            # Precompute the per-column masks once (g/f are constant across the bars).
            cols = [self._masks(x / (w - 1)) for x in range(w)]
            for i in range(3):
                y0 = i * (self.BAR_H + self.GAP)
                for x in range(w):
                    g = int(255 * max(0.0, min(1.0, cols[x][i])))
                    p.fillRect(x, y0, 1, self.BAR_H, QColor(g, g, g))
        p.end()
