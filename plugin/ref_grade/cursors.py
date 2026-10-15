"""CR pipette cursors — corporate design: one UI colour, no blue, no decoration.

Source = classic eyedropper at 45° (bulb top-right, thin tube) + cross at the
bottom-left sample point (user ref 2026-10-15). Target = clean sniper crosshair.
Both monochrome, drawn crisp (no antialiasing) for the pixel look.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap

from . import style

_COLOR = QColor(style.ACCENT_HI)  # single UI colour (corporate design)


def _base(size: int) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    return pm, p


def pipette() -> QCursor:
    """Eyedropper at 45° + cross at the bottom-left sample point. Hotspot = cross centre."""
    size = 32
    pm, p = _base(size)
    pen = QPen(_COLOR)
    pen.setWidth(2)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setPen(pen)
    # bulb (top-right), hollow
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(21, 2, 9, 9)
    # thin tube: two parallel 45° lines (hollow glass tube) from bulb down to the tip
    p.drawLine(22, 11, 15, 18)
    p.drawLine(19, 13, 12, 20)
    p.drawLine(15, 18, 12, 20)  # tip cap
    # cross = sample point (bottom-left)
    cx, cy = 9, 23
    p.drawLine(cx - 4, cy, cx + 4, cy)
    p.drawLine(cx, cy - 4, cx, cy + 4)
    p.end()
    return QCursor(pm, cx, cy)


def crosshair() -> QCursor:
    """Clean sniper crosshair: cross lines + ring + centre dot. Hotspot = centre."""
    size = 32
    pm, p = _base(size)
    pen = QPen(_COLOR)
    pen.setWidth(2)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    c = 16
    p.drawLine(c, 1, c, size - 1)
    p.drawLine(1, c, size - 1, c)
    p.drawEllipse(9, 9, 14, 14)
    p.setBrush(_COLOR)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(c - 1, c - 1, 2, 2)
    p.end()
    return QCursor(pm, c, c)
