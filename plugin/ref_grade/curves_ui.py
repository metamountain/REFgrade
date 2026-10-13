"""Kurven-Editor-Widget: Gitter 4×4, Kurven, Punkte setzen/ziehen/löschen.

Kein krita-Import. Die PCHIP-Mathematik steht in pchip.py.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from . import style
from .pchip import pchip_curve, pchip_cyclic_curve

CHANNELS = ("rgb", "r", "g", "b")
HUE_CHANNELS = ("hue_hue", "hue_sat", "hue_lum", "lum_sat", "sat_sat")
ALL_CHANNELS = CHANNELS + HUE_CHANNELS
CHANNEL_COLORS = {
    "rgb": style.TEXT,
    "r": "#B43C3C",
    "g": "#3CB43C",
    "b": "#3C3CB4",
    "hue_hue": "#CC8833",
    "hue_sat": "#CC3388",
    "hue_lum": "#3388CC",
    "lum_sat": "#33CC88",
    "sat_sat": "#8833CC",
}
_HUE_CYCLIC = ("hue_hue", "hue_sat", "hue_lum")
_MAX_POINTS = 16
_IDENTITY = [[0.0, 0.0], [1.0, 1.0]]
_HUE_IDENTITY = [[0.0, 0.5], [1.0, 0.5]]


class CurvesEditor(QWidget):
    """Quadratischer Kurven-Editor: Gitter, Kurve, Punkte.

    Klick setzt Punkt, Ziehen bewegt, Rechtsklick/Doppelklick entfernt (nicht Endpunkte).
    Höchstens 16 Punkte; Endpunkte nur in y beweglich.
    Hue-Tabs: zyklisch, Hintergrund-Verlauf, Neutral bei y=0.5.
    """

    changed = pyqtSignal(str, list)  # channel, [[x, y], …]
    drag_started = pyqtSignal()      # K31b-F
    drag_stopped = pyqtSignal()      # K31b-F

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.curves: dict[str, list[list[float]]] = {}
        for ch in CHANNELS:
            self.curves[ch] = [list(_IDENTITY[0]), list(_IDENTITY[1])]
        for ch in HUE_CHANNELS:
            self.curves[ch] = [list(_HUE_IDENTITY[0]), list(_HUE_IDENTITY[1])]
        self.active_channel = "rgb"
        self.histogram: list[float] | None = None
        self._drag_idx: int | None = None
        self._sel_idx: int | None = None          # selected point (Krita-style handles)
        self._hover_idx: int | None = None
        self._doomed = False                      # dragged outside: released → deleted
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _identity_for(self, ch: str) -> list[list[float]]:
        if ch in HUE_CHANNELS:
            return [list(_HUE_IDENTITY[0]), list(_HUE_IDENTITY[1])]
        return [list(_IDENTITY[0]), list(_IDENTITY[1])]

    def _is_cyclic(self, ch: str) -> bool:
        return ch in _HUE_CYCLIC

    # ── Daten ─────────────────────────────────────────────────────────────────

    def set_curve(self, channel: str, points: list[list[float]]) -> None:
        """Kurve eines Kanals setzen und neu zeichnen."""
        self.curves[channel] = [list(p) for p in points]
        self.update()

    def get_curve(self, channel: str) -> list[list[float]]:
        return [list(p) for p in self.curves[channel]]

    def set_active(self, channel: str) -> None:
        if channel in ALL_CHANNELS:
            self.active_channel = channel
            self.update()

    def set_histogram(self, hist: list[float] | None) -> None:
        self.histogram = hist
        self.update()

    def reset_channel(self) -> None:
        """Reset the active channel to identity and emit changed."""
        self.curves[self.active_channel] = self._identity_for(self.active_channel)
        self.update()
        self.changed.emit(self.active_channel, self.get_curve(self.active_channel))

    # ── Geometrie ─────────────────────────────────────────────────────────────

    def _plot_rect(self) -> QRectF:
        m = 4
        return QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)

    def _to_screen(self, x: float, y: float) -> QPointF:
        r = self._plot_rect()
        return QPointF(r.x() + x * r.width(), r.y() + (1.0 - y) * r.height())

    def _to_data(self, pos: QPointF) -> tuple[float, float]:
        r = self._plot_rect()
        x = (pos.x() - r.x()) / r.width()
        y = 1.0 - (pos.y() - r.y()) / r.height()
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    # ── Zeichnen ──────────────────────────────────────────────────────────────

    def _draw_background(self, p: QPainter, r: QRectF) -> None:
        """Hintergrund je nach aktivem Kanal."""
        ch = self.active_channel
        if ch in _HUE_CYCLIC:
            # Hue-Verlauf: Rot → Gelb → Grün → Cyan → Blau → Magenta → Rot, 25 % abgedunkelt
            grad = QLinearGradient(r.left(), 0, r.right(), 0)
            steps = 7
            for i in range(steps):
                hue_frac = i / (steps - 1)
                c = QColor.fromHsvF(hue_frac, 1.0, 1.0)
                c.setAlphaF(0.25)
                grad.setColorAt(i / (steps - 1), c)
            p.fillRect(r, grad)
        elif ch == "lum_sat":
            # Schwarz → Weiß, 25 %
            grad = QLinearGradient(r.left(), 0, r.right(), 0)
            c0 = QColor(0, 0, 0); c0.setAlphaF(0.25)
            c1 = QColor(255, 255, 255); c1.setAlphaF(0.25)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, c1)
            p.fillRect(r, grad)
        elif ch == "sat_sat":
            # Grau → gesättigt, 25 %
            grad = QLinearGradient(r.left(), 0, r.right(), 0)
            c0 = QColor(128, 128, 128); c0.setAlphaF(0.25)
            c1 = QColor(255, 0, 0); c1.setAlphaF(0.25)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, c1)
            p.fillRect(r, grad)

    def _draw_neutral(self, p: QPainter, r: QRectF) -> None:
        """Neutral-Linie: Diagonale für RGB, waagerecht y=0.5 für Hue."""
        p.setPen(QPen(QColor(style.LINE), 1, Qt.PenStyle.DashLine))
        if self.active_channel in HUE_CHANNELS:
            y = r.y() + (1.0 - 0.5) * r.height()
            p.drawLine(QPointF(r.x(), y), QPointF(r.x() + r.width(), y))
        else:
            p.drawLine(r.bottomLeft(), r.topRight())

    def _curve_values(self, ch: str, n: int) -> list[float]:
        """PCHIP-Werte für einen Kanal (zyklisch oder linear)."""
        pts = self.curves[ch]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if self._is_cyclic(ch):
            return pchip_cyclic_curve(xs, ys, n)
        return pchip_curve(xs, ys, n)

    def _is_identity_curve(self, ch: str) -> bool:
        pts = self.curves[ch]
        if ch in HUE_CHANNELS:
            return all(abs(p[1] - 0.5) < 1e-6 for p in pts)
        return (len(pts) == 2 and pts[0] == [0.0, 0.0] and pts[1] == [1.0, 1.0])

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot_rect()

        # Hintergrund (nur Hue-Kanäle)
        self._draw_background(p, r)

        # Histogramm 10 % Grau dahinter
        if self.histogram:
            peak = max(self.histogram) or 1.0
            n = len(self.histogram)
            for i, v in enumerate(self.histogram):
                h = v / peak * r.height()
                p.fillRect(QRectF(r.x() + i * r.width() / n, r.y() + r.height() - h,
                                  r.width() / n + 0.5, h), QColor(255, 255, 255, 26))

        # Gitter 4×4
        p.setPen(QPen(QColor(style.LINE), 1))
        for i in range(1, 4):
            x = r.x() + i * r.width() / 4
            y = r.y() + i * r.height() / 4
            p.drawLine(QPointF(x, r.y()), QPointF(x, r.y() + r.height()))
            p.drawLine(QPointF(r.x(), y), QPointF(r.x() + r.width(), y))

        # Neutral-Linie
        self._draw_neutral(p, r)

        # Inaktive Kanäle als dünne Linien
        for ch in ALL_CHANNELS:
            if ch == self.active_channel:
                continue
            if self._is_identity_curve(ch):
                continue
            vals = self._curve_values(ch, 128)
            p.setPen(QPen(QColor(CHANNEL_COLORS[ch]), 1))
            path_pts = [self._to_screen(i / 127, v) for i, v in enumerate(vals)]
            for i in range(len(path_pts) - 1):
                p.drawLine(path_pts[i], path_pts[i + 1])

        # Aktive Kurve
        vals = self._curve_values(self.active_channel, 256)
        p.setPen(QPen(QColor(CHANNEL_COLORS[self.active_channel]), 2))
        path_pts = [self._to_screen(i / 255, v) for i, v in enumerate(vals)]
        for i in range(len(path_pts) - 1):
            p.drawLine(path_pts[i], path_pts[i + 1])

        # Punkte
        # Handles like Krita's curve widget: hollow rings, bigger on hover, selected = filled
        # red (the current choice); a point dragged outside is crossed out (release deletes).
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pts = self.curves[self.active_channel]
        for i, pt in enumerate(pts):
            pos = self._to_screen(pt[0], pt[1])
            selected = i == self._sel_idx
            radius = 6.5 if (i == self._hover_idx or selected) else 5.0
            p.setPen(QPen(QColor(style.ACCENT_HI if selected else style.TEXT), 1.5))
            p.setBrush(QColor(style.ACCENT_HI) if selected else QColor(style.BLACK))
            p.drawEllipse(pos, radius, radius)
            if selected and self._doomed:
                p.setPen(QPen(QColor(style.ACCENT_HI), 1.5))
                p.drawLine(pos + QPointF(-8, -8), pos + QPointF(8, 8))
                p.drawLine(pos + QPointF(-8, 8), pos + QPointF(8, -8))

    # ── Maus ──────────────────────────────────────────────────────────────────

    def _hit_point(self, pos: QPointF) -> int | None:
        pts = self.curves[self.active_channel]
        cyclic = self._is_cyclic(self.active_channel)
        for i, pt in enumerate(pts):
            screen = self._to_screen(pt[0], pt[1])
            if math.hypot(pos.x() - screen.x(), pos.y() - screen.y()) <= 10:
                return i
            if cyclic:
                # Wrapped position (point near 0 also appears near 1)
                wx = (pt[0] + 1.0) % 1.0
                if abs(wx - pt[0]) > 1e-6:
                    screen_w = self._to_screen(wx, pt[1])
                    if math.hypot(pos.x() - screen_w.x(), pos.y() - screen_w.y()) <= 10:
                        return i
                wx2 = (pt[0] - 1.0) % 1.0
                if abs(wx2 - pt[0]) > 1e-6:
                    screen_w2 = self._to_screen(wx2, pt[1])
                    if math.hypot(pos.x() - screen_w2.x(), pos.y() - screen_w2.y()) <= 10:
                        return i
        return None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        pts = self.curves[self.active_channel]
        if e.button() == Qt.MouseButton.RightButton or \
           (e.button() == Qt.MouseButton.LeftButton and e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            idx = self._hit_point(e.position())
            if idx is not None and idx not in (0, len(pts) - 1):
                pts.pop(idx)
                self._emit()
            return

        if e.button() != Qt.MouseButton.LeftButton:
            return

        idx = self._hit_point(e.position())
        self.setFocus()
        if idx is not None:
            self._drag_idx = self._sel_idx = idx
            self._doomed = False
            self.drag_started.emit()
            self.update()
            return

        # Neuen Punkt setzen
        if len(pts) >= _MAX_POINTS:
            return
        x, y = self._to_data(e.position())
        if self._is_cyclic(self.active_channel):
            # Zyklisch: x muss nicht zwischen Nachbarn liegen, aber nicht zu nah
            if any(abs(x - pt[0]) < 0.005 for pt in pts):
                return
            insert_i = 0
            for i, pt in enumerate(pts):
                if pt[0] < x:
                    insert_i = i + 1
            pts.insert(insert_i, [x, y])
        else:
            insert_i = 0
            for i, pt in enumerate(pts):
                if pt[0] < x:
                    insert_i = i + 1
            if insert_i > 0 and pts[insert_i - 1][0] >= x:
                return
            if insert_i < len(pts) and pts[insert_i][0] <= x:
                return
            pts.insert(insert_i, [x, y])
        self._drag_idx = self._sel_idx = insert_i
        self._doomed = False
        self.drag_started.emit()
        self._emit()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        """Drag the grabbed point. y is always kept in 0..1 (the worker rejects anything else,
        which looked like 'handles do nothing'); x stays between the neighbours, on hue
        curves the list is re-sorted so the order never breaks. Live preview while dragging."""
        if self._drag_idx is None:
            hover = self._hit_point(e.position())
            if hover != self._hover_idx:
                self._hover_idx = hover
                self.update()
            return
        pts = self.curves[self.active_channel]
        idx = self._drag_idx
        pos = e.position()
        outside = (pos.x() < -12 or pos.y() < -12 or pos.x() > self.width() + 12
                   or pos.y() > self.height() + 12)
        endpoint = idx in (0, len(pts) - 1) and not self._is_cyclic(self.active_channel)
        if self._doomed != (outside and not endpoint and len(pts) > 2):
            self._doomed = outside and not endpoint and len(pts) > 2
            self.update()
        if self._doomed:
            return
        x, y = self._to_data(pos)
        y = max(0.0, min(1.0, y))
        if idx == 0 or idx == len(pts) - 1:
            if not self._is_cyclic(self.active_channel):
                pts[idx][1] = y                      # end points: y only
                self.update()
                self._emit()
                return
        if self._is_cyclic(self.active_channel):
            x = max(0.0, min(1.0, x))
            if any(j != idx and abs(x - pt[0]) < 0.005 for j, pt in enumerate(pts)):
                return
            moved = [x, y]
            pts[idx] = moved
            pts.sort(key=lambda pt: pt[0])
            self._drag_idx = next(k for k, pt in enumerate(pts) if pt is moved)
        else:
            x = max(pts[idx - 1][0] + 1e-4, min(pts[idx + 1][0] - 1e-4, x))
            pts[idx][0], pts[idx][1] = x, y
        self.update()
        self._emit()

    def mouseReleaseEvent(self, _e) -> None:  # noqa: N802
        if self._drag_idx is None:
            return
        if self._doomed:                          # Krita: drag a point out to delete it
            self.curves[self.active_channel].pop(self._drag_idx)
            self._sel_idx = None
        self._drag_idx = None
        self._doomed = False
        self.drag_stopped.emit()
        self._emit()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        if self._hover_idx is not None:
            self._hover_idx = None
            self.update()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        """Del/Backspace deletes the selected point; arrows nudge it (Shift = ×10)."""
        pts = self.curves[self.active_channel]
        i = self._sel_idx
        if i is None or i >= len(pts):
            return super().keyPressEvent(e)
        endpoint = i in (0, len(pts) - 1) and not self._is_cyclic(self.active_channel)
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if not endpoint and len(pts) > 2:
                pts.pop(i)
                self._sel_idx = None
                self._emit()
            return
        step = (10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1) / 255
        dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(e.key(), 0.0)
        dy = {Qt.Key.Key_Down: -step, Qt.Key.Key_Up: step}.get(e.key(), 0.0)
        if not (dx or dy):
            return super().keyPressEvent(e)
        x, y = pts[i]
        y = max(0.0, min(1.0, y + dy))
        if not endpoint and dx:
            lo = pts[i - 1][0] + 1e-4 if i > 0 else 0.0
            hi = pts[i + 1][0] - 1e-4 if i < len(pts) - 1 else 1.0
            x = max(lo, min(hi, x + dx))
        pts[i] = [x, y]
        self._emit()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        pts = self.curves[self.active_channel]
        idx = self._hit_point(e.position())
        if idx is not None and idx not in (0, len(pts) - 1):
            pts.pop(idx)
            self._emit()

    def add_point_at(self, x: float, y: float = 0.5) -> bool:
        """Punkt bei x (y=Neutral) einfügen; False wenn zu nah an bestehendem."""
        pts = self.curves[self.active_channel]
        if len(pts) >= _MAX_POINTS:
            return False
        if any(abs(x - pt[0]) < 0.005 for pt in pts):
            return False
        insert_i = 0
        for i, pt in enumerate(pts):
            if pt[0] < x:
                insert_i = i + 1
        pts.insert(insert_i, [x, y])
        self._emit()
        return True

    def _emit(self) -> None:
        self.update()
        self.changed.emit(self.active_channel, self.get_curve(self.active_channel))

    def resizeEvent(self, event) -> None:  # noqa: N802
        # Box layouts ignore minimumHeightForWidth; a real minimum height is the only way the
        # column keeps 2:1 instead of squashing the curve (user 2026-10-11: "1:2 Mindesthöhe")
        super().resizeEvent(event)
        want = max(1, self.width() // 2)
        if self.minimumHeight() != want:
            self.setMinimumHeight(want)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return max(1, w // 2)  # bevorzugt 2:1 (Breite:Höhe)
