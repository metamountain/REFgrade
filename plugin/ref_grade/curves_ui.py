"""Kurven-Editor-Widget: Gitter 4×4, Kurven, Punkte setzen/ziehen/löschen.

Kein krita-Import. Die PCHIP-Mathematik steht in pchip.py.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from . import schemes, style
from .pchip import pchip_curve, pchip_cyclic_curve

CHANNELS = ("rgb", "r", "g", "b")
HUE_CHANNELS = ("hue_hue", "hue_sat", "hue_lum", "lum_sat", "sat_sat")
ALL_CHANNELS = CHANNELS + HUE_CHANNELS
CHANNEL_COLORS = {"rgb": style.TEXT, **style.CHANNEL_COLORS}


def _sc(role: str, alpha: int = 255) -> QColor:
    """Active-scheme colour (``bg``/``surface``/``text``/``accent``/``dim``) with alpha."""
    c = QColor(getattr(schemes.current_scheme(), role))
    c.setAlpha(alpha)
    return c
_HUE_CYCLIC = ("hue_hue", "hue_sat", "hue_lum")
_MAX_POINTS = 16
_IDENTITY = [[0.0, 0.0], [1.0, 1.0]]
_HUE_IDENTITY = [[0.0, 0.5], [1.0, 0.5]]

# Spelled-out channel names for the top-left overlay (user 2026-10-16: "title ausgeschrieben").
CHANNEL_TITLES = {
    "rgb": "RGB",
    "r": "Red",
    "g": "Green",
    "b": "Blue",
    "hue_hue": "Hue vs Hue",
    "hue_sat": "Hue vs Saturation",
    "hue_lum": "Hue vs Lightness",
    "lum_sat": "Lightness vs Saturation",
    "sat_sat": "Saturation vs Saturation",
}

# Detailed per-channel help for the bottom-right overlay (all English).
CHANNEL_HELP = {
    "rgb": ("Brightness curve for all three channels together. Across: input brightness, "
            "left = shadows, right = highlights. Up: brighter, down: darker. An S-shape adds "
            "contrast. The flat diagonal means no change."),
    "r": ("Red channel curve. Across: brightness. Up adds red, down adds cyan in that "
          "brightness range. Lift the lower end for warm shadows, the upper end for warm "
          "highlights."),
    "g": ("Green channel curve. Across: brightness. Up adds green, down adds magenta in "
          "that brightness range."),
    "b": ("Blue channel curve. Across: brightness. Up adds blue, down adds yellow in that "
          "brightness range. Lift the upper end for cool highlights (the classic "
          "teal-and-orange look)."),
    "hue_hue": ("Hue vs Hue. Across: the colour of a pixel (red to yellow, green, cyan, "
                "blue, magenta, back to red). Up/down shifts that colour to a neighbouring "
                "hue, e.g. turn orange skin more red or green grass more yellow. The flat "
                "middle line means no change."),
    "hue_sat": ("Hue vs Saturation. Across: the colour of a pixel. Up: that colour gets more "
                "intense; down: it fades toward grey. E.g. calm a loud blue sky while leaving "
                "reds vivid. The flat middle line means no change."),
    "hue_lum": ("Hue vs Lightness. Across: the colour of a pixel. Up: that colour gets "
                "brighter; down: darker. E.g. darken a blue sky or brighten foliage. The flat "
                "middle line means no change."),
    "lum_sat": ("Lightness vs Saturation. Across: dark to bright pixels. Up: more colour "
                "there; down: less. E.g. take the colour out of the deep shadows. The flat "
                "middle line means no change."),
    "sat_sat": ("Saturation vs Saturation. Across: grey to fully coloured pixels. Up: boost "
                "them; down: reduce them. E.g. lift muted colours without pushing loud ones "
                "further. The flat middle line means no change."),
}


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
        self._hover_help = False                  # mouse over editor → show help overlay
        self._help_pinned = False                 # K40a: "?" glyph toggles the help panel permanently
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

    def reset_all(self) -> None:
        """Reset every channel to identity and emit changed (active channel)."""
        for ch in ALL_CHANNELS:
            self.curves[ch] = self._identity_for(ch)
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
            c0 = QColor(style.RAMP_BLACK); c0.setAlphaF(0.25)
            c1 = QColor(style.RAMP_WHITE); c1.setAlphaF(0.25)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, c1)
            p.fillRect(r, grad)
        elif ch == "sat_sat":
            # Grau → gesättigt, 25 %
            grad = QLinearGradient(r.left(), 0, r.right(), 0)
            c0 = QColor(style.RAMP_GREY); c0.setAlphaF(0.25)
            c1 = QColor(style.RAMP_RED); c1.setAlphaF(0.25)
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

    # ── Overlay (title top-left, help bottom-right) ───────────────────────────

    def _reset_rect(self, r: QRectF) -> QRectF:
        """Bottom-right hit rect of the per-curve reset glyph."""
        s = 22.0
        return QRectF(r.right() - s - 6, r.bottom() - s - 6, s, s)

    def _help_rect(self, r: QRectF) -> QRectF:
        """K40a: hit rect of the help glyph, same size as the reset glyph, left of it."""
        s = 22.0
        rr = self._reset_rect(r)
        return QRectF(rr.x() - s - 6, rr.y(), s, s)

    def _background_color_at(self, fx: float) -> QColor:
        """Background colour at horizontal fraction fx (0=left, 1=right) as drawn by
        _draw_background (gradient blended 25 % over BLACK)."""
        base = QColor(style.BLACK)
        ch = self.active_channel
        if ch in _HUE_CYCLIC:
            fg = QColor.fromHsvF(max(0.0, min(1.0, fx)), 1.0, 1.0)
        elif ch == "lum_sat":
            v = int(max(0.0, min(1.0, fx)) * 255)
            fg = QColor(v, v, v)
        elif ch == "sat_sat":
            t = max(0.0, min(1.0, fx))
            fg = QColor(int(128 + t * 127), int(128 - t * 128), int(128 - t * 128))
        else:
            return base
        fg.setAlphaF(0.25)
        return self._blend(fg, base)

    @staticmethod
    def _blend(fg: QColor, bg: QColor) -> QColor:
        a = fg.alphaF()
        return QColor(int(fg.red() * a + bg.red() * (1 - a)),
                      int(fg.green() * a + bg.green() * (1 - a)),
                      int(fg.blue() * a + bg.blue() * (1 - a)))

    @staticmethod
    def _luminance(c: QColor) -> float:
        def chan(v: int) -> float:
            x = v / 255
            return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
        return 0.2126 * chan(c.red()) + 0.7152 * chan(c.green()) + 0.0722 * chan(c.blue())

    def _draw_overlays(self, p: QPainter, r: QRectF) -> None:
        """Top-left: spelled-out channel title (always visible, black/white by the
        background behind it). Bottom-right: per-curve reset glyph (always) and, on
        hover, a detailed help panel for the active channel (hidden while dragging)."""
        # ── Title (top-left, always) ──
        title = CHANNEL_TITLES.get(self.active_channel, self.active_channel)
        f_title = style.font("small", QFont.Weight.Bold)
        fm_title = QFontMetrics(f_title)
        p.setFont(f_title)
        pad_x, pad_y = 6, 3
        tw, th = fm_title.horizontalAdvance(title), fm_title.height()
        text_col = _sc("text")
        pill_col = _sc("surface", 200)
        tx, ty = r.x() + 6, r.y() + 6
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(pill_col)
        p.drawRoundedRect(QRectF(tx, ty, tw + 2 * pad_x, th + 2 * pad_y), 3, 3)
        p.setPen(QPen(text_col))
        p.drawText(QRectF(tx + pad_x, ty + pad_y, tw, th),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)

        # ── Per-curve reset glyph (bottom-right, always) ──
        rr = self._reset_rect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_sc("surface", 200))
        p.drawRoundedRect(rr, 4, 4)
        f_glyph = style.font("small", QFont.Weight.Bold)
        p.setFont(f_glyph)
        p.setPen(QPen(_sc("text")))
        p.drawText(rr, Qt.AlignmentFlag.AlignCenter, "\u21BA")

        # ── Help glyph (K40a: left of the reset glyph, same size and style) ──
        hr = self._help_rect(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_sc("surface", 200))
        p.drawRoundedRect(hr, 4, 4)
        p.setPen(QPen(_sc("text")))
        p.drawText(hr, Qt.AlignmentFlag.AlignCenter, "?")

        # ── Help panel (bottom-right, on hover or pinned, left of the help glyph) ──
        if not (self._hover_help or self._help_pinned) or self._drag_idx is not None:
            return
        help_text = CHANNEL_HELP.get(self.active_channel, "")
        if not help_text:
            return
        f_help = style.font("small")
        fm_help = QFontMetrics(f_help)
        p.setFont(f_help)
        pad = 8
        panel_w = min(int(r.width() * 0.6), 400)
        flags = (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                 | Qt.TextFlag.TextWordWrap)
        h = fm_help.boundingRect(QRect(0, 0, panel_w - 2 * pad, 1000), flags, help_text).height()
        panel_h = h + 2 * pad
        panel_x = hr.x() - panel_w - 6
        panel_y = r.bottom() - panel_h - 6
        help_text_col = _sc("text")
        help_pill_col = _sc("surface", 230)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(help_pill_col)
        p.drawRoundedRect(QRectF(panel_x, panel_y, panel_w, panel_h), 4, 4)
        p.setPen(QPen(help_text_col))
        p.drawText(QRectF(panel_x + pad, panel_y + pad, panel_w - 2 * pad, panel_h - 2 * pad),
                   flags, help_text)

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
                                  r.width() / n + 0.5, h), _sc("text", 26))

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

        # Overlays on top: title (top-left), reset glyph + help (bottom-right)
        self._draw_overlays(p, r)

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
        # K40a: help glyph (left of the reset glyph): left click pins/unpins the help panel.
        if e.button() == Qt.MouseButton.LeftButton and self._help_rect(self._plot_rect()).contains(e.position()):
            self._help_pinned = not self._help_pinned
            self.update()
            return
        # Per-curve reset glyph (bottom-right): left click resets the active channel only.
        if e.button() == Qt.MouseButton.LeftButton and self._reset_rect(self._plot_rect()).contains(e.position()):
            self.reset_channel()
            return
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

    def enterEvent(self, _e) -> None:  # noqa: N802
        self._hover_help = True
        self.update()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        if self._hover_idx is not None:
            self._hover_idx = None
        if self._hover_help:
            self._hover_help = False
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
