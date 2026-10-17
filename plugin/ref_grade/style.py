"""Colours, font and stylesheet of the REFgrade window — the only place they are defined.

Chrome colours (background, surface, text, accent, dim) come from the active colour
scheme (``schemes``). Functional / data-visualisation colours (slider tracks, curve
channel colours, scopes, colour-replace swatches, overlay states) are fixed and live
in the ``# functional colours`` section below — the single place they are defined.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget

from . import scale, schemes

# The nine legacy chrome constants (BLACK, TEXT, MUTED, LINE, LINE_STRONG, DIM,
# ACCENT, ACCENT_HI, ACCENT_HOVER) are no longer fixed values: they resolve live to
# the active colour scheme (see schemes.role_color), so every ``style.X`` reference
# and the window stylesheet re-tint when the scheme changes.
_CHROME_ROLES = ("BLACK", "TEXT", "MUTED", "LINE", "LINE_STRONG",
                 "DIM", "ACCENT", "ACCENT_HI", "ACCENT_HOVER")


def __getattr__(name):  # noqa: N807 - module-level live chrome colours
    if name in _CHROME_ROLES:
        return schemes.role_color(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── functional colours ─────────────────────────────────────────────────────────
# Fixed regardless of the active scheme: data-visualisation and functional colours.
# This is the single place they are defined (spec K41a §2: "nicht verstreut").

# Status-strip error mark (K27e: grey, not red).
MARK_ERROR = "#B4B4B4"

# Superslim colour tracks (user 2026-10-07: "Farbanzeige superslim"): the 2 px slider groove
# is a gradient that shows where the slider pushes. Stops are 0..1 over the slider range.
TRACKS = {
    "exposure": ((0.0, "#000000"), (1.0, "#FFFFFF")),
    "temperature": ((0.0, "#2F6BFF"), (0.5, "#5A5A5A"), (1.0, "#FFAE2E")),
    "tint": ((0.0, "#2EFF7A"), (0.5, "#5A5A5A"), (1.0, "#FF2EE6")),
    "saturation": ((0.0, "#5A5A5A"), (1 / 3, "#6E655C"), (1.0, "#FF6A00")),
    "vibrance": ((0.0, "#5A5A5A"), (0.5, "#6E655C"), (1.0, "#FF6A00")),
}

# Overlay states (K27f): (alpha, hex_color) for idle / hover / busy / done / error.
OVERLAY = {
    "idle": (60, "#5A5A5A"),
    "hover": (140, "#5A5A5A"),
    "busy": (200, "#8A8A8A"),
    "done": (200, "#B4B4B4"),
    "error": (200, "#A04040"),
}


def overlay_style(state: str) -> tuple[int, str]:
    """Return (alpha, hex_color) for the given overlay state (idle, hover, busy, done, error)."""
    return OVERLAY.get(state, OVERLAY["idle"])


# ── Reference tiles (RefSlots) ─────────────────────────────────────────────────
TILE_SCRIM = QColor(0, 0, 0, 170)          # label background strip
TILE_PAIR_LABEL = QColor(255, 255, 255, 160)  # pair label text
TILE_DIM = QColor(0, 0, 0, 120)            # dimmed tile overlay

# ── ColorWheel ─────────────────────────────────────────────────────────────────
WHEEL_FADE_CENTER = QColor(20, 20, 20, 255)  # dark centre (muted, black-first look)
WHEEL_FADE_EDGE = QColor(0, 0, 0, 90)
WHEEL_DISABLED = QColor(0, 0, 0, 204)        # section off: 80 % black over the disk

# ── Histogram ──────────────────────────────────────────────────────────────────
HIST_BG = QColor(20, 20, 20)
HIST_GRID = QColor(40, 40, 40)
HIST_SOURCE = QColor(160, 160, 160)          # source grey outline
HIST_CLIP = QColor(200, 200, 200)            # clipping triangles
HIST_GHOST = QColor(255, 255, 255)           # luminance ghost / haze
HIST_CHANNEL_COLORS = {
    "r": QColor(220, 40, 40),
    "g": QColor(40, 200, 40),
    "b": QColor(40, 80, 220),
    "l": QColor(180, 180, 180),
}

# ── Scope channel buttons ──────────────────────────────────────────────────────
SCOPE_CH_COLORS = {
    "r": QColor(0xFF, 0x3B, 0x3B),
    "g": QColor(0x3B, 0xFF, 0x6A),
    "b": QColor(0x3B, 0x8B, 0xFF),
    "l": QColor(0xED, 0xED, 0xED),
}
SCOPE_L_OFF = QColor(0x5A, 0x5A, 0x5A)       # L off: dark grey, on: white (user 2026-10-11)
SCOPE_ACTIVE_TEXT = QColor(0, 0, 0)          # text on an active (filled) scope button

# ── Image view overlays ────────────────────────────────────────────────────────
SPLIT_LINE_SHADOW = QColor(0, 0, 0)          # 1 px black shadow behind the split line
SPLIT_LINE = QColor(255, 255, 255)           # the split line itself
LOAD_BAR_BG = QColor(0, 0, 0, 90)            # loading-bar background
LOAD_TEXT_SHADOW = QColor(0, 0, 0)           # loading-% text shadow
STATUS_STRIP_BG = QColor(16, 16, 16)         # status strip at the bottom of the image
TOGGLE_OFF_BG = QColor(0, 0, 0, 160)         # S / C toggle, off state

# ── Vectorscope ────────────────────────────────────────────────────────────────
VS_BG = QColor(0, 0, 0)
VS_CROSSHAIR = QColor(40, 40, 40)
VS_SKIN_LINE = QColor(100, 100, 100)
# Rec.709 Cb/Cr für reine Primärfarbton bei 75 % Sättigung.
VS_MARKS = (
    ("R", 0.5 - 0.1144, 0.5 + 0.5, QColor(220, 40, 40)),
    ("Y", 0.5 - 0.5, 0.5 + 0.0458, QColor(220, 220, 40)),
    ("G", 0.5 - 0.3853, 0.5 - 0.4538, QColor(40, 200, 40)),
    ("C", 0.5 + 0.1144, 0.5 - 0.5, QColor(40, 200, 220)),
    ("B", 0.5 + 0.5, 0.5 - 0.0458, QColor(40, 80, 220)),
    ("M", 0.5 + 0.3853, 0.5 + 0.4538, QColor(220, 40, 220)),
)

# ── Waveform (luma) ────────────────────────────────────────────────────────────
WAVE_BG = QColor(0, 0, 0)
WAVE_GRID = QColor(38, 38, 38)

# ── RGB Parade ─────────────────────────────────────────────────────────────────
PARADE_BG = QColor(0, 0, 0)
PARADE_CHANNEL_COLORS = {
    "r": QColor(0xFF, 0x30, 0x30),
    "g": QColor(0x30, 0xFF, 0x60),
    "b": QColor(0x40, 0x60, 0xFF),
}

# ── Curves editor ──────────────────────────────────────────────────────────────
# Channel colours (the "rgb" entry is the scheme text colour, resolved live).
CURVE_CHANNEL_COLORS = {
    "r": "#B43C3C",
    "g": "#3CB43C",
    "b": "#3C3CB4",
    "hue_hue": "#CC8833",
    "hue_sat": "#CC3388",
    "hue_lum": "#3388CC",
    "lum_sat": "#33CC88",
    "sat_sat": "#8833CC",
}
CURVE_LUM_SAT = (QColor(0, 0, 0), QColor(255, 255, 255))   # lum_sat background gradient
CURVE_SAT_SAT = (QColor(128, 128, 128), QColor(255, 0, 0))  # sat_sat background gradient
CURVE_HISTOGRAM = QColor(255, 255, 255, 26)                 # histogram bars behind the curve

# ── Color Replace ──────────────────────────────────────────────────────────────
CR_BAR_BLACK = QColor(0, 0, 0)            # BrightBar gradient, black end
CR_BAR_WHITE = QColor(255, 255, 255)      # BrightBar gradient, white end
CR_HANDLE_EDGE = QColor(0, 0, 0)          # BrightBar handle edge
CR_SQUARE_DEFAULT = "#808080"             # default src/tgt swatch colour
CR_SQUARE_BORDER = "#2A2A2A"              # swatch border
CR_RESET_BG = QColor(0, 0, 0, 150)        # wheel reset button background
CR_RESET_HOVER = QColor(255, 255, 255)    # reset glyph, hover
CR_RESET_IDLE = QColor(0xA0, 0xA0, 0xA0)  # reset glyph, idle
# Curve-editor channel colours (the "rgb" master curve uses the scheme text colour instead).
CHANNEL_COLORS = {
    "r": "#B43C3C",
    "g": "#3CB43C",
    "b": "#3C3CB4",
    "hue_hue": "#CC8833",
    "hue_sat": "#CC3388",
    "hue_lum": "#3388CC",
    "lum_sat": "#33CC88",
    "sat_sat": "#8833CC",
}

# Neutral ramps / swatch constants for gradients and colour-replace swatches.
RAMP_BLACK = "#000000"
RAMP_WHITE = "#FFFFFF"
RAMP_GREY = "#808080"
RAMP_RED = "#FF0000"
TRACK_NEUTRAL = "#5A5A5A"      # middle stop of the desaturated slider tracks
SWATCH_EDGE = "#2A2A2A"

# ── end functional colours ─────────────────────────────────────────────────────


def track_sheet(stops) -> str:
    """Stylesheet for a QSlider whose groove is the gradient `stops` (dimmed when disabled)."""
    def grad(alpha: int) -> str:
        parts = []
        for pos, hexcol in stops:
            r, g, b = int(hexcol[1:3], 16), int(hexcol[3:5], 16), int(hexcol[5:7], 16)
            parts.append(f"stop:{pos:.4f} rgba({r},{g},{b},{alpha})")
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(parts)})"
    return (f"QSlider::groove:horizontal {{ height: 2px; background: {grad(255)}; }}"
            f"QSlider::groove:horizontal:disabled {{ background: {grad(70)}; }}"
            "QSlider::sub-page:horizontal { background: transparent; }")


GAP = 8                  # px between panels
SPLIT = 32               # px between control column and image (user: more space)
EDGE = 12                # px left screen edge → control column (user 2026-10-07)
FONT_FAMILY = "Inter"
_FONT_FILES = (Path(__file__).parent / "fonts",
               Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts")


def load_font() -> str:
    """Make Inter available (bundled file first, then the user's font folder); return the family."""
    if FONT_FAMILY in QFontDatabase.families():
        return FONT_FAMILY
    for folder in _FONT_FILES:
        for f in sorted(folder.glob("Inter-VariableFont*.ttf")) if folder.is_dir() else ():
            fid = QFontDatabase.addApplicationFont(str(f))
            if fid >= 0:
                return QFontDatabase.applicationFontFamilies(fid)[0]
    return "Segoe UI"


# ── Responsive scale ──────────────────────────────────────────────────────────
_current_s: float = scale.unit(1920, 1032)


def set_scale(s: float) -> None:
    """Set the current scale unit (call from resizeEvent)."""
    global _current_s
    _current_s = s


def current_s() -> float:
    """Return the current scale unit."""
    return _current_s


def font(role: str = "base", weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Inter at the pixel size for `role` with tabular figures for numbers."""
    px = scale.role_px(role, _current_s)
    f = QFont(load_font(), px, weight)
    f.setPixelSize(px)
    f.setFeature(QFont.Tag("tnum"), 1)
    return f


def apply_fonts(root: QWidget) -> None:
    """Re-apply fonts and (scheme-driven) stylesheet to the widget tree after a scale change."""
    root.setStyleSheet(stylesheet())
    for w in root.findChildren(QWidget):
        role = w.property("font_role")
        if role is None:
            continue
        weight = w.property("font_weight") or QFont.Weight.Normal
        w.setFont(font(role, weight))


def stylesheet(s: float | None = None) -> str:
    """Full window stylesheet; tooltip font-size derives from the scale unit.

    The chrome colours are resolved from the active colour scheme at call time, so
    re-calling this after a scheme change yields a re-tinted sheet.
    """
    if s is None:
        s = _current_s
    tip_px = scale.base_px(s)
    BLACK = schemes.role_color("BLACK")
    TEXT = schemes.role_color("TEXT")
    MUTED = schemes.role_color("MUTED")
    LINE = schemes.role_color("LINE")
    LINE_STRONG = schemes.role_color("LINE_STRONG")
    DIM = schemes.role_color("DIM")
    ACCENT = schemes.role_color("ACCENT")
    ACCENT_HI = schemes.role_color("ACCENT_HI")
    ACCENT_HOVER = schemes.role_color("ACCENT_HOVER")
    return f"""
QDialog, QWidget {{ background: {BLACK}; color: {TEXT}; }}
QLabel {{ color: {TEXT}; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[section="true"] {{ color: {MUTED}; letter-spacing: 2px; }}
QLabel[section="true"][state="on"] {{ color: {TEXT}; }}
QLabel[section="true"][state="off"] {{ color: {DIM}; }}
QLabel[wordmark="ref"] {{ color: {ACCENT}; }}
QLabel[wordmark="grade"] {{ color: {TEXT}; }}
QLabel[wordmark="ver"] {{ color: {DIM}; }}
QPushButton {{ background: {BLACK}; color: {TEXT}; border: 1px solid {LINE_STRONG};
              padding: 6px 12px; }}
QPushButton:hover {{ border-color: {MUTED}; }}
QPushButton:checked, QPushButton[accent="true"] {{ color: {ACCENT_HI}; border-color: {ACCENT_HI}; }}
QPushButton[small="true"] {{ padding: 0px 5px; border-color: {LINE}; color: {MUTED}; }}
QPushButton[small="true"]:hover {{ color: {TEXT}; border-color: {LINE_STRONG}; }}
QPushButton[small="true"]:checked {{ color: {ACCENT_HI}; border-color: {ACCENT_HI}; }}
QPushButton[role="on"]:checked {{ color: {TEXT}; background: {ACCENT}; border-color: {ACCENT}; }}
QPushButton[role="on"]:!checked {{ color: {MUTED}; background: {BLACK}; border-color: {LINE_STRONG}; }}
QPushButton[dim="true"] {{ color: {DIM}; border-color: {LINE}; }}
QPushButton[help="true"] {{ padding: 0px; border-color: {LINE}; color: {MUTED}; }}
QPushButton[help="true"]:hover {{ color: {TEXT}; border-color: {LINE_STRONG}; }}
QPushButton[help="true"][open="true"] {{ border-color: {ACCENT_HI}; }}
QPushButton[main="reset"] {{ background: {BLACK}; color: {MUTED}; border: 1px solid {LINE}; }}
QPushButton[main="reset"]:hover {{ border-color: {TEXT}; }}
QPushButton[main="cancel"] {{ background: {BLACK}; color: {TEXT}; border: 1px solid {LINE_STRONG}; }}
QPushButton[main="cancel"]:hover {{ border-color: {TEXT}; }}
QPushButton[main="ok"] {{ background: {ACCENT}; color: {TEXT}; border: 1px solid {ACCENT}; font-weight: bold; }}
QPushButton[main="ok"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QLabel:disabled {{ color: {DIM}; }}
QLabel[dim="true"] {{ color: {DIM}; }}
QSlider::groove:horizontal:disabled {{ background: {LINE}; }}
QSlider::sub-page:horizontal:disabled {{ background: {LINE_STRONG}; }}
QSlider::handle:horizontal:disabled {{ background: {DIM}; }}
QSlider[dim="true"]::groove:horizontal {{ background: {LINE}; }}
QSlider[dim="true"]::sub-page:horizontal {{ background: {LINE_STRONG}; }}
QSlider[dim="true"]::handle:horizontal {{ background: {DIM}; }}
QSlider::groove:horizontal {{ height: 2px; background: {LINE_STRONG}; }}
QSlider::sub-page:horizontal {{ background: {MUTED}; }}
QSlider::handle:horizontal {{ width: 10px; height: 10px; margin: -4px 0; background: {TEXT};
                             border-radius: 5px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HI}; }}
QSplitter::handle {{ background: {BLACK}; }}
QComboBox {{ background: {BLACK}; color: {TEXT}; border: 1px solid {LINE_STRONG};
            padding: 2px 6px; }}
QComboBox[dim="true"] {{ color: {DIM}; border-color: {LINE}; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{ background: {BLACK}; color: {MUTED};
                               border: 1px solid {LINE_STRONG}; }}
QComboBox QAbstractItemView::item {{ color: {MUTED}; }}
QComboBox QAbstractItemView::item:hover {{ background: {LINE_STRONG}; color: {TEXT}; }}
QComboBox QAbstractItemView::item:selected {{ color: {ACCENT_HI}; background: {BLACK}; }}
QToolTip {{ background: {BLACK}; color: {TEXT}; border: 1px solid {MUTED};
            font-family: "Inter"; font-size: {tip_px}px; max-width: 320px; }}
"""


# Backward-compatible module-level constant (uses default scale).
STYLESHEET: str = stylesheet()
