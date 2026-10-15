"""Colours, font and stylesheet of the REF Grade window — the only place they are defined."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QWidget

from . import scale

BLACK = "#000000"
TEXT = "#EDEDED"
MUTED = "#8A8A8A"
LINE = "#1A1A1A"
LINE_STRONG = "#2A2A2A"
DIM = "#3A3A3A"           # text and controls of a section that is OFF
ACCENT = "#6E6E6E"       # grey accent: ON button, OK
ACCENT_HI = "#B4B4B4"    # grey accent (bright): text/border for checked, hover, selected
ACCENT_HOVER = "#828282" # OK button hover

# Superslim colour tracks (user 2026-10-07: "Farbanzeige superslim"): the 2 px slider groove
# is a gradient that shows where the slider pushes. Stops are 0..1 over the slider range.
TRACKS = {
    "exposure": ((0.0, "#000000"), (1.0, "#FFFFFF")),
    "temperature": ((0.0, "#2F6BFF"), (0.5, "#5A5A5A"), (1.0, "#FFAE2E")),
    "tint": ((0.0, "#2EFF7A"), (0.5, "#5A5A5A"), (1.0, "#FF2EE6")),
    "saturation": ((0.0, "#5A5A5A"), (1 / 3, "#6E655C"), (1.0, "#FF6A00")),
    "vibrance": ((0.0, "#5A5A5A"), (0.5, "#6E655C"), (1.0, "#FF6A00")),
}


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
    """Re-apply fonts and stylesheet to the widget tree after a scale change."""
    theme_name = getattr(root, "_theme_name", None)
    if theme_name:
        from . import themes
        themes.apply_theme(root, theme_name, base_sheet=stylesheet())
    else:
        root.setStyleSheet(stylesheet())
    for w in root.findChildren(QWidget):
        role = w.property("font_role")
        if role is None:
            continue
        weight = w.property("font_weight") or QFont.Weight.Normal
        w.setFont(font(role, weight))


def stylesheet(s: float | None = None) -> str:
    """Full window stylesheet; tooltip font-size derives from the scale unit."""
    if s is None:
        s = _current_s
    tip_px = scale.base_px(s)
    return f"""
QDialog, QWidget {{ background: {BLACK}; color: {TEXT}; }}
QLabel {{ color: {TEXT}; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[section="true"] {{ color: {MUTED}; letter-spacing: 2px; }}
QLabel[section="true"][state="on"] {{ color: {TEXT}; }}
QLabel[section="true"][state="off"] {{ color: {DIM}; }}
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
QPushButton[main="ok"] {{ background: {ACCENT}; color: #FFFFFF; border: 1px solid {ACCENT}; font-weight: bold; }}
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


# ── Overlay states (K27f) ─────────────────────────────────────────────────────

def overlay_style(state: str) -> tuple[int, str]:
    """Return (alpha, hex_color) for the given overlay state.

    States: idle, hover, busy, done, error.
    """
    table = {
        "idle": (60, "#5A5A5A"),
        "hover": (140, "#5A5A5A"),
        "busy": (200, "#8A8A8A"),
        "done": (200, "#B4B4B4"),
        "error": (200, "#A04040"),
    }
    return table.get(state, table["idle"])
