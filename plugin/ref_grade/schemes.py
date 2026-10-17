"""REFgrade UI colour schemes — the single source of truth for the 5 chrome colours.

A scheme is exactly five colours (``bg``, ``surface``, ``text``, ``accent``, ``dim``).
The window stylesheet (``style.stylesheet``) and the widget QPalette are both derived
from these, so switching scheme re-tints the whole UI.

Functional/semantic colours are NOT part of a scheme and stay fixed: the slider
gradient tracks (``style.TRACKS``), the curve channel colours, the overlay states
(``style.overlay_style``) and colour-replace swatches. They live as named constants
in the ``# functional colours`` section of ``style.py``.

The nine legacy chrome constants (``style.BLACK``, ``style.TEXT``, ...) are resolved
onto these five colours via :func:`role_color`, so existing ``style.X`` references
keep working and stay scheme-aware.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorScheme:
    name: str
    bg: str        # main background
    surface: str   # panel / section background, borders
    text: str      # primary text
    accent: str    # sliders, buttons, active / checked elements
    dim: str       # secondary text, inactive elements


SCHEMES: dict[str, ColorScheme] = {
    "Black":      ColorScheme("Black",      "#0a0a0a", "#1a1a1a", "#e0e0e0", "#4a9eff", "#666666"),
    "Dark Grey":  ColorScheme("Dark Grey",  "#1e1e1e", "#2c2c2c", "#d6d6d6", "#4a9eff", "#8a8a8a"),
    "Light Grey": ColorScheme("Light Grey", "#c8c8c8", "#d8d8d8", "#1a1a1a", "#2563eb", "#5a5a5a"),
    "White":      ColorScheme("White",      "#f7f7f7", "#ffffff", "#1a1a1a", "#2563eb", "#8c8c8c"),
}

DEFAULT = "Black"

# Old scheme names (pre-K41a) → their successor, so saved settings keep working.
_LEGACY = {"Carbon": "Black", "Anthrazite": "Dark Grey", "Paper": "White"}

# Legacy chrome role -> the one of the 5 scheme colours it maps onto.
_ROLE = {
    "BLACK": "bg",
    "TEXT": "text",
    "MUTED": "dim",
    "LINE": "surface",
    "LINE_STRONG": "surface",
    "DIM": "dim",
    "ACCENT": "accent",
    "ACCENT_HI": "accent",
    "ACCENT_HOVER": "accent",
}

_current: ColorScheme = SCHEMES[DEFAULT]


def resolve_name(name: str) -> str:
    """Map a (possibly legacy or unknown) scheme name onto a current one; unknown → DEFAULT."""
    name = _LEGACY.get(name, name)
    return name if name in SCHEMES else DEFAULT


def current_scheme() -> ColorScheme:
    return _current


def current_scheme_name() -> str:
    return _current.name


def set_current(name: str) -> ColorScheme:
    """Set the active scheme by name (legacy/unknown names fall back); return it."""
    global _current
    _current = SCHEMES[resolve_name(name)]
    return _current


def role_color(role: str) -> str:
    """Resolve a legacy chrome role (e.g. ``'TEXT'``) to the active scheme's colour."""
    return getattr(_current, _ROLE.get(role, "dim"))


def apply_scheme(widget, scheme: ColorScheme | None = None) -> None:
    """Apply a scheme to a top-level widget: QPalette + full window stylesheet."""
    global _current
    if scheme is None:
        scheme = _current
    _current = scheme

    from PyQt6.QtGui import QColor, QPalette
    from . import style

    pal = widget.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(scheme.bg))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(scheme.text))
    pal.setColor(QPalette.ColorRole.Base, QColor(scheme.surface))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(scheme.surface))
    pal.setColor(QPalette.ColorRole.Text, QColor(scheme.text))
    pal.setColor(QPalette.ColorRole.Button, QColor(scheme.surface))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(scheme.text))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(scheme.accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(scheme.bg))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(scheme.surface))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(scheme.text))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(scheme.dim))
    widget.setPalette(pal)
    widget.setStyleSheet(style.stylesheet())
