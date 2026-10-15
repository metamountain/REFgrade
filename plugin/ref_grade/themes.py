"""RefGrade UI Themes - Black, Anthrazite, White.

A theme is a colour palette that layers ON TOP of the base window stylesheet
(``style.STYLESHEET``). The base keeps its property-specific rules (section
headers, main buttons, combo box, tooltip, splitter, ...); the theme re-tints
the generic surface (background, text, buttons, sliders, checkboxes) and the
labels. ``theme_sheet`` returns the theme rules as a string so the dialog can
combine base + theme and re-apply both after a scale change (``style.apply_fonts``).
"""

from __future__ import annotations

THEMES = {
    "black": {
        "name": "Black",
        "bg": "#000000",
        "bg_alt": "#0A0A0A",
        "fg": "#FFFFFF",
        "fg_dim": "#888888",
        "accent": "#FFFFFF",
        "accent_dim": "#444444",
        "border": "#222222",
        "hover": "#1A1A1A",
        "selected": "#333333",
        "error": "#FF4444",
        "success": "#44FF44",
        "radius": "0px",
        "shadow": "none",
    },
    "anthrazite": {
        "name": "Anthrazite",
        "bg": "#1E1E24",
        "bg_alt": "#26262E",
        "fg": "#E8E8EC",
        "fg_dim": "#888890",
        "accent": "#7AA2F7",
        "accent_dim": "#3D5A8A",
        "border": "#33333B",
        "hover": "#2A2A34",
        "selected": "#3A3A46",
        "error": "#FF6B6B",
        "success": "#6BFF8B",
        "radius": "2px",
        "shadow": "none",
    },
    "white": {
        "name": "White",
        "bg": "#FFFFFF",
        "bg_alt": "#F5F5F5",
        "fg": "#111111",
        "fg_dim": "#666666",
        "accent": "#111111",
        "accent_dim": "#AAAAAA",
        "border": "#E0E0E0",
        "hover": "#F0F0F0",
        "selected": "#E8E8E8",
        "error": "#CC0000",
        "success": "#00AA00",
        "radius": "0px",
        "shadow": "none",
    },
}

# Cycle order for the theme switch button (black -> anthrazite -> white -> black).
ORDER = ("black", "anthrazite", "white")


def get_theme(name: str = "anthrazite") -> dict:
    """Get theme by name. Falls back to anthrazite."""
    return THEMES.get(name, THEMES["anthrazite"])


def theme_sheet(name: str = "anthrazite") -> str:
    """Stylesheet string for `name`; appended after the base window stylesheet.

    The QLabel rules override the base's hardcoded light label colours so the
    White theme keeps readable text; the more specific base rules (section
    state, muted, dim) still win where they are more specific.
    """
    t = get_theme(name)
    return f"""
QWidget {{
    background-color: {t['bg']};
    color: {t['fg']};
}}
QLabel {{
    color: {t['fg']};
}}
QLabel[muted="true"] {{
    color: {t['fg_dim']};
}}
QLabel[section="true"] {{
    color: {t['fg_dim']};
    letter-spacing: 2px;
}}
QLabel[section="true"][state="on"] {{
    color: {t['fg']};
}}
QLabel[section="true"][state="off"] {{
    color: {t['accent_dim']};
}}
QGroupBox {{
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
    margin-top: 8px;
    padding-top: 16px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}
QPushButton {{
    background-color: {t['bg_alt']};
    color: {t['fg']};
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
    padding: 4px 12px;
}}
QPushButton:hover {{
    background-color: {t['hover']};
    border-color: {t['accent_dim']};
}}
QPushButton:pressed {{
    background-color: {t['selected']};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {t['border']};
    border-radius: {t['radius']};
}}
QSlider::handle:horizontal {{
    width: 10px;
    height: 10px;
    margin: -3px 0;
    border-radius: 5px;
    background: {t['accent']};
}}
QSlider::handle:horizontal:hover {{
    background: {t['fg']};
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {t['border']};
    border-radius: {t['radius']};
    background: {t['bg_alt']};
}}
QCheckBox::indicator:checked {{
    background: {t['accent']};
    border-color: {t['accent']};
}}
"""


def apply_theme(widget, theme_name: str = "anthrazite", base_sheet: str | None = None) -> None:
    """Apply a theme to a QWidget.

    With ``base_sheet`` (the window's base stylesheet, e.g. ``style.stylesheet()``)
    the theme rules are appended after it, so the theme re-tints the generic
    widgets while the base's property-specific rules still win. Without it the
    widget's stylesheet is replaced by the theme alone.
    """
    sheet = theme_sheet(theme_name)
    if base_sheet:
        sheet = base_sheet + "\n" + sheet
    widget.setStyleSheet(sheet)


def theme_name_from_settings() -> str:
    """Read theme from QSettings, default 'anthrazite'."""
    try:
        from PyQt6.QtWidgets import QSettings
        s = QSettings("RefGrade", "RefGrade")
        return s.value("theme", "anthrazite", type=str)
    except Exception:
        return "anthrazite"
