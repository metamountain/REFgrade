"""Windows title-bar / frame theming for a top-level widget (ctypes only).

On Windows 10/11 the native title bar and window frame follow the OS theme, not the
app. This module pushes the active colour scheme onto that chrome via
``DwmSetWindowAttribute``:

* ``DWMWA_USE_IMMERSIVE_DARK_MODE`` (20) — dark title bar for dark schemes.
* ``DWMWA_BORDER_COLOR`` (34), ``DWMWA_CAPTION_COLOR`` (35), ``DWMWA_TEXT_COLOR`` (36)
  — Windows 11 frame/caption/title text, set to the scheme's ``bg``/``text``.

Everything is best-effort: on non-Windows, on older Windows, or when the widget has
no native window yet, the call is a silent no-op. Only ``ctypes`` + the stdlib are
used, so this module is safe to import from both the plugin and the standalone GUI.
"""

from __future__ import annotations

import sys

# DWM window-attribute indices (winuser.h / dwmapi.h).
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_BORDER_COLOR = 34
DWMWA_CAPTION_COLOR = 35
DWMWA_TEXT_COLOR = 36

# Schemes whose title bar should use the immersive dark mode.
_DARK_SCHEMES = ("Black", "Dark Grey")


def _colorref(hexcolor: str) -> int:
    """``#rrggbb`` → Windows COLORREF ``0x00BBGGRR``."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def apply(widget, scheme) -> None:
    """Theme the native title bar / frame of *widget* to *scheme* (no-op off-Windows)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import byref, c_int

        win_id = widget.winId()
        if not win_id:
            return
        hwnd = int(win_id)
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi

        def _set(attr: int, value: int) -> None:
            dwmapi.DwmSetWindowAttribute(hwnd, c_int(attr), byref(c_int(value)), 4)

        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if scheme.name in _DARK_SCHEMES else 0)
        # Windows 11 frame + caption + title text (ignored by older Windows).
        _set(DWMWA_BORDER_COLOR, _colorref(scheme.bg))
        _set(DWMWA_CAPTION_COLOR, _colorref(scheme.bg))
        _set(DWMWA_TEXT_COLOR, _colorref(scheme.text))
    except Exception:
        # Older Windows / no DWM / no native window yet: leave the chrome alone.
        return
