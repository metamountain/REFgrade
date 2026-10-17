"""K41a: win_chrome.apply ruft DwmSetWindowAttribute mit 20/34/35/36 und richtigen COLORREFs;
Standalone-Fenster ist im Schema (Toolbar-Hintergrund == bg)."""

from __future__ import annotations

import ctypes
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_ROOT / "plugin" / "ref_grade")]
sys.modules["ref_grade"] = _pkg
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import schemes, win_chrome  # noqa: E402

_app = QApplication.instance() or QApplication([])


class _Widget:
    def winId(self):  # noqa: N802
        return 4242


class TestWinChrome(unittest.TestCase):
    def _calls(self, scheme):
        dwm = MagicMock()
        fake = types.SimpleNamespace(dwmapi=dwm, user32=MagicMock())
        with patch.object(sys, "platform", "win32"), \
                patch.object(ctypes, "windll", fake, create=True), \
                patch.object(ctypes, "byref", lambda o: o):
            win_chrome.apply(_Widget(), scheme)
        return {c.args[1].value: c.args[2].value for c in dwm.DwmSetWindowAttribute.call_args_list}

    def test_black(self):
        got = self._calls(schemes.SCHEMES["Black"])
        self.assertEqual(set(got), {20, 34, 35, 36})
        self.assertEqual(got[20], 1)
        self.assertEqual(got[34], 0x000A0A0A)
        self.assertEqual(got[35], 0x000A0A0A)
        self.assertEqual(got[36], 0x00E0E0E0)

    def test_light_scheme_not_dark_and_bgr_order(self):
        got = self._calls(schemes.SCHEMES["White"])
        self.assertEqual(got[20], 0)
        self.assertEqual(got[35], 0x00F7F7F7)
        self.assertEqual(win_chrome._colorref("#4a9eff"), 0x00FF9E4A)

    def test_noop_off_windows(self):
        with patch.object(sys, "platform", "linux"):
            win_chrome.apply(_Widget(), schemes.SCHEMES["Black"])   # must not raise


class TestStandaloneWindowScheme(unittest.TestCase):
    def test_toolbar_bg_is_scheme_bg(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("sa_gui", _ROOT / "standalone" / "gui.py")
        gui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gui)
        with patch("PyQt6.QtCore.QSettings.value", return_value="Black"):
            win, _lbl, _btn, _cont = gui.build_window("REFgrade test")
        try:
            from PyQt6.QtWidgets import QToolBar
            tb = win.findChild(QToolBar)
            self.assertIsNotNone(tb)
            self.assertEqual(win.windowTitle(), "REFgrade test")
            self.assertEqual(tb.palette().window().color().name().lower(),
                             schemes.SCHEMES["Black"].bg)
            self.assertIn(schemes.SCHEMES["Black"].bg, win.styleSheet().lower())
        finally:
            win.close()
            schemes.set_current(schemes.DEFAULT)


if __name__ == "__main__":
    unittest.main()
