"""Tests für plugin/ref_grade/config.py – Systempython, ohne Krita."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import config  # noqa: E402


class ConfigTest(unittest.TestCase):

    def test_root(self):
        """ROOT = repository root (samefile)."""
        expected = Path(__file__).resolve().parents[2]
        self.assertTrue(config.ROOT.samefile(expected))

    def test_core(self):
        """CORE = ROOT / core."""
        self.assertTrue(config.CORE.samefile(config.ROOT / "core"))

    def test_pool(self):
        """POOL = ROOT / bench / pool."""
        self.assertTrue(config.POOL.samefile(config.ROOT / "bench" / "pool"))

    def test_port(self):
        """PORT = 47123."""
        self.assertEqual(config.PORT, 47123)

    def test_python_exe_env(self):
        """REFGRADE_PYTHON wird bevorzugt."""
        old = os.environ.get("REFGRADE_PYTHON")
        os.environ["REFGRADE_PYTHON"] = "C:/custom/python.exe"
        try:
            self.assertEqual(config.python_exe(), "C:/custom/python.exe")
        finally:
            if old is None:
                del os.environ["REFGRADE_PYTHON"]
            else:
                os.environ["REFGRADE_PYTHON"] = old

    def test_python_exe_which(self):
        """Ohne Env: kein WindowsApps-/PythonManager-Stub im Ergebnis."""
        old = os.environ.pop("REFGRADE_PYTHON", None)
        try:
            result = config.python_exe()
            self.assertIsNotNone(result, "python_exe() lieferte None")
            self.assertNotIn("windowsapps", result.lower())
            self.assertNotIn("pythonmanager", result.lower())
        finally:
            if old is not None:
                os.environ["REFGRADE_PYTHON"] = old

    def test_lut_dir_with_resource(self):
        """kritarc mit ResourceDirectory → <dir>/luts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("ResourceDirectory=C:/X/krita\nOther=stuff\n")
            path = Path(f.name)
        try:
            result = config.lut_dir(path)
            self.assertEqual(result, Path("C:/X/krita/luts"))
        finally:
            path.unlink(missing_ok=True)

    def test_lut_dir_fallback(self):
        """Ohne ResourceDirectory-Zeile → %APPDATA%\\krita\\luts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Other=stuff\n")
            path = Path(f.name)
        try:
            result = config.lut_dir(path)
            appdata = os.environ.get("APPDATA", "")
            expected = Path(appdata) / "krita" / "luts"
            self.assertEqual(result, expected)
        finally:
            path.unlink(missing_ok=True)

    def test_lut_dir_missing_file(self):
        """Nicht existierende kritarc → Fallback."""
        result = config.lut_dir(Path("/nonexistent/kritarc"))
        appdata = os.environ.get("APPDATA", "")
        expected = Path(appdata) / "krita" / "luts"
        self.assertEqual(result, expected)

    def test_temp_dir(self):
        """temp_dir() existiert nach Aufruf."""
        d = config.temp_dir()
        self.assertTrue(d.is_dir())
        self.assertTrue(d.name == "ref_grade")


if __name__ == "__main__":
    unittest.main()
