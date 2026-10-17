"""Tests für Paket K38f (refgrade.bat-Bootstrap, minimales Release-ZIP)."""

from __future__ import annotations

import sys
import unittest
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_DIR = _ROOT / "build"
_TMP = _ROOT / "tmp" / "release_k38f"
sys.path.insert(0, str(_BUILD_DIR))

from make_release_zip import build as build_release_zip  # noqa: E402


class TestReleaseZip(unittest.TestCase):
    """make_release_zip in einen Temp-Ordner: ZIP existiert, < 3 MB,
    enthält refgrade.bat + standalone/gui.py, keine .pyc, kein tests/."""

    def test_zip_contents(self):
        _TMP.mkdir(parents=True, exist_ok=True)
        out = build_release_zip(out=_TMP / "refgrade-test.zip")
        # ZIP existiert
        self.assertTrue(out.is_file(), f"ZIP fehlt: {out}")
        # < 3 MB
        size = out.stat().st_size
        self.assertLess(size, 3 * 1024 * 1024, f"ZIP zu groß: {size} bytes")
        names = zipfile.ZipFile(out).namelist()
        # enthält refgrade.bat und standalone/gui.py
        self.assertIn("refgrade.bat", names)
        self.assertIn("standalone/gui.py", names)
        # keine .pyc
        pyc = [n for n in names if n.endswith(".pyc")]
        self.assertFalse(pyc, f".pyc im ZIP: {pyc}")
        # kein Pfad mit tests/
        tests = [n for n in names if "tests/" in n]
        self.assertFalse(tests, f"tests/ im ZIP: {tests}")


class TestRefgradeBat(unittest.TestCase):
    """refgrade.bat enthält UV_LINK_MODE, --managed-python, PyQt6, standalone\\gui.py."""

    def test_bat_contains_required(self):
        bat = (_ROOT / "refgrade.bat").read_text(encoding="utf-8")
        for needle in ("UV_LINK_MODE", "--managed-python", "PyQt6", "standalone\\gui.py"):
            self.assertIn(needle, bat, f"refgrade.bat fehlt: {needle}")


if __name__ == "__main__":
    unittest.main()
