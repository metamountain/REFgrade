"""Tests für Paket K41a: 4 Schemata, Legacy-Namen, keine harten Chrome-Farben, Branding."""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGrade_test_k41a")

from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from ref_grade import schemes  # noqa: E402

_app = QApplication.instance() or QApplication([])

TABLE = {
    "Black": ("#0a0a0a", "#1a1a1a", "#e0e0e0", "#4a9eff", "#666666"),
    "Dark Grey": ("#1e1e1e", "#2c2c2c", "#d6d6d6", "#4a9eff", "#8a8a8a"),
    "Light Grey": ("#c8c8c8", "#d8d8d8", "#1a1a1a", "#2563eb", "#5a5a5a"),
    "White": ("#f7f7f7", "#ffffff", "#1a1a1a", "#2563eb", "#8c8c8c"),
}


class TestSchemes(unittest.TestCase):
    def tearDown(self):
        schemes.set_current(schemes.DEFAULT)

    def test_four_schemes_in_order(self):
        self.assertEqual(list(schemes.SCHEMES), ["Black", "Dark Grey", "Light Grey", "White"])
        for name, vals in TABLE.items():
            s = schemes.SCHEMES[name]
            self.assertEqual((s.bg, s.surface, s.text, s.accent, s.dim), vals)
        self.assertEqual(schemes.DEFAULT, "Black")

    def test_legacy_names(self):
        self.assertEqual(schemes.set_current("Carbon").name, "Black")
        self.assertEqual(schemes.set_current("Anthrazite").name, "Dark Grey")
        self.assertEqual(schemes.set_current("Paper").name, "White")


class TestNoHardcodedColours(unittest.TestCase):
    def test_scan(self):
        hexre = re.compile(r"#[0-9a-fA-F]{6}")
        qcre = re.compile(r"QColor\(\d")
        bad = []
        for f in sorted(_PLUGIN_DIR.glob("*.py")):
            if f.name == "schemes.py":
                continue
            in_functional = False
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if f.name == "style.py" and line.startswith("# ── functional colours"):
                    in_functional = True
                elif f.name == "style.py" and line.startswith("# ── end functional colours"):
                    in_functional = False
                if in_functional or "# functional" in line:
                    continue
                if hexre.search(line) or qcre.search(line):
                    bad.append(f"{f.name}:{n}: {line.strip()}")
        self.assertEqual(bad, [], "hard-coded colours:\n" + "\n".join(bad[:40]))


class TestBranding(unittest.TestCase):
    def test_dialog_wordmark_and_scheme(self):
        from ref_grade.dialog import RefGradeDialog
        from ref_grade.version import __version__
        dlg = RefGradeDialog()
        try:
            self.assertTrue(dlg.windowTitle().startswith("REFgrade"))
            self.assertIn(__version__, dlg.windowTitle())
            refs = [l for l in dlg.findChildren(QLabel) if l.text() == "REF"]
            self.assertTrue(refs)
            sc = schemes.current_scheme()
            self.assertEqual(refs[0].property("wordmark"), "ref")
            self.assertIn(sc.accent.lower(), dlg.styleSheet().lower())
            dlg._set_scheme("White")
            self.assertEqual(dlg.palette().window().color().name().lower(), "#f7f7f7")
            self.assertEqual(schemes.current_scheme().name, "White")
        finally:
            dlg.close()


if __name__ == "__main__":
    unittest.main()
