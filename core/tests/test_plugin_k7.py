"""Tests for Paket K7: Tooltips (Englisch), ui_help.py, install.py --dry-run."""

from __future__ import annotations

import html
import os
import subprocess
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

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade.dialog import RefGradeDialog, _fmt_num, load_help  # noqa: E402
from ref_grade.ui_help import UI_HELP  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


class SliderTooltipTest(unittest.TestCase):
    """Every visible slider tooltip contains HELP[name]['en'] and the range line."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.dlg = RefGradeDialog()
        self.help = load_help()
        self.dlg.close()

    def tearDown(self):
        self.dlg.deleteLater()
        self.app.processEvents()

    def _visible_sliders(self) -> list:
        """All SliderRow widgets in the sections (excluding the global MASTER)."""
        rows = []
        for key, (_h, section_rows) in self.dlg.sections.items():
            rows.extend(section_rows)
        return rows

    def test_slider_tooltips_contain_short_and_tip(self):
        """K33a: tooltip shows short (line 1) + tip (line 2)."""
        checked = 0
        for r in self._visible_sliders():
            h = self.help.get(r.name)
            if not h:
                continue
            tip = r.toolTip()
            self.assertIn(html.escape(h["short"]), tip,
                          f"Slider {r.name}: tooltip missing short text")
            self.assertIn(html.escape(h["tip"]), tip,
                          f"Slider {r.name}: tooltip missing tip text")
            checked += 1
        self.assertGreater(checked, 0, "No sliders checked")


class UiHelpTest(unittest.TestCase):
    """Every UI_HELP entry maps to an existing widget with a non-empty tooltip."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.dlg = RefGradeDialog()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

    def test_all_ui_help_entries_have_widget(self):
        missing = []
        for obj_name in UI_HELP:
            w = self.dlg.findChild(QWidget, obj_name)
            if w is None:
                missing.append(obj_name)
        self.assertEqual(missing, [],
                         f"UI_HELP entries without a widget: {missing}")

    def test_all_ui_help_widgets_have_tooltip(self):
        empty = []
        for obj_name in UI_HELP:
            w = self.dlg.findChild(QWidget, obj_name)
            if w is not None:
                tip = w.toolTip()
                if not tip.strip():
                    empty.append(obj_name)
        self.assertEqual(empty, [],
                         f"Widgets with objectName from UI_HELP but empty tooltip: {empty}")

    def test_no_orphan_objectnames(self):
        """Every widget with an objectName that is in UI_HELP has a tooltip."""
        # Collect all widgets with objectNames
        all_widgets = self.dlg.findChildren(QWidget)
        orphans = []
        for w in all_widgets:
            name = w.objectName()
            if name in UI_HELP and not w.toolTip().strip():
                orphans.append(name)
        self.assertEqual(orphans, [],
                         f"Widgets with UI_HELP objectName but no tooltip: {orphans}")


class InstallDryRunTest(unittest.TestCase):
    """install.py --dry-run prints the three targets and writes nothing."""

    def test_dry_run(self):
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_install_"))
        env = os.environ.copy()
        env["APPDATA"] = str(tmp)

        result = subprocess.run(
            [sys.executable, str(_ROOT / "install.py"), "--dry-run"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")

        # All three targets are printed
        self.assertIn("krita/pykrita/ref_grade", result.stdout.replace("\\", "/"))
        self.assertIn("ref_grade.desktop", result.stdout)
        self.assertIn("ref_grade.action", result.stdout)

        # Nothing was created under the temp APPDATA
        krita_dir = tmp / "krita"
        self.assertFalse(krita_dir.exists(),
                         f"APPDATA/krita was created during --dry-run: {krita_dir}")


if __name__ == "__main__":
    unittest.main()
