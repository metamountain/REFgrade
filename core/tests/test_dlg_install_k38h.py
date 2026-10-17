"""Tests für Paket K38h: Installer-Dialog kompakt, im Farbschema."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import runtime, schemes  # noqa: E402
from ref_grade.dlg_install import InstallDialog  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 5000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _wait_dialog_done(app: QApplication, dlg: InstallDialog, timeout_ms: int = 5000) -> bool:
    """Wait until the dialog is no longer visible (accepted or rejected)."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if not dlg.isVisible():
            return True
        time.sleep(0.005)
    return False


def _detect_cpu(adapters=None):
    return {"gpu_name": None, "flavor": "cpu", "message": "",
            "adapters": adapters if adapters is not None else []}


def _detect_gpu():
    return {"gpu_name": "RTX 4090", "flavor": "cu130", "message": "", "adapters": []}


class K38hTestBase(unittest.TestCase):
    """Offscreen QApplication; REFGRADE_CONFIG in einem Temp-Ordner, REFGRADE_RUNTIME aus."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k38h_"))
        self._old_cfg = os.environ.get("REFGRADE_CONFIG")
        self._old_rt = os.environ.get("REFGRADE_RUNTIME")
        os.environ.pop("REFGRADE_RUNTIME", None)
        os.environ["REFGRADE_CONFIG"] = str(self.tmp / "config.json")
        runtime.reset_cache()
        self._scheme = schemes.current_scheme_name()

    def tearDown(self):
        import shutil
        schemes.set_current(self._scheme)
        if self._old_cfg is None:
            os.environ.pop("REFGRADE_CONFIG", None)
        else:
            os.environ["REFGRADE_CONFIG"] = self._old_cfg
        if self._old_rt is None:
            os.environ.pop("REFGRADE_RUNTIME", None)
        else:
            os.environ["REFGRADE_RUNTIME"] = self._old_rt
        runtime.reset_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestCompactLayout(K38hTestBase):
    """1: Bei 760×560 — Abstand Combo-Unterkante → 'Install to'-Oberkante ≤ 3× Combo-Höhe."""

    def test_gap_combo_to_install_to(self):
        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu, install_fn=lambda *a: "")
        dlg.show()
        dlg.resize(760, 560)
        self.app.processEvents()

        combo_bottom = dlg.combo_build.y() + dlg.combo_build.height()
        gap = dlg.lbl_install_to.y() - combo_bottom
        self.assertLessEqual(gap, 3 * dlg.combo_build.height(),
                             f"gap {gap}px > 3x combo height {3 * dlg.combo_build.height()}px")
        # Der Download-Label liegt dazwischen (neue Zeile zwischen Combo und 'Install to').
        self.assertGreaterEqual(dlg.lbl_download.y(), combo_bottom)
        self.assertLess(dlg.lbl_download.y(), dlg.lbl_install_to.y())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestDownloadLabel(K38hTestBase):
    """2: Download-Label enthält 'GB'; nach Wechsel auf 'CPU only' → '0.3 GB'."""

    def test_download_label(self):
        dlg = InstallDialog(parent=None, detect_fn=_detect_gpu, install_fn=lambda *a: "")
        self.app.processEvents()

        self.assertIn("GB", dlg.lbl_download.text())
        self.assertIn("3.0 GB", dlg.lbl_download.text())  # cu130 = 3000 MB

        idx = next(i for i in range(dlg.combo_build.count())
                   if dlg.combo_build.itemData(i) == "cpu")
        dlg.combo_build.setCurrentIndex(idx)
        self.app.processEvents()

        self.assertIn("0.3 GB", dlg.lbl_download.text())   # 250 MB gerundet
        self.assertIn("0.5 GB", dlg.lbl_download.text())   # 2×250 MB = 500 MB

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestLogVisibility(K38hTestBase):
    """3: Log-Feld vor Install unsichtbar, nach Klick auf Install sichtbar."""

    def test_log_hidden_until_install(self):
        def install_fn(flavor, log, cancelled):
            log("working")
            time.sleep(0.3)
            return "2.14.1+cpu True"

        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu, install_fn=install_fn)
        dlg.show()
        self.app.processEvents()

        self.assertTrue(dlg.log_view.isHidden())

        dlg.btn_install.click()
        self.app.processEvents()
        self.assertFalse(dlg.log_view.isHidden())

        self.assertTrue(_wait_dialog_done(self.app, dlg))
        dlg.deleteLater()
        self.app.processEvents()


class TestButtonRow(K38hTestBase):
    """4: Install rechts von Cancel, beide in derselben Zeile; 'Use existing Python' links."""

    def test_install_right_of_cancel(self):
        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu, install_fn=lambda *a: "")
        dlg.show()
        dlg.resize(760, 560)
        self.app.processEvents()

        self.assertGreater(dlg.btn_install.x(), dlg.btn_cancel.x())
        self.assertEqual(dlg.btn_install.y(), dlg.btn_cancel.y())
        self.assertLess(dlg.btn_existing.x(), dlg.btn_cancel.x())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestSchemeBackground(K38hTestBase):
    """5: Hintergrundfarbe des Dialogs == bg des aktiven Schemas."""

    def test_background_matches_scheme(self):
        schemes.set_current("White")
        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu, install_fn=lambda *a: "")
        self.app.processEvents()

        sc = schemes.current_scheme()
        self.assertEqual(dlg.palette().window().color().name().lower(), sc.bg.lower())
        self.assertIn(sc.accent.lower(), dlg.styleSheet().lower())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
