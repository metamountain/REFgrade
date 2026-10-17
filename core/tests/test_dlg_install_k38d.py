"""Tests für Paket K38d: Installer-Dialog (Build-Auswahl, Ort, vorhandenes Python)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog  # noqa: E402

from ref_grade import runtime  # noqa: E402
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


class K38dTestBase(unittest.TestCase):
    """Offscreen QApplication; REFGRADE_CONFIG in einem Temp-Ordner, REFGRADE_RUNTIME aus."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k38d_"))
        self._old_cfg = os.environ.get("REFGRADE_CONFIG")
        self._old_rt = os.environ.get("REFGRADE_RUNTIME")
        os.environ.pop("REFGRADE_RUNTIME", None)
        os.environ["REFGRADE_CONFIG"] = str(self.tmp / "config.json")
        runtime.reset_cache()

    def tearDown(self):
        import shutil
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


class TestComboEntries(K38dTestBase):
    """1: Auto-Eintrag 0, 1 + len(FLAVORS) Einträge, Adapter in der Detected-Zeile."""

    def test_combo_entries(self):
        detect_fn = lambda: _detect_cpu(adapters=["AMD Radeon(TM) Graphics"])
        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=lambda *a: "")
        self.app.processEvents()

        self.assertEqual(dlg.combo_build.count(), 1 + len(runtime.FLAVORS))
        self.assertTrue(
            dlg.combo_build.itemText(0).startswith("Auto (recommended): CPU only"),
            f"unexpected entry 0: {dlg.combo_build.itemText(0)!r}")
        self.assertEqual(dlg.combo_build.itemData(0), "cpu")
        self.assertIn("AMD Radeon(TM) Graphics", dlg.lbl_detected.text())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestNonAsciiPath(K38dTestBase):
    """2: Ort mit 'ü' → Hinweis-Label nicht leer, Install gesperrt."""

    def test_non_ascii_locks_install(self):
        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu,
                            install_fn=lambda *a: "")
        self.app.processEvents()

        dlg.edit_path.setText(str(self.tmp / "pfad_mit_ü"))
        self.app.processEvents()

        self.assertNotEqual(dlg.lbl_hint.text(), "")
        self.assertFalse(dlg.btn_install.isEnabled())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestInstallFlavorAndDir(K38dTestBase):
    """3: Combo cu126 + Install → install_fn bekommt cu126; runtime_dir() = Ort."""

    def test_install_cu126_to_custom_dir(self):
        target = self.tmp / "rt"
        target.mkdir()
        received: list[str] = []

        def install_fn(flavor, log, cancelled):
            received.append(flavor)
            log("ok")
            return "2.14.1+cu126 True"

        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        idx = next(i for i in range(dlg.combo_build.count())
                   if dlg.combo_build.itemData(i) == "cu126")
        dlg.combo_build.setCurrentIndex(idx)
        dlg.edit_path.setText(str(target))
        self.app.processEvents()
        self.assertTrue(dlg.btn_install.isEnabled(),
                        f"Install locked, hint: {dlg.lbl_hint.text()!r}")

        dlg.btn_install.click()
        self.assertTrue(_wait_dialog_done(self.app, dlg))

        self.assertEqual(received, ["cu126"])
        self.assertEqual(runtime.runtime_dir(), target)

        dlg.deleteLater()
        self.app.processEvents()


class TestUseExisting(K38dTestBase):
    """4: 'Use existing Python' → use_existing "" → Dialog accepted."""

    def test_use_existing_accepts(self):
        dlg = InstallDialog(parent=None, detect_fn=_detect_cpu,
                            install_fn=lambda *a: "")
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        with patch.object(QFileDialog, "getOpenFileName",
                          return_value=(r"C:\Python312\python.exe", "python.exe")), \
             patch.object(runtime, "use_existing", return_value=""):
            dlg.btn_existing.click()
            self.app.processEvents()

        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)
        self.assertFalse(dlg.isVisible())

        dlg.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
