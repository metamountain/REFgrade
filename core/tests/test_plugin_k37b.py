"""Tests für Paket K37b: Install-Dialog und Anbindung."""

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

from PyQt6.QtWidgets import QApplication, QDialog  # noqa: E402

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


def _make_install_fn_success():
    """Returns (install_fn, calls). install_fn logs two lines and returns the check
    string; calls records every flavor it was invoked with."""
    calls: list[str] = []
    def install_fn(flavor, log, cancelled):
        calls.append(flavor)
        log(f"Installing {flavor}")
        log("Done")
        return "2.14.1+cu130 True"
    return install_fn, calls


def _make_install_fn_error():
    """Returns an install_fn that raises InstallError."""
    def install_fn(flavor, log, cancelled):
        log("Step 1: ok")
        raise runtime.InstallError("Step 2 failed")
    return install_fn


def _make_install_fn_cancel():
    """Returns an install_fn that waits for cancelled() then raises Cancelled."""
    def install_fn(flavor, log, cancelled):
        log("Waiting...")
        for _ in range(200):
            if cancelled():
                raise runtime.Cancelled()
            time.sleep(0.01)
        raise runtime.Cancelled()
    return install_fn


class K37bTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k37b_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestDetectGPU(K37bTestBase):
    """Test 1: GPU detected → radio_gpu active and checked."""

    def test_gpu_detected(self):
        detect_fn = lambda: {"gpu_name": "RTX 4090", "flavor": "cu130", "message": ""}
        install_fn, calls = _make_install_fn_success()
        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()

        self.assertTrue(dlg.radio_gpu.isEnabled())
        self.assertTrue(dlg.radio_gpu.isChecked())
        self.assertIn("RTX 4090", dlg.radio_gpu.text())
        self.assertIn("cu130", dlg.radio_gpu.text())
        self.assertTrue(dlg.lbl_message.isHidden())
        self.assertEqual(calls, [])

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestDetectCPU(K37bTestBase):
    """Test 2: No GPU → radio_gpu disabled, radio_cpu checked, message shown."""

    def test_cpu_only(self):
        detect_fn = lambda: {"gpu_name": None, "flavor": "cpu", "message": "No NVIDIA GPU found - x"}
        install_fn, calls = _make_install_fn_success()
        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()

        self.assertFalse(dlg.radio_gpu.isEnabled())
        self.assertTrue(dlg.radio_cpu.isChecked())
        self.assertFalse(dlg.lbl_message.isHidden())
        self.assertEqual(dlg.lbl_message.text(), "No NVIDIA GPU found - x")
        self.assertEqual(calls, [])

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestInstallGPU(K37bTestBase):
    """Test 3: GPU selected, Install → success."""

    def test_install_gpu(self):
        detect_fn = lambda: {"gpu_name": "RTX 4090", "flavor": "cu130", "message": ""}
        received_flavor = []

        def install_fn(flavor, log, cancelled):
            received_flavor.append(flavor)
            log("line1")
            log("line2")
            return "2.14.1+cu130 True"

        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        dlg.btn_install.click()
        self.assertTrue(_wait_dialog_done(self.app, dlg))

        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(received_flavor, ["cu130"])
        self.assertEqual(dlg.result_text, "2.14.1+cu130 True")
        self.assertIn("line1", dlg.log_view.toPlainText())
        self.assertIn("line2", dlg.log_view.toPlainText())

        dlg.deleteLater()
        self.app.processEvents()


class TestInstallCPU(K37bTestBase):
    """Test 4: CPU selected → install_fn got 'cpu'."""

    def test_install_cpu(self):
        detect_fn = lambda: {"gpu_name": None, "flavor": "cpu", "message": ""}
        received_flavor = []

        def install_fn(flavor, log, cancelled):
            received_flavor.append(flavor)
            log("ok")
            return "2.14.1+cpu True"

        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        dlg.btn_install.click()
        self.assertTrue(_wait_dialog_done(self.app, dlg))

        self.assertEqual(dlg.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(received_flavor, ["cpu"])

        dlg.deleteLater()
        self.app.processEvents()


class TestInstallError(K37bTestBase):
    """Test 5: Error → dialog stays open, Retry button."""

    def test_install_error(self):
        detect_fn = lambda: {"gpu_name": "RTX 4090", "flavor": "cu130", "message": ""}
        install_fn = _make_install_fn_error()
        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        dlg.btn_install.click()
        _wait(self.app, lambda: dlg.btn_install.text() == "Retry")
        self.app.processEvents()

        self.assertTrue(dlg.isVisible())
        self.assertEqual(dlg.btn_install.text(), "Retry")
        self.assertTrue(dlg.btn_install.isEnabled())
        self.assertIn("Step 2 failed", dlg.log_view.toPlainText())

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestCancelDuringInstall(K37bTestBase):
    """Test 6: Cancel during install → Rejected."""

    def test_cancel_during(self):
        detect_fn = lambda: {"gpu_name": "RTX 4090", "flavor": "cu130", "message": ""}
        # The fake raises Cancelled after 2 s even without a cancel, so the test must
        # prove the click propagated: record whether cancelled() ever returned True.
        state = {"cancelled_seen": False}

        def install_fn(flavor, log, cancelled):
            log("Waiting...")
            for _ in range(200):
                if cancelled():
                    state["cancelled_seen"] = True
                    raise runtime.Cancelled()
                time.sleep(0.01)
            raise runtime.Cancelled()

        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        dlg.btn_install.click()
        self.app.processEvents()
        # Wait a bit for the install to start, then cancel
        time.sleep(0.05)
        self.app.processEvents()
        dlg.btn_cancel.click()
        self.assertTrue(_wait_dialog_done(self.app, dlg))

        self.assertFalse(dlg.isVisible())
        self.assertEqual(dlg.result(), QDialog.DialogCode.Rejected)
        self.assertTrue(state["cancelled_seen"], "cancel click did not reach the worker")

        dlg.deleteLater()
        self.app.processEvents()


class TestCancelBeforeStart(K37bTestBase):
    """Test 7: Cancel before start → Rejected, install_fn never called."""

    def test_cancel_before(self):
        detect_fn = lambda: {"gpu_name": "RTX 4090", "flavor": "cu130", "message": ""}
        called = []

        def install_fn(flavor, log, cancelled):
            called.append(flavor)
            return "should not reach"

        dlg = InstallDialog(parent=None, detect_fn=detect_fn, install_fn=install_fn)
        self.app.processEvents()
        dlg.show()
        self.app.processEvents()

        dlg.btn_cancel.click()
        self.app.processEvents()

        self.assertFalse(dlg.isVisible())
        self.assertEqual(dlg.result(), QDialog.DialogCode.Rejected)
        self.assertEqual(called, [])

        dlg.deleteLater()
        self.app.processEvents()


class TestKritaExtensionSource(K37bTestBase):
    """Test 8: krita_extension.py contains run_installer, python_ok, REFGRADE_PYTHON."""

    def test_source_contains(self):
        src = (_PLUGIN_DIR / "krita_extension.py").read_text(encoding="utf-8")
        self.assertIn("run_installer", src)
        self.assertIn("python_ok", src)
        self.assertIn("REFGRADE_PYTHON", src)


if __name__ == "__main__":
    unittest.main()
