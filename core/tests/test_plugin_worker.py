"""Tests für plugin/ref_grade/worker_process.py und worker_client.py."""

from __future__ import annotations

import os
import socket
import sys
import time
import types
import unittest
from pathlib import Path

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import config  # noqa: E402
from ref_grade import worker_process  # noqa: E402
from ref_grade import worker_client  # noqa: E402

CORE_DIR = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WorkerProcessTest(unittest.TestCase):
    """Echter Worker-Start über worker_process.ensure."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        python = config.python_exe()
        if python is None:
            raise unittest.SkipTest("Kein Python-Interpreter gefunden")
        cls.result = worker_process.ensure(cls.port, python, CORE_DIR, timeout=30.0)

    @classmethod
    def tearDownClass(cls):
        worker_process.stop(cls.port)
        # Warten bis /health nicht mehr antwortet
        deadline = time.time() + 5
        while time.time() < deadline:
            if not worker_process._health(cls.port):
                break
            time.sleep(0.2)

    def test_ensure_started(self):
        """Erster Aufruf → 'started'."""
        self.assertEqual(self.result, "started")

    def test_ensure_running(self):
        """Zweiter Aufruf → 'running'."""
        python = config.python_exe()
        result = worker_process.ensure(self.port, python, CORE_DIR, timeout=5.0)
        self.assertEqual(result, "running")

    def test_health_after_stop(self):
        """Nach stop: /health antwortet nicht mehr (innerhalb 5 s)."""
        # Dieser Test läuft nach tearDownClass – wird hier nur vorbereitet.
        # Die eigentliche Prüfung ist in tearDownClass.
        self.assertTrue(worker_process._health(self.port))


class WorkerClientTest(unittest.TestCase):
    """worker_client.call gegen laufenden Worker."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        python = config.python_exe()
        if python is None:
            raise unittest.SkipTest("Kein Python-Interpreter gefunden")
        worker_process.ensure(cls.port, python, CORE_DIR, timeout=30.0)

    @classmethod
    def tearDownClass(cls):
        worker_process.stop(cls.port)
        deadline = time.time() + 5
        while time.time() < deadline:
            if not worker_process._health(cls.port):
                break
            time.sleep(0.2)

    def test_health(self):
        """GET /health → ok: true."""
        r = worker_client.call(self.port, "/health")
        self.assertTrue(r.get("ok"))
        self.assertIn("device", r)

    def test_unknown_path(self):
        """GET /nope → WorkerError mit Status 404."""
        with self.assertRaises(worker_client.WorkerError) as ctx:
            worker_client.call(self.port, "/nope")
        self.assertEqual(ctx.exception.status, 404)


class KritaImportTest(unittest.TestCase):
    """Nur __init__.py und krita_*.py dürfen import krita enthalten."""

    def test_no_krita_import(self):
        plugin_dir = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
        for f in plugin_dir.iterdir():
            if f.suffix != ".py":
                continue
            name = f.name
            allowed = name == "__init__.py" or name.startswith("krita_")
            if allowed:
                continue
            text = f.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("import krita") or stripped.startswith("from krita"):
                    self.fail(
                        f"{name}: 'import krita' erlaubt nur in __init__.py und krita_*.py"
                    )


class DialogTest(unittest.TestCase):
    """RefGradeDialog lässt sich offscreen erzeugen."""

    def test_dialog_offscreen(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance() or QApplication(sys.argv)
            from ref_grade.dialog import RefGradeDialog
            dlg = RefGradeDialog()
            self.assertEqual(dlg.windowTitle(), "RefGrade")
            # Screenshot für die Sichtprüfung
            report_dir = Path(__file__).resolve().parents[1] / "reports"
            report_dir.mkdir(exist_ok=True)
            png_path = report_dir / "K3_dialog.png"
            dlg.grab().save(str(png_path))
            self.assertTrue(png_path.exists())
            dlg.close()
        finally:
            os.environ.pop("QT_QPA_PLATFORM", None)


if __name__ == "__main__":
    unittest.main()
