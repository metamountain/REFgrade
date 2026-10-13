"""Tests für plugin/ref_grade/runtime.py – Systempython, ohne Krita, ohne Netz."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import config, runtime  # noqa: E402


class _Base(unittest.TestCase):
    """Setzt REFGRADE_RUNTIME auf ein Temp-Verzeichnis."""

    def setUp(self):
        self._old = os.environ.get("REFGRADE_RUNTIME")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["REFGRADE_RUNTIME"] = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        if self._old is None:
            del os.environ["REFGRADE_RUNTIME"]
        else:
            os.environ["REFGRADE_RUNTIME"] = self._old


class TestParseNvidiaSmi(_Base):

    def test_new_driver(self):
        text = "| NVIDIA-SMI 616.92   KMD Version: 616.92   CUDA UMD Version: 13.4 |"
        result = runtime.parse_nvidia_smi(text)
        self.assertEqual(result, {"driver": "616.92", "cuda": (13, 4)})

    def test_old_driver(self):
        text = "| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6 |"
        result = runtime.parse_nvidia_smi(text)
        self.assertEqual(result, {"driver": "560.94", "cuda": (12, 6)})

    def test_empty(self):
        result = runtime.parse_nvidia_smi("")
        self.assertEqual(result, {"driver": None, "cuda": None})


class TestParseCaps(_Base):

    def test_mixed(self):
        self.assertEqual(runtime.parse_caps("8.9\n[N/A]\n12.0\n"), [8.9, 12.0])


class TestChooseFlavor(_Base):

    def test_no_gpu(self):
        flavor, msg = runtime.choose_flavor(False, None, [])
        self.assertEqual(flavor, "cpu")

    def test_cuda_none(self):
        flavor, msg = runtime.choose_flavor(True, None, [])
        self.assertEqual(flavor, "cu130")
        self.assertEqual(msg, "")

    def test_cuda_13(self):
        flavor, msg = runtime.choose_flavor(True, (13, 4), [8.9])
        self.assertEqual(flavor, "cu130")
        self.assertEqual(msg, "")

    def test_cuda_126_older_gpu(self):
        flavor, msg = runtime.choose_flavor(True, (12, 6), [8.9])
        self.assertEqual(flavor, "cu126")
        self.assertIn("cu126", msg)

    def test_cuda_128_blackwell(self):
        flavor, msg = runtime.choose_flavor(True, (12, 8), [12.0])
        self.assertEqual(flavor, "cpu")
        self.assertIn("580", msg)

    def test_cuda_too_old(self):
        flavor, msg = runtime.choose_flavor(True, (12, 4), [8.9])
        self.assertEqual(flavor, "cpu")


class TestDetect(_Base):

    def test_no_nvidia_smi(self):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("nvidia-smi not found")

        result = runtime.detect(run=fake_run)
        self.assertIsNone(result["gpu_name"])
        self.assertEqual(result["flavor"], "cpu")

    def test_with_gpu(self):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) == 1:
                return MagicMock(returncode=0, stdout="| NVIDIA-SMI 616.92   CUDA UMD Version: 13.4 |")
            return MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 8.9\n")

        result = runtime.detect(run=fake_run)
        self.assertEqual(result["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(result["flavor"], "cu130")


class TestInstallCommands(_Base):

    def test_structure(self):
        cmds = runtime.install_commands("cu130")
        self.assertEqual(len(cmds), 3)
        # Alle Elemente sind str
        for cmd in cmds:
            for elem in cmd:
                self.assertIsInstance(elem, str)
        # Befehl 1: --managed-python und 3.12
        self.assertIn("--managed-python", cmds[0])
        self.assertIn("3.12", cmds[0])
        # Befehl 2: torch==2.14.1 und cu130 Index
        self.assertIn("torch==2.14.1", cmds[1])
        self.assertIn("https://download.pytorch.org/whl/cu130", cmds[1])
        # Befehl 3: endet mit numpy, kornia, pillow
        self.assertEqual(cmds[2][-3:], ["numpy", "kornia", "pillow"])


class TestInstallEnv(_Base):

    def test_no_python_keys(self):
        old = os.environ.get("PYTHONHOME")
        os.environ["PYTHONHOME"] = "x"
        try:
            env = runtime.install_env()
            for key in env:
                self.assertFalse(key.upper().startswith("PYTHON"),
                                 f"Key {key} starts with PYTHON")
            self.assertTrue(env["UV_PYTHON_INSTALL_DIR"].startswith(str(runtime.runtime_dir())))
        finally:
            if old is None:
                del os.environ["PYTHONHOME"]
            else:
                os.environ["PYTHONHOME"] = old


class _FakeProc:
    """Fake-Subprozess: liefert vordefinierte Zeilen, zeichnet terminate auf."""

    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.returncode


class TestInstall(_Base):

    def _make_zip(self) -> bytes:
        """ZIP-Bytes mit uv.exe (Inhalt b'fake')."""
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("uv.exe", b"fake")
        return buf.getvalue()

    def _fake_fetch(self, zip_bytes: bytes):
        def fetch(url, timeout=None):
            resp = MagicMock()
            resp.headers = {"Content-Length": str(len(zip_bytes))}
            data = io.BytesIO(zip_bytes)
            resp.read = lambda n: data.read(n)
            resp.close = lambda: None
            return resp
        return fetch

    def _fake_popen(self, commands: list[str], check_output: str = "2.14.1+cpu False"):
        """Fake-Popen: zeichnet auf, liefert je 2 Zeilen (Check: 1 Zeile)."""
        recorded: list[list[str]] = []
        procs: list[_FakeProc] = []

        def popen(cmd, **kwargs):
            recorded.append(cmd)
            if any(runtime.CHECK_CODE in c for c in cmd):
                proc = _FakeProc([check_output + "\n"], returncode=0)
            else:
                proc = _FakeProc(["line1\n", "line2\n"], returncode=0)
            procs.append(proc)
            return proc

        return popen, recorded, procs

    def test_happy_path(self):
        zip_bytes = self._make_zip()
        fetch = self._fake_fetch(zip_bytes)
        popen, recorded, procs = self._fake_popen([], "2.14.1+cpu False")

        lines: list[str] = []
        result = runtime.install("cpu", log=lines.append, popen=popen, fetch=fetch)

        # uv.exe existiert mit Inhalt b'fake'
        self.assertTrue(runtime.uv_exe().is_file())
        self.assertEqual(runtime.uv_exe().read_bytes(), b"fake")
        # 4 Befehle aufgezeichnet (3 + Prüfung)
        self.assertEqual(len(recorded), 4)
        # Rückgabe
        self.assertEqual(result, "2.14.1+cpu False")
        # is_installed() False (venv_python fehlt)
        self.assertFalse(runtime.is_installed())
        # venv_python anlegen → is_installed() True
        runtime.venv_python().parent.mkdir(parents=True, exist_ok=True)
        runtime.venv_python().touch()
        self.assertTrue(runtime.is_installed())
        # runtime.json
        data = json.loads(runtime.marker_path().read_text(encoding="utf-8"))
        self.assertEqual(data["flavor"], "cpu")

    def test_step2_failure(self):
        zip_bytes = self._make_zip()
        fetch = self._fake_fetch(zip_bytes)

        recorded: list[list[str]] = []
        procs: list[_FakeProc] = []

        def popen(cmd, **kwargs):
            recorded.append(cmd)
            if any(runtime.CHECK_CODE in c for c in cmd):
                proc = _FakeProc(["ok\n"], returncode=0)
            elif len(recorded) == 2:
                proc = _FakeProc(["error\n"], returncode=1)
            else:
                proc = _FakeProc(["line1\n", "line2\n"], returncode=0)
            procs.append(proc)
            return proc

        with self.assertRaises(runtime.InstallError) as ctx:
            runtime.install("cpu", log=lambda s: None, popen=popen, fetch=fetch)
        self.assertIn("Step 2", str(ctx.exception))
        self.assertFalse(runtime.marker_path().exists())

    def test_cancelled(self):
        # uv.exe vorab anlegen, damit kein Download nötig ist
        runtime.uv_exe().parent.mkdir(parents=True, exist_ok=True)
        runtime.uv_exe().write_bytes(b"fake")

        recorded: list[list[str]] = []
        procs: list[_FakeProc] = []
        cancel_count = [0]

        def cancelled():
            cancel_count[0] += 1
            return cancel_count[0] > 2  # 3. Aufruf → True

        def popen(cmd, **kwargs):
            recorded.append(cmd)
            proc = _FakeProc(["line1\n", "line2\n"], returncode=0)
            procs.append(proc)
            return proc

        with self.assertRaises(runtime.Cancelled):
            runtime.install("cpu", log=lambda s: None, cancelled=cancelled, popen=popen)

        # Der laufende Fake-Prozess hat terminate() erhalten
        self.assertTrue(any(p.terminated for p in procs))


class TestIsInstalled(_Base):

    def test_wrong_torch_version(self):
        runtime.venv_python().parent.mkdir(parents=True, exist_ok=True)
        runtime.venv_python().touch()
        runtime.marker_path().write_text(
            json.dumps({"torch": "2.0.0", "packages": list(runtime.PACKAGES)}),
            encoding="utf-8")
        self.assertFalse(runtime.is_installed())

    def test_broken_json(self):
        runtime.venv_python().parent.mkdir(parents=True, exist_ok=True)
        runtime.venv_python().touch()
        runtime.marker_path().write_text("{invalid json", encoding="utf-8")
        self.assertFalse(runtime.is_installed())


class TestConfigPythonExe(_Base):

    def test_runtime_installed(self):
        old = os.environ.pop("REFGRADE_PYTHON", None)
        try:
            # Runtime "installieren"
            runtime.venv_python().parent.mkdir(parents=True, exist_ok=True)
            runtime.venv_python().touch()
            runtime.marker_path().write_text(
                json.dumps({"torch": runtime.TORCH_VERSION,
                            "packages": list(runtime.PACKAGES)}),
                encoding="utf-8")
            result = config.python_exe()
            self.assertEqual(result, str(runtime.venv_python()))
        finally:
            if old is not None:
                os.environ["REFGRADE_PYTHON"] = old

    def test_env_overrides(self):
        old = os.environ.get("REFGRADE_PYTHON")
        os.environ["REFGRADE_PYTHON"] = "C:\\x\\python.exe"
        try:
            result = config.python_exe()
            self.assertEqual(result, "C:\\x\\python.exe")
        finally:
            if old is None:
                del os.environ["REFGRADE_PYTHON"]
            else:
                os.environ["REFGRADE_PYTHON"] = old


class TestPythonOk(_Base):

    def test_cache(self):
        runtime.reset_cache()
        calls = [0]

        def fake_run(cmd, **kwargs):
            calls[0] += 1
            return MagicMock(returncode=0)

        self.assertTrue(runtime.python_ok("C:/fake/python.exe", popen=fake_run))
        self.assertEqual(calls[0], 1)
        # Zweiter Aufruf: Cache, kein neuer run
        self.assertTrue(runtime.python_ok("C:/fake/python.exe", popen=fake_run))
        self.assertEqual(calls[0], 1)

    def test_failure_after_reset(self):
        runtime.reset_cache()
        calls = [0]

        def fake_run(cmd, **kwargs):
            calls[0] += 1
            return MagicMock(returncode=1)

        self.assertFalse(runtime.python_ok("C:/fake2/python.exe", popen=fake_run))
        self.assertEqual(calls[0], 1)
        # Nach reset_cache: neuer Aufruf
        runtime.reset_cache()
        self.assertFalse(runtime.python_ok("C:/fake2/python.exe", popen=fake_run))
        self.assertEqual(calls[0], 2)


if __name__ == "__main__":
    unittest.main()
