"""Tests für Paket K38g (Installer-Fixes) in plugin/ref_grade/runtime.py."""

from __future__ import annotations

import json
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

from ref_grade import runtime  # noqa: E402


class _Base(unittest.TestCase):
    def setUp(self):
        self._old_rt = os.environ.get("REFGRADE_RUNTIME")
        self._old_cfg = os.environ.get("REFGRADE_CONFIG")
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["REFGRADE_RUNTIME"] = self._tmp.name
        self.cfg_path = Path(self._tmp.name) / "config.json"
        os.environ["REFGRADE_CONFIG"] = str(self.cfg_path)
        runtime.reset_cache()

    def tearDown(self):
        for var in ("REFGRADE_RUNTIME", "REFGRADE_CONFIG"):
            if var in os.environ:
                del os.environ[var]
        if self._old_rt is not None:
            os.environ["REFGRADE_RUNTIME"] = self._old_rt
        if self._old_cfg is not None:
            os.environ["REFGRADE_CONFIG"] = self._old_cfg
        self._tmp.cleanup()
        runtime.reset_cache()


class _FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.returncode


class TestCheckTargetNonexistent(_Base):

    def test_probe_in_first_existing_dir_creates_nothing(self):
        target = Path(self._tmp.name) / "a" / "b" / "c"
        self.assertFalse(target.exists())
        result = runtime.check_target(target, "cpu")
        self.assertEqual(result, "")
        # Nichts angelegt: a existiert nachher nicht.
        self.assertFalse((Path(self._tmp.name) / "a").exists())


class TestCheckCode(_Base):

    def test_cpu_is_check_code(self):
        self.assertEqual(runtime.check_code("cpu"), runtime.CHECK_CODE)

    def test_rocm_has_cuda_exit(self):
        code = runtime.check_code("rocm")
        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("sys.exit", code)

    def test_xpu_has_xpu_exit(self):
        code = runtime.check_code("xpu")
        self.assertIn("torch.xpu.is_available()", code)

    def test_cu130_has_cuda_exit(self):
        code = runtime.check_code("cu130")
        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("sys.exit", code)


class TestInstallFallbackAllGpu(_Base):

    def test_cu130_check_exit3_falls_back_to_cpu(self):
        # uv.exe vorab anlegen, damit kein Download nötig ist.
        runtime.uv_exe().parent.mkdir(parents=True, exist_ok=True)
        runtime.uv_exe().write_bytes(b"fake")

        def popen(cmd, **kwargs):
            joined = " ".join(cmd)
            # cu130-GPU-Check (enthält die CUDA-Exit-Zeile) → Exit 3.
            if "sys.exit(0 if torch.cuda.is_available() else 3)" in joined:
                return _FakeProc(["2.14.1+cu130 True\n"], returncode=3)
            # CPU-Check (nur CHECK_CODE) → Exit 0.
            if runtime.CHECK_CODE in joined:
                return _FakeProc(["2.14.1+cpu False\n"], returncode=0)
            return _FakeProc(["line1\n", "line2\n"], returncode=0)

        lines: list[str] = []
        result = runtime.install("cu130", log=lines.append, popen=popen)

        self.assertEqual(result, "2.14.1+cpu False")
        self.assertIn("GPU build failed the check - installing the CPU version.", lines)
        data = json.loads(runtime.marker_path().read_text(encoding="utf-8"))
        self.assertEqual(data["flavor"], "cpu")


@unittest.skipIf(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                 "GPU hidden for test runs (keeps VRAM free for BeeLlama)")
class TestCheckCodeRealRun(_Base):
    """Echter Lauf: System-Python mit torch; erwartet Exit 0 auf dieser Maschine (RTX 4090)."""

    def test_cu130_real_run_exit0(self):
        r = subprocess.run(
            [sys.executable, "-E", "-c", runtime.check_code("cu130")],
            capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, msg=f"stdout={r.stdout!r} stderr={r.stderr!r}")


if __name__ == "__main__":
    unittest.main()
