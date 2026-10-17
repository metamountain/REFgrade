"""Tests für Paket K38b (GPU-Erkennung) in plugin/ref_grade/runtime.py – ohne Netz."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import runtime  # noqa: E402


class _Base(unittest.TestCase):
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


def _fake_run(adapters, nvidia="missing", nvidia_query=""):
    """Fake-Subprocess: PowerShell → Adapter, nvidia-smi → konfigurierbar."""
    def run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "Get-CimInstance" in joined:
            return MagicMock(returncode=0, stdout="\n".join(adapters) + "\n")
        if "nvidia-smi" in joined:
            if nvidia == "missing":
                raise FileNotFoundError("nvidia-smi not found")
            if "--query-gpu" in joined:
                return MagicMock(returncode=0, stdout=nvidia_query)
            return MagicMock(returncode=0, stdout=nvidia)
        raise AssertionError(f"unexpected command: {cmd}")
    return run


class TestDetectK38b(_Base):

    def test_integrated_amd_is_cpu(self):
        run = _fake_run(["AMD Radeon(TM) Graphics"], nvidia="missing")
        result = runtime.detect(run=run)
        self.assertEqual(result["flavor"], "cpu")
        self.assertIsNone(result["gpu_name"])

    def test_discrete_amd_is_rocm(self):
        run = _fake_run(["AMD Radeon RX 7800 XT"], nvidia="missing")
        result = runtime.detect(run=run)
        self.assertEqual(result["flavor"], "rocm")
        self.assertEqual(result["gpu_name"], "AMD Radeon RX 7800 XT")

    def test_intel_arc_is_xpu(self):
        run = _fake_run(["Intel(R) Arc(TM) A770 Graphics"], nvidia="missing")
        result = runtime.detect(run=run)
        self.assertEqual(result["flavor"], "xpu")
        self.assertEqual(result["gpu_name"], "Intel(R) Arc(TM) A770 Graphics")

    def test_nvidia_wins_over_adapters(self):
        run = _fake_run(
            ["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4090"],
            nvidia="| NVIDIA-SMI 616.92   CUDA UMD Version: 13.4 |",
            nvidia_query="NVIDIA GeForce RTX 4090, 8.9\n")
        result = runtime.detect(run=run)
        self.assertEqual(result["flavor"], "cu130")
        self.assertEqual(result["adapters"],
                         ["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 4090"])


class TestInstallCommandsK38b(_Base):

    def test_rocm_four_commands(self):
        cmds = runtime.install_commands("rocm")
        self.assertEqual(len(cmds), 4)
        self.assertEqual(cmds[1][-4:], list(runtime.ROCM_SDK))
        self.assertEqual(cmds[2][-1:], list(runtime.ROCM_TORCH))

    def test_xpu_version_and_index(self):
        cmds = runtime.install_commands("xpu")
        self.assertIn("torch==2.14.1", cmds[1])
        self.assertIn("https://download.pytorch.org/whl/xpu", cmds[1])


class TestIsInstalledK38b(_Base):

    def _marker(self, flavor, torch):
        runtime.venv_python().parent.mkdir(parents=True, exist_ok=True)
        runtime.venv_python().touch()
        runtime.marker_path().write_text(
            json.dumps({"flavor": flavor, "torch": torch,
                        "packages": list(runtime.PACKAGES)}),
            encoding="utf-8")

    def test_rocm_correct_torch(self):
        self._marker("rocm", "2.9.1")
        self.assertTrue(runtime.is_installed())

    def test_rocm_wrong_torch(self):
        self._marker("rocm", "2.14.1")
        self.assertFalse(runtime.is_installed())


class _FakeProc:
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return self.returncode


class TestInstallFallback(_Base):

    def test_rocm_check_fails_falls_back_to_cpu(self):
        runtime.uv_exe().parent.mkdir(parents=True, exist_ok=True)
        runtime.uv_exe().write_bytes(b"fake")

        check_calls = [0]

        def popen(cmd, **kwargs):
            if any(runtime.CHECK_CODE in c for c in cmd):
                check_calls[0] += 1
                if check_calls[0] == 1:
                    return _FakeProc(["rocm check failed\n"], returncode=1)
                return _FakeProc(["2.14.1+cpu False\n"], returncode=0)
            return _FakeProc(["line1\n", "line2\n"], returncode=0)

        lines: list[str] = []
        result = runtime.install("rocm", log=lines.append, popen=popen,
                                 fetch=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no net")))

        self.assertEqual(result, "2.14.1+cpu False")
        self.assertEqual(check_calls[0], 2)
        self.assertIn("GPU build failed the check - installing the CPU version.", lines)
        data = json.loads(runtime.marker_path().read_text(encoding="utf-8"))
        self.assertEqual(data["flavor"], "cpu")


if __name__ == "__main__":
    unittest.main()
