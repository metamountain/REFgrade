"""Tests für Paket K38e (Standalone nutzt denselben Installer, CLI, torch_check weg)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
import unittest
from pathlib import Path

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import runtime  # noqa: E402
from slow import slow  # noqa: E402


class TestRuntimeCliDetect(unittest.TestCase):
    """`python -m ref_grade.runtime --detect` (subprocess, cwd plugin)."""

    @slow
    def test_detect_json(self):
        r = subprocess.run(
            [sys.executable, "-m", "ref_grade.runtime", "--detect"],
            cwd=str(_PLUGIN_DIR.parent),
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        data = json.loads(r.stdout)
        self.assertIn("flavor", data)
        self.assertEqual(data["flavor"], "cu130")  # diese Maschine: RTX 4090, CUDA 13


class TestInstallCommandsExtra(unittest.TestCase):
    """install_commands mit extra=("PyQt6",) → letzter pip-Befehl endet auf PyQt6."""

    def test_cu130_last_command_ends_with_pyyqt6(self):
        cmds = runtime.install_commands("cu130", extra=("PyQt6",))
        self.assertTrue(cmds[-1][-1].endswith("PyQt6"))

    def test_rocm_last_command_ends_with_pyyqt6(self):
        cmds = runtime.install_commands("rocm", extra=("PyQt6",))
        self.assertTrue(cmds[-1][-1].endswith("PyQt6"))

    def test_default_extra_empty(self):
        cmds = runtime.install_commands("cu130")
        self.assertFalse(cmds[-1][-1].endswith("PyQt6"))


class TestTorchCheckGone(unittest.TestCase):
    """Kein torch_check mehr in standalone/*.py."""

    def test_no_torch_check_in_standalone_py(self):
        standalone_dir = Path(__file__).resolve().parents[2] / "standalone"
        files = list(standalone_dir.glob("*.py"))
        self.assertTrue(files)
        for f in files:
            self.assertNotIn("torch_check", f.read_text(encoding="utf-8"),
                            msg=str(f))

    def test_torch_check_file_deleted(self):
        standalone_dir = Path(__file__).resolve().parents[2] / "standalone"
        self.assertFalse((standalone_dir / "torch_check.py").exists())


if __name__ == "__main__":
    unittest.main()
