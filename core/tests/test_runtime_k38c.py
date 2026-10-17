"""Tests für Paket K38c (Installationsort, vorhandenes Python) – ohne Netz."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from ref_grade import config, runtime  # noqa: E402


class _Base(unittest.TestCase):
    """Setzt REFGRADE_CONFIG auf eine Temp-Datei; REFGRADE_RUNTIME bleibt aus."""

    def setUp(self):
        self._old_cfg = os.environ.get("REFGRADE_CONFIG")
        self._old_rt = os.environ.get("REFGRADE_RUNTIME")
        self._tmp = tempfile.TemporaryDirectory()
        # REFGRADE_RUNTIME absichtlich löschen → der config.json-Pfad wird geprüft.
        os.environ.pop("REFGRADE_RUNTIME", None)
        self.cfg_path = Path(self._tmp.name) / "config.json"
        os.environ["REFGRADE_CONFIG"] = str(self.cfg_path)
        runtime.reset_cache()

    def tearDown(self):
        os.environ.pop("REFGRADE_CONFIG", None)
        if self._old_rt is None:
            os.environ.pop("REFGRADE_RUNTIME", None)
        else:
            os.environ["REFGRADE_RUNTIME"] = self._old_rt
        self._tmp.cleanup()
        runtime.reset_cache()

    def _localappdata_runtime(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA", "")) / "RefGrade" / "runtime"

    def _settings(self) -> dict:
        return json.loads(self.cfg_path.read_text(encoding="utf-8"))


class TestRuntimeDir(_Base):

    def test_set_and_clear(self):
        # Fremder Schlüssel, der später erhalten bleiben muss.
        config.write_settings({"theme": "dark"})
        runtime.set_runtime_dir("X:/rt")
        self.assertEqual(runtime.runtime_dir(), Path("X:/rt"))
        data = self._settings()
        self.assertEqual(data.get("runtime_dir"), "X:/rt")
        self.assertEqual(data.get("theme"), "dark")
        # None → wieder der LOCALAPPDATA-Pfad, fremder Schlüssel bleibt.
        runtime.set_runtime_dir(None)
        self.assertEqual(runtime.runtime_dir(), self._localappdata_runtime())
        data = self._settings()
        self.assertNotIn("runtime_dir", data)
        self.assertEqual(data.get("theme"), "dark")


class TestCheckTarget(_Base):

    def test_ok(self):
        self.assertEqual(runtime.check_target(self._tmp.name, "cpu"), "")

    def test_low_space(self):
        with patch.object(runtime, "free_mb", return_value=100):
            msg = runtime.check_target(self._tmp.name, "cpu")
        self.assertIn("space", msg)

    def test_non_ascii(self):
        bad = str(Path(self._tmp.name) / "pfad_mit_ü")
        msg = runtime.check_target(bad, "cpu")
        self.assertIn("ASCII", msg)


class TestUseExisting(_Base):

    def test_true_stores(self):
        with patch.object(runtime, "python_ok", return_value=True):
            result = runtime.use_existing("C:/py/python.exe")
        self.assertEqual(result, "")
        self.assertEqual(self._settings().get("python_exe"), "C:/py/python.exe")

    def test_false_message(self):
        with patch.object(runtime, "python_ok", return_value=False):
            result = runtime.use_existing("C:/py/python.exe")
        self.assertEqual(
            result, "This Python cannot import torch, numpy, kornia and PIL.")


if __name__ == "__main__":
    unittest.main()
