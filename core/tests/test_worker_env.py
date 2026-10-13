"""Regression 2026-10-08 (user, Krita): the worker died at once when started from Krita.

Krita sets PYTHONHOME/PYTHONPATH for its own Python 3.13; inherited by the system Python
3.12 they load the wrong stdlib ("bad magic number in 'encodings'"). The worker must start
with those variables set exactly as Krita sets them.
"""

import os
import socket
import sys
import unittest
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", pkg)

from ref_grade import config, worker_process  # noqa: E402
from slow import slow  # noqa: E402

KRITA = r"C:\Program Files\Krita (x64)"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestWorkerEnv(unittest.TestCase):

    def test_clean_env_drops_python_vars(self):
        env = {"PATH": "x", "PYTHONHOME": "y", "PYTHONPATH": "z", "pythonstartup": "w"}
        self.assertEqual(worker_process.clean_env(env), {"PATH": "x"})

    @slow

    def test_worker_starts_under_krita_python_env(self):
        port = _free_port()
        krita_env = {"PYTHONHOME": KRITA,
                     "PYTHONPATH": os.pathsep.join([KRITA + r"\lib\site-packages",
                                                    KRITA + r"\lib\krita-python-libs"])}
        with mock.patch.dict(os.environ, krita_env):
            try:
                state = worker_process.ensure(port, config.python_exe(), config.CORE, timeout=60)
                self.assertEqual(state, "started")
            finally:
                worker_process.stop(port)


if __name__ == "__main__":
    unittest.main()
