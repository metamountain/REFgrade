"""Regressionen aus der Debug-Runde 2026-10-17 (Sonnet-Review, von Opus nachgeprüft)."""

from __future__ import annotations

import sys
import types
import unittest
import zipfile
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "core"))
_PLUGIN_DIR = _ROOT / "plugin" / "ref_grade"
if "ref_grade" not in sys.modules:
    _pkg = types.ModuleType("ref_grade")
    _pkg.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["ref_grade"] = _pkg

from refgrade import grade as G  # noqa: E402


class TestRolloffNegativeLuma(unittest.TestCase):
    def test_dark_pixels_stay_dark_with_contrast(self):
        """contrast 1.5 + rolloff 0.3: x=0.1 gives luma < 0 — must not flip to white."""
        x = torch.full((1, 2, 2, 3), 0.1)
        y = G._contrast(x, {"contrast": 1.5, "pivot": 0.5, "highlight_rolloff": 0.3})
        self.assertTrue(torch.all(y <= 0.0 + 1e-6), y.flatten()[:3])

    def test_rolloff_still_compresses_highlights(self):
        x = torch.full((1, 1, 1, 3), 0.95)
        y = G._rolloff(x, 0.5)
        self.assertLess(y.max().item(), 0.95)


class TestUpdater(unittest.TestCase):
    def test_extract_puts_core_under_core(self):
        import tempfile
        from ref_grade import updater
        tmp = Path(tempfile.mkdtemp(dir=_ROOT / "tmp"))
        z = tmp / "repo.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("REFgrade-main/plugin/ref_grade/dialog.py", "x")
            zf.writestr("REFgrade-main/core/worker.py", "w")
            zf.writestr("REFgrade-main/core/refgrade/grade.py", "g")
            zf.writestr("REFgrade-main/core/tests/test_x.py", "t")
            zf.writestr("REFgrade-main/core/refgrade/__pycache__/grade.pyc", "c")
        dest = tmp / "new"
        updater._extract_zip(z, dest)
        self.assertTrue((dest / "dialog.py").exists())
        self.assertTrue((dest / "core" / "worker.py").exists())
        self.assertTrue((dest / "core" / "refgrade" / "grade.py").exists())
        self.assertFalse((dest / "core" / "tests").exists())
        self.assertFalse((dest / "worker.py").exists())
        self.assertFalse(any(p.suffix == ".pyc" for p in dest.rglob("*")))

    def test_batch_is_crlf_and_finds_worker(self):
        import tempfile
        from ref_grade import updater
        tmp = Path(tempfile.mkdtemp(dir=_ROOT / "tmp"))
        b = updater._write_batch(tmp, tmp / "new", tmp / "target", None).read_bytes()
        self.assertEqual(b.count(b"\r\n"), b.count(b"\n"))
        self.assertIn(b"-m worker", b)
        self.assertNotIn(b"wmic", b)

    def test_refuses_in_git_checkout(self):
        from ref_grade import updater
        ok, msg = updater.check_and_update()
        self.assertFalse(ok)
        self.assertIn("git pull", msg)


@unittest.skipUnless((_ROOT / "build" / "make_plugin_zip.py").exists(), "build/ is not in the repo")
class TestZips(unittest.TestCase):
    def test_plugin_zip_has_desktop(self):
        import tempfile
        sys.path.insert(0, str(_ROOT / "build"))
        import make_plugin_zip
        out = Path(tempfile.mkdtemp(dir=_ROOT / "tmp")) / "p.zip"
        make_plugin_zip.build(out)
        names = zipfile.ZipFile(out).namelist()
        self.assertIn("ref_grade.desktop", names)
        self.assertNotIn("ref_grade/installed_version.json", names)


if __name__ == "__main__":
    unittest.main()
