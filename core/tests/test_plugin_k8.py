"""Tests für Paket K8: LUT-Bereich im Fenster (dialog.py mit Fake-Client)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


class FakeClient:
    """Records all calls; returns fixed responses."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        elif path == "/references":
            items = []
            for i in range(3):
                tp = _make_png(self._tmp / f"ref{i}.png", 32, 32, (i * 40, 100, 200))
                items.append({"path": f"/pool/ref{i}.jpg", "name": f"ref{i}",
                              "score": 0.9 - i * 0.05, "thumb": tp})
            return {"items": items}
        elif path == "/crop":
            return {"image": self._png, "scale": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/apply":
            raw = self._tmp / "result.raw"
            raw.write_bytes(b"\x00" * 1024)
            return {"raw": str(raw), "ms": 1.0}
        elif path == "/cube":
            p = self._tmp / "test.cube"
            p.write_text("TITLE \"test\"\n")
            return {"path": str(p)}
        elif path == "/prerender":
            return {"image": self._png}
        elif path == "/thumb":
            return {"thumb": self._png}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def count(self, path: str) -> int:
        return sum(1 for p, _ in self.calls if p == path)

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path:
                return payload
        return None


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _make_source(tmp: Path) -> dict:
    return {
        "raw": str(tmp / "source.raw"),
        "width": 100,
        "height": 80,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "session": "fake_session_id",
    }


def _make_image(w: int = 200, h: int = 150) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(80, 120, 160))
    return img


class K8TestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient + LUT temp folder."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k8_"))
        # LUT folder with b.cube, A.CUBE, c.cube, x.txt
        cls.lut_folder = cls.tmp / "luts"
        cls.lut_folder.mkdir()
        (cls.lut_folder / "b.cube").write_text("TITLE \"b\"\n")
        (cls.lut_folder / "A.CUBE").write_text("TITLE \"A\"\n")
        (cls.lut_folder / "c.cube").write_text("TITLE \"c\"\n")
        (cls.lut_folder / "x.txt").write_text("not a lut\n")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()

        def choose_folder(start: str) -> str | None:
            return str(self.lut_folder)

        self.dlg = RefGradeDialog(
            self.image, None,
            client=self.fake, source=self.source,
            choose_folder=choose_folder,
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestLutFolder(K8TestBase):
    """choose_folder delivers the folder → list = — none —, A, b, c; folder in QSettings."""

    def test_folder_list(self):
        """After FOLDER… click, combo has 4 entries: — none —, A, b, c."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)

        # Click FOLDER…
        self.dlg._lut_folder_btn.click()
        _wait(self.app, lambda: self.dlg._lut_combo.count() == 4, timeout_ms=2000)

        # Check entries
        self.assertEqual(self.dlg._lut_combo.count(), 4)
        self.assertEqual(self.dlg._lut_combo.itemText(0), "— none —")
        self.assertEqual(self.dlg._lut_combo.itemText(1), "A")
        self.assertEqual(self.dlg._lut_combo.itemText(2), "b")
        self.assertEqual(self.dlg._lut_combo.itemText(3), "c")

        # Folder is in QSettings
        from PyQt6.QtCore import QSettings
        settings = QSettings("RefGrade", os.environ.get("REFGRADE_SETTINGS", "RefGrade"))
        saved = settings.value("lut_folder", "")
        self.assertEqual(Path(saved), self.lut_folder)


class TestLutSelection(K8TestBase):
    """Selecting b → section lut ON, exactly one /preview with mix.lut = path of b.cube."""

    def test_select_b(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: any(t.active for t in self.dlg.slots.tiles), timeout_ms=2000)

        # Click FOLDER… to load the list
        self.dlg._lut_folder_btn.click()
        _wait(self.app, lambda: self.dlg._lut_combo.count() == 4, timeout_ms=2000)

        initial = self.fake.count("/preview")

        # Select "b" (index 2)
        self.dlg._lut_combo.setCurrentIndex(2)
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # Section is ON
        self.assertTrue(self.dlg._lut_header.is_on())

        # Exactly one new /preview
        self.assertEqual(self.fake.count("/preview"), initial + 1)

        # mix.lut = path of b.cube
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertIn("lut", payload["mix"])
        self.assertEqual(Path(payload["mix"]["lut"]), self.lut_folder / "b.cube")
        self.assertIn("lut", payload["mix"]["weights"])


class TestLutReset(K8TestBase):
    """↺ → — none —, OFF, mix without lut."""

    def test_reset(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _wait(self.app, lambda: any(t.active for t in self.dlg.slots.tiles), timeout_ms=2000)

        # Load folder and select b
        self.dlg._lut_folder_btn.click()
        _wait(self.app, lambda: self.dlg._lut_combo.count() == 4, timeout_ms=2000)
        self.dlg._lut_combo.setCurrentIndex(2)
        _wait(self.app, lambda: self.dlg._lut_header.is_on(), timeout_ms=2000)

        initial = self.fake.count("/preview")

        # Click reset (↺)
        self.dlg._lut_header.undo.click()
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # Combo is back to — none —
        self.assertEqual(self.dlg._lut_combo.currentIndex(), 0)
        # Section is OFF
        self.assertFalse(self.dlg._lut_header.is_on())

        # mix has no lut
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertNotIn("lut", payload["mix"].get("weights", {}))


class TestLutPersist(K8TestBase):
    """A new dialog reads the saved folder (list has 4 entries)."""

    def test_persist(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)

        # Load folder in first dialog
        self.dlg._lut_folder_btn.click()
        _wait(self.app, lambda: self.dlg._lut_combo.count() == 4, timeout_ms=2000)

        # Close first dialog
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

        # Open a new dialog
        self.dlg2 = RefGradeDialog(
            self.image, None,
            client=self.fake, source=self.source,
            choose_folder=lambda start: str(self.lut_folder),
        )
        self.dlg2.resize(1920, 1080)
        self.dlg2.show()
        _wait(self.app, lambda: self.dlg2.isVisible())

        # The new dialog should have the folder loaded (4 entries)
        self.assertEqual(self.dlg2._lut_combo.count(), 4)
        self.assertEqual(self.dlg2._lut_combo.itemText(0), "— none —")
        self.assertEqual(self.dlg2._lut_combo.itemText(1), "A")
        self.assertEqual(self.dlg2._lut_combo.itemText(2), "b")
        self.assertEqual(self.dlg2._lut_combo.itemText(3), "c")

        self.dlg2.close()
        self.dlg2.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
