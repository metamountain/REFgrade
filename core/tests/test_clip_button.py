"""C-Button in ImageView: Klick → Signal, Toggle → _clip_view, Fake-Client sieht clip_view."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent

_PLUGIN_DIR = PROJECT_DIR / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.widgets import ImageView  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return condition()


class FakeClient:
    """Records calls; returns a fixed preview response with clip data."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        png = tmp / "resp.png"
        img = QImage(64, 64, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 150, 200))
        img.save(str(png), "PNG")
        self._png = str(png)

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64},
                "clip": {"r_hi": 0.025, "g_hi": 0.01, "b_hi": 0.0,
                         "r_lo": 0.012, "g_lo": 0.0, "b_lo": 0.0},
                "ms": 1.0}
        elif path == "/references":
            return {"items": []}
        elif path == "/crop":
            return {"image": self._png, "scale": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/apply":
            raw = self._tmp / "result.raw"
            raw.write_bytes(b"\x00" * 1024)
            return {"raw": str(raw), "ms": 1.0}
        elif path == "/cube":
            d = self._tmp / "luts" / "RefGrade"
            d.mkdir(parents=True, exist_ok=True)
            p = d / "saved.cube"
            p.write_text('TITLE "RefGrade saved"\n')
            return {"path": str(p)}
        elif path == "/prerender":
            return {"image": self._png}
        elif path == "/thumb":
            return {"thumb": self._png}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path:
                return payload
        return None


class TestClipButtonSignal(unittest.TestCase):
    """Klick in _clip_btn_rect → Signal genau einmal."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_click_emits_signal_once(self):
        v = ImageView()
        v.resize(600, 400)
        img = QImage(64, 64, QImage.Format.Format_RGB32)
        img.fill(QColor(120, 120, 120))
        v.set_image(img)
        v.show()
        self.app.processEvents()
        # paintEvent muss laufen, damit _clip_btn_rect gesetzt ist
        v.update()
        self.app.processEvents()
        self.assertFalse(v._clip_btn_rect.isEmpty(), "clip_btn_rect muss nach paint gesetzt sein")
        signals = []
        v.clip_button_clicked.connect(lambda: signals.append(1))
        pos = QPoint(int(v._clip_btn_rect.center().x()), int(v._clip_btn_rect.center().y()))
        QTest.mouseClick(v, Qt.MouseButton.LeftButton, pos=pos)
        self.app.processEvents()
        self.assertEqual(len(signals), 1, "Signal genau einmal")
        v.close()


class TestClipToggle(unittest.TestCase):
    """Zweimal umschalten → _clip_view False; Fake-Client sieht clip_view True, dann False."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_clipbtn_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        self.fake = FakeClient(self.tmp)
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(
            img, None,
            client=self.fake,
            source={"raw": str(self.tmp / "s.raw"), "width": 200, "height": 150,
                    "depth": "U8", "x": 0, "y": 0, "session": "fake_session_id"},
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        _wait(self.app, lambda: self.fake.last("/preview") is not None)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

    def test_toggle_twice_back_to_false(self):
        self.assertFalse(self.dlg._clip_view, "initial: False")
        self.dlg._toggle_clip_view()
        self.assertTrue(self.dlg._clip_view, "nach 1. Toggle: True")
        self.dlg._toggle_clip_view()
        self.assertFalse(self.dlg._clip_view, "nach 2. Toggle: False")

    def test_fake_client_sees_clip_view(self):
        # Erster Toggle → True
        self.dlg._toggle_clip_view()
        _wait(self.app, lambda: self.fake.last("/preview") is not None
              and self.fake.last("/preview").get("clip_view") is True)
        payload = self.fake.last("/preview")
        self.assertTrue(payload.get("clip_view"), "Fake-Client sieht clip_view=True")
        # Zweiter Toggle → False
        self.dlg._toggle_clip_view()
        _wait(self.app, lambda: self.fake.last("/preview") is not None
              and self.fake.last("/preview").get("clip_view") is False)
        payload = self.fake.last("/preview")
        self.assertFalse(payload.get("clip_view"), "Fake-Client sieht clip_view=False")

    def test_clip_text_set(self):
        """Nach Preview mit clip-Daten: view._clip_text enthält ▲ und ▼."""
        _wait(self.app, lambda: self.dlg.view._clip_text != "")
        self.assertIn("▲", self.dlg.view._clip_text)
        self.assertIn("▼", self.dlg.view._clip_text)


if __name__ == "__main__":
    unittest.main()
