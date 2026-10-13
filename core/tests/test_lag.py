"""K29c: Drag halbiert max_side; nach Loslassen volle Auflösung."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable
from urllib.request import urlopen, Request

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
PYTHON = sys.executable

_PLUGIN_DIR = PROJECT_DIR / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade import worker_client  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class FakeClient:
    """Records calls; returns a fixed preview response."""

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
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
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

    def count(self, path: str) -> int:
        return sum(1 for p, _ in self.calls if p == path)

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path:
                return payload
        return None


class TestDragHalfResolution(unittest.TestCase):
    """K29c: beim Ziehen max_side == volle // 2, nach Loslassen == volle."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_lag_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
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
        # Let the initial preview settle
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        # Drain any in-flight requests
        self._drain()

    def _drain(self) -> None:
        """Process events until no new preview arrives within 100 ms."""
        deadline = time.monotonic() + 2.0
        last_n, last_t = self.fake.count("/preview"), time.monotonic()
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
            n = self.fake.count("/preview")
            if n > last_n:
                last_n, last_t = n, time.monotonic()
            elif time.monotonic() - last_t > 0.1:
                break

    def test_drag_half_resolution(self):
        """During drag: max_side is half; after release: full."""
        full_side = self.dlg._max_side()
        self.assertGreater(full_side, 0)

        # Start drag
        self.dlg._on_drag_started()
        self.assertTrue(self.dlg._dragging)

        # Trigger a preview during drag
        self.dlg._trigger_preview()
        _wait(self.app, lambda: self.fake.count("/preview") >= 2, timeout_ms=2000)
        self._drain()

        drag_payload = self.fake.last("/preview")
        self.assertIsNotNone(drag_payload, "no /preview received during drag")
        drag_max = drag_payload["max_side"]
        self.assertEqual(drag_max, full_side // 2,
                         f"during drag: max_side={drag_max}, expected {full_side // 2}")

        # Stop drag → triggers a fresh preview at full resolution
        self.dlg._on_drag_stopped()
        self.assertFalse(self.dlg._dragging)
        _wait(self.app, lambda: self.fake.count("/preview") >= 3, timeout_ms=2000)
        self._drain()

        release_payload = self.fake.last("/preview")
        self.assertIsNotNone(release_payload, "no /preview received after release")
        release_max = release_payload["max_side"]
        self.assertEqual(release_max, full_side,
                         f"after release: max_side={release_max}, expected {full_side}")

    def test_max_side_method_direct(self):
        """_max_side() returns half when _dragging is True, full when False."""
        full = self.dlg._max_side()
        self.dlg._dragging = True
        half = self.dlg._max_side()
        self.assertEqual(half, full // 2)
        self.dlg._dragging = False
        self.assertEqual(self.dlg._max_side(), full)


if __name__ == "__main__":
    unittest.main()
