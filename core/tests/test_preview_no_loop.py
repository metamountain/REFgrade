"""Regression (2026-10-10): clarity on + one change while the lut stage runs must not start an
endless lut-stage loop; the full stage has to arrive and then the dialog goes quiet."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
_PLUGIN = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
if "ref_grade" not in sys.modules:
    _pkg = types.ModuleType("ref_grade")
    _pkg.__path__ = [str(_PLUGIN)]
    sys.modules["ref_grade"] = _pkg

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv)
from ref_grade.dialog import RefGradeDialog  # noqa: E402

DELAY_S = 0.25            # slower than the 120 ms full-stage timer, as on a real 2560 preview


class _SlowClient:
    def __init__(self, tmp: Path):
        self.tmp, self.calls = tmp, []

    def _png(self, name: str) -> str:
        img = QImage(64, 64, QImage.Format.Format_RGB32)
        img.fill(QColor(120, 120, 120))
        img.save(str(self.tmp / name))
        return str(self.tmp / name)

    def __call__(self, port, path, payload=None):
        if path == "/preview":
            self.calls.append(payload.get("stage"))
            time.sleep(DELAY_S)
            return {"image": self._png("r.png"), "ms": DELAY_S * 1000,
                    "histogram": {"source": [0.01] * 64, "result": [0.01] * 64}}
        if path == "/references":
            return {"items": []}
        if path == "/thumb":
            return {"thumb": self._png("t.png")}
        return {}


class PreviewNoLoopTest(unittest.TestCase):
    def test_change_during_lut_stage_ends_with_full_stage(self):
        tmp = Path(tempfile.mkdtemp())
        client = _SlowClient(tmp)
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        src = {"raw": str(tmp / "s.raw"), "width": 200, "height": 150, "depth": "U8",
               "x": 0, "y": 0, "session": "x"}
        d = RefGradeDialog(img, None, client=client, source=src)
        d._session_id = "x"
        d.resize(1600, 900)
        d.show()
        APP.processEvents()
        d._set_slider_value("clarity", 0.5)
        client.calls.clear()
        d._schedule_preview()
        t0, poked = time.time(), False
        while time.time() - t0 < 4.0:
            APP.processEvents()
            time.sleep(0.01)
            if not poked and client.calls and time.time() - t0 > 0.33:
                d._schedule_preview()            # one change inside the full-stage window
                poked = True
        d.close()
        self.assertIn("full", client.calls, f"full stage never sent: {client.calls}")
        self.assertLessEqual(len(client.calls), 6, f"re-render loop: {client.calls}")


if __name__ == "__main__":
    unittest.main()
