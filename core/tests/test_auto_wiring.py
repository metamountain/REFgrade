"""AUTO sends the same context fields as /preview (references, method, mix).

Offscreen tests with a Fake client (no worker, no torch):
(a) click auto_tone → the /auto payload contains references, method, mix
    and they equal the fields of the last /preview payload
(b) WHEELS and RGB have no visible AUTO button; COLOR, TONE, CURVES do
"""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Callable

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.dlg_common import AUTO_SECTIONS  # noqa: E402


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 5000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _wait_stable(app: QApplication, get_count: Callable[[], int],
                 timeout_ms: int = 3000, stable_ms: int = 120) -> int:
    """Wait until get_count() stays the same for stable_ms; return that count."""
    last = -1
    stable_since: float | None = None
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        c = get_count()
        if c != last:
            last = c
            stable_since = time.time()
        elif stable_since is not None and time.time() - stable_since >= stable_ms / 1000:
            return c
        time.sleep(0.005)
    return last


class WiringFake:
    """Callable client that records /auto and /preview payloads."""

    def __init__(self, auto_response: dict, png: str = ""):
        self.auto_calls: list[dict] = []
        self.preview_calls: list[dict] = []
        self._auto_response = auto_response
        self._png = png

    def __call__(self, port: int, path: str, payload: dict) -> dict:
        if path == "/auto":
            self.auto_calls.append(payload)
            return self._auto_response
        if path == "/preview":
            self.preview_calls.append(payload)
            return {"image": self._png, "histogram": {}}
        return {}


class _Base(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls._png = str(Path(os.environ.get("TEMP", "/tmp")) / "auto_wiring_preview.png")
        QImage(100, 100, QImage.Format.Format_RGB32).save(cls._png)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()

    def _make_dialog(self, fake: WiringFake, W: int = 1920, H: int = 1032) -> RefGradeDialog:
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        dlg = RefGradeDialog(img, client=fake,
                             source={"session": "s1", "width": 200, "height": 150})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        return dlg

    def _close(self, dlg: RefGradeDialog) -> None:
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestAutoPayloadMatchesPreview(_Base):
    """(a) /auto payload contains references, method, mix — equal to the last /preview."""

    def test_auto_tone_payload_fields(self):
        fake = WiringFake({"values": {"exposure": 0.5}}, png=self._png)
        dlg = self._make_dialog(fake)
        try:
            # Wait for the initial preview so we have a baseline
            n0 = _wait_stable(self.app, lambda: len(fake.preview_calls))
            self.assertGreaterEqual(n0, 1, "setup should send at least one preview")

            # Click AUTO in the TONE header
            dlg.sections["tone"][0].auto_btn.click()
            self.assertTrue(_wait(self.app, lambda: len(fake.auto_calls) >= 1),
                            "no /auto request was sent")

            auto_payload = fake.auto_calls[0]
            self.assertEqual(auto_payload["section"], "tone")

            # The three context fields must be present
            for field in ("references", "method", "mix"):
                self.assertIn(field, auto_payload,
                              f"/auto payload missing '{field}'")

            # Compare against the last *regular* /preview (has a "stage" key,
            # not the initial "purpose": "original" identity preview).
            regular = [p for p in fake.preview_calls if "stage" in p]
            self.assertTrue(regular, "no regular /preview was sent")
            last_preview = regular[-1]
            self.assertEqual(auto_payload["references"], last_preview["references"],
                             "references differ between /auto and last /preview")
            self.assertEqual(auto_payload["method"], last_preview["method"],
                             "method differs between /auto and last /preview")
            self.assertEqual(auto_payload["mix"], last_preview["mix"],
                             "mix differs between /auto and last /preview")
        finally:
            self._close(dlg)


class TestAutoButtonVisibility(_Base):
    """(b) WHEELS and RGB have no visible AUTO; COLOR, TONE, CURVES do."""

    def test_visibility(self):
        fake = WiringFake({}, png=self._png)
        dlg = self._make_dialog(fake)
        try:
            # Sections that MUST have a visible AUTO button
            for key in ("color", "tone", "curves"):
                header = dlg.sections[key][0]
                self.assertTrue(header.has_auto, f"{key} should have an auto button")
                self.assertFalse(header.auto_btn.isHidden(),
                                 f"{key} auto button should be visible")

            # Sections that must NOT have a visible AUTO button
            for key in ("wheels", "rgb"):
                header = dlg.sections[key][0]
                self.assertFalse(header.has_auto, f"{key} must NOT have an auto button")
                self.assertTrue(header.auto_btn.isHidden(),
                                f"{key} auto button should be hidden")
        finally:
            self._close(dlg)


if __name__ == "__main__":
    unittest.main()
