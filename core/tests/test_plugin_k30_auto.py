"""K30c: "A" (Auto) button in section chapter headers.

Offscreen tests with a Fake client (no worker, no torch):
(a) ``auto_<key>`` exists for every ``AUTO_SECTIONS`` key, not for reference/hsl/lut
(b) click ``auto_tone`` → exactly one ``/auto`` (section="tone") → exposure shows
    "1,25" → exactly one preview
(c) click ``auto_curves`` → the curves editor gets the returned points
(d) the Fake throws → no crash, the status line shows the error
(e) header buttons (auto/help/ON/↺) don't overlap and the title is not truncated,
    in V and H at 1366×720 and 2560×1392
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


def _overlaps(a: tuple, b: tuple) -> bool:
    """a, b = (x, y, w, h). True if the rects overlap with positive area."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class AutoFake:
    """Callable client ``(port, path, payload) -> dict``; records /auto and /preview calls."""

    def __init__(self, auto_response: dict, throw: bool = False, png: str = ""):
        self.auto_calls: list[dict] = []
        self.preview_calls: list[dict] = []
        self._auto_response = auto_response
        self._throw = throw
        self._png = png

    def __call__(self, port: int, path: str, payload: dict) -> dict:
        if path == "/auto":
            self.auto_calls.append(payload)
            if self._throw:
                raise RuntimeError("boom: auto failed")
            return self._auto_response
        if path == "/preview":
            self.preview_calls.append(payload)
            return {"image": self._png, "histogram": {}}
        return {}


class _Base(unittest.TestCase):
    """Offscreen QApplication + shared temp PNG for non-null preview images."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls._png = str(Path(os.environ.get("TEMP", "/tmp")) / "k30c_preview.png")
        QImage(100, 100, QImage.Format.Format_RGB32).save(cls._png)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()

    def _make_dialog(self, fake: AutoFake, W: int, H: int) -> RefGradeDialog:
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

    def _slider_text(self, dlg: RefGradeDialog, name: str) -> str | None:
        for _key, (_h, rows) in dlg.sections.items():
            for r in rows:
                if r.name == name:
                    return r.value_label.text()
        return None


class TestAutoButtonsExist(_Base):
    """(a) auto_<key> for every AUTO_SECTIONS key; none for reference/hsl/lut."""

    def test_auto_buttons(self):
        fake = AutoFake({}, png=self._png)
        dlg = self._make_dialog(fake, 1920, 1032)
        try:
            for key in AUTO_SECTIONS:
                header = dlg.sections[key][0]
                self.assertTrue(header.has_auto, f"{key} should have an auto button")
                self.assertFalse(header.auto_btn.isHidden())
                self.assertEqual(header.auto_btn.objectName(), f"auto_{key}")
            for key in ("reference", "hsl", "lut"):
                header = dlg.sections[key][0]
                # 2026-10-11: the button exists hidden (keeps the column aligned)
                self.assertFalse(header.has_auto, f"{key} must NOT have an auto button")
                self.assertTrue(header.auto_btn.isHidden())
        finally:
            self._close(dlg)


class TestAutoTone(_Base):
    """(b) auto_tone → one /auto (section="tone") → exposure "1,25" → one preview."""

    def test_auto_tone(self):
        fake = AutoFake({"values": {"exposure": 1.25}}, png=self._png)
        dlg = self._make_dialog(fake, 1920, 1032)
        try:
            n0 = _wait_stable(self.app, lambda: len(fake.preview_calls))
            self.assertGreaterEqual(n0, 1, "setup should send at least one preview")
            dlg.sections["tone"][0].auto_btn.click()
            self.assertTrue(_wait(self.app, lambda: len(fake.auto_calls) >= 1),
                            "no /auto request was sent")
            self.assertEqual(len(fake.auto_calls), 1, "exactly one /auto request")
            self.assertEqual(fake.auto_calls[0]["section"], "tone")
            self.assertTrue(_wait(self.app, lambda: self._slider_text(dlg, "exposure") == "1,25"),
                            f"exposure not '1,25', got {self._slider_text(dlg, 'exposure')!r}")
            self.assertTrue(_wait(self.app, lambda: len(fake.preview_calls) >= n0 + 1),
                            "no preview after auto")
            self.assertEqual(len(fake.preview_calls), n0 + 1, "exactly one preview after auto")
        finally:
            self._close(dlg)


class TestAutoCurves(_Base):
    """(c) auto_curves → the curves editor gets the returned points."""

    def test_auto_curves(self):
        pts = [[0, 0], [0.3, 0.2], [1, 1]]
        fake = AutoFake({"curves": {"rgb": pts}}, png=self._png)
        dlg = self._make_dialog(fake, 1920, 1032)
        try:
            _wait_stable(self.app, lambda: len(fake.preview_calls))
            dlg.sections["curves"][0].auto_btn.click()
            self.assertTrue(_wait(self.app, lambda: len(fake.auto_calls) >= 1),
                            "no /auto request was sent")
            self.assertEqual(fake.auto_calls[0]["section"], "curves")
            want = [[0.0, 0.0], [0.3, 0.2], [1.0, 1.0]]
            self.assertTrue(_wait(self.app, lambda: dlg.curves_editor.get_curve("rgb") == want),
                            f"curves editor not updated, got {dlg.curves_editor.get_curve('rgb')}")
            self.assertEqual(dlg.curves_editor.get_curve("rgb"), want)
        finally:
            self._close(dlg)


class TestAutoError(_Base):
    """(d) the Fake throws → no crash, the status line shows the error."""

    def test_auto_error(self):
        fake = AutoFake({}, throw=True, png=self._png)
        dlg = self._make_dialog(fake, 1920, 1032)
        try:
            _wait_stable(self.app, lambda: len(fake.preview_calls))
            dlg.sections["tone"][0].auto_btn.click()
            self.assertTrue(_wait(self.app, lambda: dlg.view.mark.startswith("ERROR ·")),
                            f"no error status, got {dlg.view.mark!r}")
            self.assertTrue(dlg.view.mark.startswith("ERROR ·"))
            self.assertIn("boom", dlg.view.mark)
        finally:
            self._close(dlg)


class TestHeaderLayout(_Base):
    """(e) header buttons don't overlap and the title is not truncated (V/H, 2 sizes)."""

    def _check_header(self, dlg: RefGradeDialog, key: str) -> None:
        header = dlg.sections[key][0]
        self.assertIsNotNone(header.auto_btn, f"{key} should have an auto button")
        rects = {
            "auto": header.auto_btn.geometry(),
            "help": header.help_btn.geometry(),
            "on": header.on.geometry(),
            "undo": header.undo.geometry(),
        }
        names = list(rects)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = rects[names[i]]
                b = rects[names[j]]
                self.assertFalse(
                    _overlaps((a.x(), a.y(), a.width(), a.height()),
                              (b.x(), b.y(), b.width(), b.height())),
                    f"{key}: '{names[i]}' overlaps '{names[j]}'")
        title = header._title
        fm = title.fontMetrics()
        self.assertLessEqual(fm.horizontalAdvance(title.text()),
                             title.contentsRect().width(),
                             f"{key}: title '{title.text()}' truncated")

    def _check_all(self, dlg: RefGradeDialog, mode: str, W: int, H: int) -> None:
        dlg._set_layout_mode(mode)
        self.app.processEvents()
        for key in AUTO_SECTIONS:
            self._check_header(dlg, key)

    def test_v_1366(self):
        fake = AutoFake({}, png=self._png)
        dlg = self._make_dialog(fake, 1366, 720)
        try:
            self._check_all(dlg, "V", 1366, 720)
        finally:
            self._close(dlg)

    def test_h_1366(self):
        fake = AutoFake({}, png=self._png)
        dlg = self._make_dialog(fake, 1366, 720)
        try:
            self._check_all(dlg, "H", 1366, 720)
        finally:
            self._close(dlg)

    def test_v_2560(self):
        fake = AutoFake({}, png=self._png)
        dlg = self._make_dialog(fake, 2560, 1392)
        try:
            self._check_all(dlg, "V", 2560, 1392)
        finally:
            self._close(dlg)

    def test_h_2560(self):
        fake = AutoFake({}, png=self._png)
        dlg = self._make_dialog(fake, 2560, 1392)
        try:
            self._check_all(dlg, "H", 2560, 1392)
        finally:
            self._close(dlg)


if __name__ == "__main__":
    unittest.main()
