"""Chain order state: /preview payload carries mix["order"]; QSettings persistence.

Offscreen tests with a Fake client (no worker, no torch):
(a) without setting: /preview payload mix.order == list(CHAIN_ORDER)
(b) set_chain_order (wheels before tone) → next payload has that order;
    a newly opened dialog loads it from QSettings
(c) set_chain_order(["tone"]) → ValueError
"""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
from slow import slow  # noqa: E402
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
from ref_grade.dlg_common import CHAIN_ORDER  # noqa: E402


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


class FakeClient:
    """Callable client (port, path, payload) -> dict; records /preview calls."""

    def __init__(self, png: str = ""):
        self.preview_calls: list[dict] = []
        self._png = png

    def __call__(self, port: int, path: str, payload: dict) -> dict:
        if path == "/preview":
            self.preview_calls.append(payload)
            return {"image": self._png, "histogram": {}}
        return {}

    def real_previews(self) -> list[dict]:
        """Previews sent by _trigger_preview (have a 'stage' key), not the 'original' one."""
        return [p for p in self.preview_calls if "stage" in p]


class _Base(unittest.TestCase):
    """Offscreen QApplication + shared temp PNG."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls._png = str(Path(os.environ.get("TEMP", "/tmp")) / "chain_order_preview.png")
        QImage(100, 100, QImage.Format.Format_RGB32).save(cls._png)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()

    def _make_dialog(self, fake: FakeClient) -> RefGradeDialog:
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        dlg = RefGradeDialog(img, client=fake,
                             source={"session": "s1", "width": 200, "height": 150})
        dlg.resize(1920, 1032)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= 1900)
        self.app.processEvents()
        return dlg

    def _close(self, dlg: RefGradeDialog) -> None:
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestChainOrderDefault(_Base):
    """(a) without setting: /preview payload mix.order == list(CHAIN_ORDER)."""

    def test_default_order_in_payload(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            self.assertTrue(_wait(self.app, lambda: len(fake.real_previews()) >= 1),
                            "no real preview sent")
            payload = fake.real_previews()[-1]
            self.assertIn("order", payload["mix"])
            self.assertEqual(payload["mix"]["order"], list(CHAIN_ORDER))
        finally:
            self._close(dlg)


class TestChainOrderSetAndPersist(_Base):
    """(b) set_chain_order (wheels before tone) → next payload has that order;
    a newly opened dialog loads it from QSettings."""

    @slow
    def test_set_and_reload(self):
        # Custom order: wheels before tone
        custom = ["reference", "color", "wheels", "tone", "rgb", "curves", "hsl", "lut"]
        self.assertEqual(set(custom), set(CHAIN_ORDER))

        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            n0 = _wait_stable(self.app, lambda: len(fake.real_previews()))
            dlg.set_chain_order(custom)
            self.assertTrue(_wait(self.app, lambda: len(fake.real_previews()) >= n0 + 1),
                            "no preview after set_chain_order")
            payload = fake.real_previews()[-1]
            self.assertEqual(payload["mix"]["order"], custom)
        finally:
            self._close(dlg)

        # New dialog loads the persisted order
        fake2 = FakeClient(png=self._png)
        dlg2 = self._make_dialog(fake2)
        try:
            self.assertTrue(_wait(self.app, lambda: len(fake2.real_previews()) >= 1),
                            "no real preview from second dialog")
            payload2 = fake2.real_previews()[-1]
            self.assertEqual(payload2["mix"]["order"], custom)
        finally:
            self._close(dlg2)


class TestChainOrderInvalid(_Base):
    """(c) set_chain_order(["tone"]) → ValueError."""

    def test_invalid_order_raises(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            _wait_stable(self.app, lambda: len(fake.real_previews()))
            with self.assertRaises(ValueError):
                dlg.set_chain_order(["tone"])
        finally:
            self._close(dlg)


if __name__ == "__main__":
    unittest.main()
