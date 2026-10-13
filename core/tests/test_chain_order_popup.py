"""Chain order popup: ⇅ button in MASTER row opens a draggable list; reorder updates _chain_order.

Offscreen tests:
(a) chain_order_btn exists and is visible
(b) click → popup with 8 items in current (CHAIN_ORDER) order
(c) move WHEELS before TONE via model().moveRow → dlg._chain_order has wheels before tone
(d) Reset → back to CHAIN_ORDER
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

from PyQt6.QtCore import QSettings, QModelIndex  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QListWidget, QPushButton  # noqa: E402

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


class FakeClient:
    def __init__(self, png: str = ""):
        self.preview_calls: list[dict] = []
        self._png = png

    def __call__(self, port: int, path: str, payload: dict) -> dict:
        if path == "/preview":
            self.preview_calls.append(payload)
            return {"image": self._png, "histogram": {}}
        return {}

    def real_previews(self) -> list[dict]:
        return [p for p in self.preview_calls if "stage" in p]


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls._png = str(Path(os.environ.get("TEMP", "/tmp")) / "chain_order_popup_preview.png")
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
        if dlg._chain_order_popup is not None:
            dlg._chain_order_popup.close()
            dlg._chain_order_popup = None
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def _popup_list(self, dlg: RefGradeDialog) -> QListWidget | None:
        """Return the chain_order_list from the open popup, or None."""
        if dlg._chain_order_popup is None:
            return None
        return dlg._chain_order_popup.findChild(QListWidget, "chain_order_list")


class TestChainOrderButton(_Base):
    """(a) chain_order_btn exists and is visible."""

    def test_button_exists_and_visible(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            btn = dlg.findChild(QPushButton, "chain_order_btn")
            self.assertIsNotNone(btn, "chain_order_btn not found")
            self.assertTrue(btn.isVisible(), "chain_order_btn is not visible")
        finally:
            self._close(dlg)


class TestChainOrderPopupOpens(_Base):
    """(b) click → popup with 8 items in current (CHAIN_ORDER) order."""

    def test_popup_shows_8_items_in_order(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            btn = dlg.findChild(QPushButton, "chain_order_btn")
            self.assertIsNotNone(btn)
            btn.click()
            self.assertTrue(_wait(self.app, lambda: dlg._chain_order_popup is not None
                                  and dlg._chain_order_popup.isVisible()),
                            "popup did not open")
            lst = self._popup_list(dlg)
            self.assertIsNotNone(lst, "chain_order_list not found in popup")
            self.assertEqual(lst.count(), 8, f"expected 8 items, got {lst.count()}")
            items = [lst.item(i).text() for i in range(lst.count())]
            self.assertEqual(items, [k.upper() for k in CHAIN_ORDER])
        finally:
            self._close(dlg)


class TestChainOrderMoveRow(_Base):
    """(c) move WHEELS before TONE via model().moveRow → dlg._chain_order has wheels before tone."""

    def test_move_wheels_before_tone(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            btn = dlg.findChild(QPushButton, "chain_order_btn")
            self.assertIsNotNone(btn)
            btn.click()
            self.assertTrue(_wait(self.app, lambda: dlg._chain_order_popup is not None
                                  and dlg._chain_order_popup.isVisible()),
                            "popup did not open")
            lst = self._popup_list(dlg)
            self.assertIsNotNone(lst)
            # CHAIN_ORDER = ("reference", "color", "tone", "rgb", "curves", "wheels", "hsl", "lut")
            # wheels is at index 5, tone is at index 2 → move row 5 to position 2
            model = lst.model()
            ok = model.moveRow(QModelIndex(), 5, QModelIndex(), 2)
            self.assertTrue(ok, "moveRow failed")
            self.app.processEvents()
            # Verify dlg._chain_order was updated: wheels before tone
            idx_wheels = dlg._chain_order.index("wheels")
            idx_tone = dlg._chain_order.index("tone")
            self.assertLess(idx_wheels, idx_tone,
                            f"wheels ({idx_wheels}) should be before tone ({idx_tone})")
        finally:
            self._close(dlg)


class TestChainOrderReset(_Base):
    """(d) Reset → back to CHAIN_ORDER."""

    def test_reset_restores_default(self):
        fake = FakeClient(png=self._png)
        dlg = self._make_dialog(fake)
        try:
            btn = dlg.findChild(QPushButton, "chain_order_btn")
            self.assertIsNotNone(btn)
            btn.click()
            self.assertTrue(_wait(self.app, lambda: dlg._chain_order_popup is not None
                                  and dlg._chain_order_popup.isVisible()),
                            "popup did not open")
            lst = self._popup_list(dlg)
            self.assertIsNotNone(lst)
            # Move wheels before tone first
            model = lst.model()
            model.moveRow(QModelIndex(), 5, QModelIndex(), 2)
            self.app.processEvents()
            self.assertNotEqual(dlg._chain_order, list(CHAIN_ORDER),
                                "order should have changed before reset")
            # Click Reset
            reset_btn = dlg._chain_order_popup.findChild(QPushButton, "chain_order_reset")
            self.assertIsNotNone(reset_btn, "chain_order_reset not found")
            reset_btn.click()
            self.app.processEvents()
            self.assertEqual(dlg._chain_order, list(CHAIN_ORDER),
                             f"expected {list(CHAIN_ORDER)}, got {dlg._chain_order}")
        finally:
            self._close(dlg)


if __name__ == "__main__":
    unittest.main()
