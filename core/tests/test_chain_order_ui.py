"""K35a: Drag & Drop chain order — panel order = chain order, QSettings persist, reset."""

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
# Use a unique namespace so our QSettings don't leak into other test modules
_CHAIN_SETTINGS = f"RefGradeTests_chain_{os.getpid()}"
os.environ["REFGRADE_SETTINGS"] = _CHAIN_SETTINGS

from PyQt6.QtCore import QPoint, QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.dlg_common import CHAIN_ORDER  # noqa: E402

W, H = 2560, 1392


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class FakeClient:
    """Records /preview payloads; returns a dummy success."""

    def __init__(self):
        self.payloads: list[dict] = []

    def __call__(self, port: int, path: str, payload: dict) -> dict:
        if path == "/preview":
            self.payloads.append(payload)
        return {"ok": True}


class ChainOrderUiBase(unittest.TestCase):
    """Base: offscreen dialog with fake client at 2560×1392."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        # Restore the original REFGRADE_SETTINGS for subsequent test modules
        os.environ["REFGRADE_SETTINGS"] = f"RefGradeTests_{os.getpid()}"

    def setUp(self):
        os.environ["REFGRADE_SETTINGS"] = _CHAIN_SETTINGS
        QSettings("RefGrade", _CHAIN_SETTINGS).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.fake = FakeClient()
        self.dlg = RefGradeDialog(
            img, client=self.fake, source={"session": "test", "width": 400, "height": 300}
        )
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()
        QSettings("RefGrade", _CHAIN_SETTINGS).clear()
        os.environ["REFGRADE_SETTINGS"] = f"RefGradeTests_{os.getpid()}"

    def _trigger_and_wait_preview(self, timeout_ms: int = 3000) -> dict:
        """Trigger a preview and wait for the fake client to receive it."""
        self.dlg._schedule_preview()
        start = len(self.fake.payloads)
        deadline = time.time() + timeout_ms / 1000
        while time.time() < deadline:
            self.app.processEvents()
            if len(self.fake.payloads) > start:
                return self.fake.payloads[-1]
            time.sleep(0.01)
        self.fail("No /preview payload received within timeout")

    def _drag_header(self, header, target_header) -> None:
        """Simulate dragging a section header title over another."""
        src = header._title
        dst = target_header._title
        # Positions in src's local coordinate system
        src_center = QPoint(src.width() // 2, src.height() // 2)
        dst_global = dst.mapToGlobal(QPoint(dst.width() // 2, dst.height() // 2))
        dst_in_src = src.mapFromGlobal(dst_global)
        QTest.mousePress(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, src_center)
        # Move > 6px to trigger drag, then to target position
        QTest.mouseMove(src, QPoint(src_center.x() + 10, src_center.y()))
        QTest.mouseMove(src, dst_in_src)
        self.app.processEvents()
        QTest.mouseRelease(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, dst_in_src)
        self.app.processEvents()


class TestDefaultOrder(ChainOrderUiBase):
    """1. Without reordering, /preview payload has mix.order == list(CHAIN_ORDER)."""

    def test_default_order_in_preview(self):
        payload = self._trigger_and_wait_preview()
        self.assertEqual(payload["mix"]["order"], list(CHAIN_ORDER))


class TestDragReorder(ChainOrderUiBase):
    """2. Drag WHEELS over TONE → panel order has wheels before tone; /preview reflects it."""

    def test_drag_wheels_over_tone(self):
        wheels_h = self.dlg.sections["wheels"][0]
        tone_h = self.dlg.sections["tone"][0]
        self._drag_header(wheels_h, tone_h)
        # Check panel order
        self.assertIn("wheels", self.dlg._panel_order)
        self.assertIn("tone", self.dlg._panel_order)
        self.assertLess(
            self.dlg._panel_order.index("wheels"),
            self.dlg._panel_order.index("tone"),
            f"Expected wheels before tone in {self.dlg._panel_order}",
        )
        # Check /preview payload
        payload = self._trigger_and_wait_preview()
        order = payload["mix"]["order"]
        self.assertLess(order.index("wheels"), order.index("tone"))


class TestPersistAndReset(ChainOrderUiBase):
    """3. Close/reopen → order restored; context menu reset → default."""

    def test_persist_and_reset(self):
        # Reorder: move wheels before tone
        wheels_h = self.dlg.sections["wheels"][0]
        tone_h = self.dlg.sections["tone"][0]
        self._drag_header(wheels_h, tone_h)
        new_order = list(self.dlg._panel_order)
        self.assertNotEqual(new_order, list(CHAIN_ORDER) + ["color_replace"])

        # Close and reopen
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg2 = RefGradeDialog(
            img, client=self.fake, source={"session": "test2", "width": 400, "height": 300}
        )
        self.dlg2.resize(W, H)
        self.dlg2.show()
        _wait(self.app, lambda: self.dlg2.width() >= W - 20)
        self.app.processEvents()

        # Order should be restored
        self.assertEqual(self.dlg2._panel_order, new_order)

        # Reset via context menu signal
        self.dlg2._on_reset_chain_order()
        self.assertEqual(self.dlg2._panel_order, list(CHAIN_ORDER) + ["color_replace"])
        self.assertEqual(self.dlg2._chain_order, list(CHAIN_ORDER))

        self.dlg2.close()
        self.dlg2.deleteLater()
        self.app.processEvents()


class TestClickCollapse(ChainOrderUiBase):
    """4. Click without movement on title still collapses/expands."""

    def test_click_collapses(self):
        tone_h = self.dlg.sections["tone"][0]
        # Ensure the section starts expanded for a clean test
        tone_h.set_collapsed(False, emit=False)
        self.assertFalse(tone_h.is_collapsed())
        # Simulate a click (press + release at same position, no move)
        src = tone_h._title
        pos = src.mapToGlobal(QPoint(src.width() // 2, src.height() // 2))
        QTest.mousePress(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        QTest.mouseRelease(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        self.app.processEvents()
        self.assertTrue(tone_h.is_collapsed(), "Click without movement should collapse the section")

        # Click again to expand
        QTest.mousePress(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        QTest.mouseRelease(src, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        self.app.processEvents()
        self.assertFalse(tone_h.is_collapsed(), "Second click should expand the section")


class TestDragToTop(ChainOrderUiBase):
    """5. Drag the last scroll section to the top (deterministic, direct calls)."""

    def _drag_last_to_top(self) -> str:
        """Drag the current last scroll section above the first section's middle."""
        scroll_keys = [k for k in self.dlg._panel_order if k != "wheels"]
        last_key = scroll_keys[-1]
        first_w = self.dlg._scroll_widgets[scroll_keys[0]]
        first_top_y = first_w.mapToGlobal(first_w.rect().topLeft()).y()
        self.dlg._on_section_drag_started(last_key)
        self.dlg._on_section_drag_moved(last_key, QPoint(0, first_top_y + 1))
        self.dlg._on_section_drag_ended(last_key, True)
        return last_key

    def test_drag_last_to_first(self):
        last_key = self._drag_last_to_top()
        self.assertEqual(self.dlg._panel_order.index(last_key), 0)

    def test_two_reorders_in_a_row(self):
        first_key = self._drag_last_to_top()
        second_key = self._drag_last_to_top()
        # Both dragged keys end up at the top, newest first.
        self.assertEqual(self.dlg._panel_order.index(second_key), 0)
        self.assertEqual(self.dlg._panel_order.index(first_key), 1)
