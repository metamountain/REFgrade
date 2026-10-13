"""K28c: PAIR-Ausrichtungshinweis in der Methodenzeile (pair_shift > 5)."""

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

W, H = 1920, 1152


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class TestPairShiftHint(unittest.TestCase):
    """pair_shift > 5 → gedämpfter Hinweis in der Methodenzeile; sonst keiner."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        client = lambda port, path, payload: {}
        source = {"session": "test", "width": 400, "height": 300}
        dlg = RefGradeDialog(img, client=client, source=source)
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        return dlg

    def _preview_result(self, **extra) -> dict:
        img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "k28c_pair.png")
        QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
        result = {"image": img_path, "histogram": {}, "ms": 5.0}
        result.update(extra)
        return result

    def test_hint_visible_at_shift_14(self):
        """pair_shift=14 → Hinweistext mit Wert gesetzt und Label nicht hidden.

        isVisible() taugt nicht: die Reference-Sektion ist standardmäßig OFF
        (K25c), dann ist die ganze Methodenzeile unsichtbar."""
        dlg = self._make_dialog()
        dlg._apply_preview(self._preview_result(pair_shift=14.0))
        self.assertFalse(dlg.pair_hint.isHidden(), "Hinweis bei shift 14 nicht sichtbar")
        self.assertIn("do not line up", dlg.pair_hint.text())
        self.assertIn("shift ≈ 14 px", dlg.pair_hint.text())
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_no_hint_at_shift_2(self):
        """pair_shift=2 → kein Hinweis."""
        dlg = self._make_dialog()
        dlg._apply_preview(self._preview_result(pair_shift=2.0))
        self.assertTrue(dlg.pair_hint.isHidden(), "Hinweis bei shift 2 sichtbar")
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_hint_hidden_without_pair_shift(self):
        """Kein pair_shift im Ergebnis (andere Methode) → kein Hinweis."""
        dlg = self._make_dialog()
        dlg._apply_preview(self._preview_result())
        self.assertTrue(dlg.pair_hint.isHidden())
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
