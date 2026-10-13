"""Tooltip-Ereignis im Bild darf nie werfen (2026-10-13: QHelpEvent.position() existiert nicht —
jede Mausbewegung über dem Bild hätte eine Ausnahme ausgelöst; PyQt6 kann Krita dabei abbrechen)."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules.setdefault("ref_grade", _pkg)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QEvent, QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QHelpEvent, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.widgets import ImageView  # noqa: E402


class TestTooltipEvent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_help_event_over_button_and_image(self):
        v = ImageView()
        v.resize(600, 400)
        img = QImage(600, 400, QImage.Format.Format_RGB32)
        img.fill(QColor(120, 120, 120))
        v.set_image(img)
        v.show()
        self.app.processEvents()
        v.grab()                                   # paintEvent setzt die Knopf-Rechtecke
        c = v._clip_btn_rect.center()
        for pt in (QPoint(int(c.x()), int(c.y())), QPoint(50, 50)):
            ev = QHelpEvent(QEvent.Type.ToolTip, pt, v.mapToGlobal(pt))
            self.assertTrue(v.event(ev))           # wirft nicht, wird behandelt
        v.close()


if __name__ == "__main__":
    unittest.main()
