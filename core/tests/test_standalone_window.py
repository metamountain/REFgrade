"""Standalone window: layout on a central widget, RefGradeDialog embedded (not a 2nd window).

Regression 2026-10-16: QVBoxLayout(QMainWindow) is silently rejected by Qt → empty white window;
a QDialog stays a top-level window unless its flags are set to Widget.
"""

import unittest
from pathlib import Path

GUI = Path(__file__).resolve().parents[2] / "standalone" / "gui.py"


class TestStandaloneWindow(unittest.TestCase):
    def setUp(self):
        self.src = GUI.read_text(encoding="utf-8")

    def test_layout_on_central_widget(self):
        self.assertIn("setCentralWidget(", self.src)
        self.assertNotIn("QVBoxLayout(win)", self.src)

    def test_toolbar_added_as_toolbar(self):
        self.assertIn("addToolBar(", self.src)

    def test_dialog_embedded_as_widget(self):
        self.assertIn("setWindowFlags(Qt.WindowType.Widget)", self.src)

    def test_torch_before_pyqt(self):
        """No PyQt6 / ref_grade.dialog import at module level (torch must load first)."""
        head = self.src[: self.src.index("def main(")]
        for line in head.splitlines():
            if line.startswith(("import ", "from ")):
                self.assertNotIn("PyQt6", line)
                self.assertNotIn("ref_grade.dialog", line)


if __name__ == "__main__":
    unittest.main()
