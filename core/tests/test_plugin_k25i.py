"""K25i: Ergonomie — Leerstellen und Unsinn = 0.

Tests bei 1366×720, 1920×1032, 2560×1392, alle Bereiche ON und aufgeklappt.
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

from PyQt6.QtCore import QSettings, QPoint, QRect  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QAbstractButton, QLabel, QWidget,
)

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

RESOLUTIONS = [(1366, 720), (1920, 1032), (2560, 1392)]


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K25iTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at given resolution, all sections ON + expanded."""

    W, H = 1920, 1032

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(self.W, self.H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= self.W - 20)
        self.app.processEvents()
        # Expand all sections (reference is collapsed by default)
        for key, (hdr, _rows) in self.dlg.sections.items():
            if hdr.is_collapsed():
                hdr._title_click()
                self.app.processEvents()
        self.s = scale.unit(self.W, self.H)
        self.row_h = scale.row_h(self.s)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestLabelAndButtonTextFits(K25iTestBase):
    """Rule 1: fontMetrics().horizontalAdvance(text) ≤ contentsRect().width() for all visible labels/buttons."""

    def test_all_labels_fit(self):
        """Every visible QLabel with text: text fits in its width."""
        for w in self.dlg.findChildren(QLabel):
            if not w.isVisible():
                continue
            text = w.text()
            if not text:
                continue
            fm = w.fontMetrics()
            needed = fm.horizontalAdvance(text)
            available = w.contentsRect().width()
            self.assertLessEqual(
                needed, available,
                msg=f"Label '{text}' needs {needed}px but has {available}px (width={w.width()})"
            )

    def test_all_buttons_fit(self):
        """Every visible QAbstractButton with text: text fits in its width."""
        for w in self.dlg.findChildren(QAbstractButton):
            if not w.isVisible():
                continue
            text = w.text()
            if not text:
                continue
            fm = w.fontMetrics()
            needed = fm.horizontalAdvance(text)
            available = w.contentsRect().width()
            self.assertLessEqual(
                needed, available,
                msg=f"Button '{text}' needs {needed}px but has {available}px (width={w.width()})"
            )


class TestStatusStrip(K25iTestBase):
    """Rule 2, revised by the user 2026-10-11: the status never sits on or under the image;
    the image gets the full height and the status line sits above the main buttons."""

    def test_status_h_set(self):
        """No strip under the image: the view's drawing area is its full height."""
        self.assertEqual(self.dlg.view._status_h, 0)
        self.assertEqual(self.dlg.view._image_rect().height(), self.dlg.view.height())

    def test_status_rect_outside_image(self):
        """The status label sits in the column, directly above the button row."""
        lbl = self.dlg._status_label
        self.assertTrue(self.dlg.panel.isAncestorOf(lbl) or lbl.parentWidget() is not None)
        self.app.processEvents()
        lbl_bottom = lbl.mapTo(self.dlg, lbl.rect().bottomLeft()).y()
        btn_top = self.dlg._btn_reset.mapTo(self.dlg, self.dlg._btn_reset.rect().topLeft()).y()
        self.assertLessEqual(lbl_bottom, btn_top)
        self.assertLess(btn_top - lbl_bottom, 12, "label right above the buttons")
        self.assertIn("%", lbl.toolTip(), "zoom shown with the status")


class TestMasterRowFirst(K25iTestBase):
    """Rule 3: MASTER is the first row of the column, with the general '?' button."""

    def test_master_is_first_row(self):
        """MASTER slider and '?' button are in the same row (same y ±2)."""
        master_y = self.dlg.master.mapTo(self.dlg.panel, QPoint(0, 0)).y()
        help_y = self.dlg._general_help_btn.mapTo(self.dlg.panel, QPoint(0, 0)).y()
        self.assertAlmostEqual(master_y, help_y, delta=2,
                               msg=f"MASTER y={master_y} != help y={help_y}")

    def test_master_is_topmost(self):
        """MASTER is above all section headers."""
        master_y = self.dlg.master.mapTo(self.dlg.panel, QPoint(0, 0)).y()
        for key, (hdr, _rows) in self.dlg.sections.items():
            hdr_y = hdr.mapTo(self.dlg.panel, QPoint(0, 0)).y()
            self.assertLess(master_y, hdr_y,
                            msg=f"MASTER y={master_y} not above {key} header y={hdr_y}")


class TestCollapseByTitleClick(K25iTestBase):
    """Rule 4: clicking the title text of TONE collapses and expands it."""

    def test_title_click_toggles(self):
        """Clicking TONE title toggles the collapsed state."""
        tone_header = self.dlg.sections["tone"][0]
        initial = tone_header.is_collapsed()
        tone_header._title_click()
        self.app.processEvents()
        self.assertEqual(tone_header.is_collapsed(), not initial,
                         "Title click should toggle collapsed state")
        # Toggle back
        tone_header._title_click()
        self.app.processEvents()
        self.assertEqual(tone_header.is_collapsed(), initial,
                         "Second title click should restore state")

    def test_triangle_symbol(self):
        """Triangle state matches section collapsed state (K27c: painted, no text)."""
        tone_header = self.dlg.sections["tone"][0]
        self.assertEqual(tone_header.triangle.is_collapsed(), tone_header.is_collapsed())
        # Toggle and check
        tone_header._title_click()
        self.app.processEvents()
        self.assertEqual(tone_header.triangle.is_collapsed(), tone_header.is_collapsed())


class TestLutFolderLabel(K25iTestBase):
    """Rule 5: folder name in same row as dropdown, not on its own row."""

    def test_folder_label_same_row_as_combo(self):
        """The LUT folder label is y-equal (±2) with the LUT dropdown."""
        combo_y = self.dlg._lut_combo.mapTo(self.dlg, QPoint(0, 0)).y()
        label_y = self.dlg._lut_folder_label.mapTo(self.dlg, QPoint(0, 0)).y()
        self.assertAlmostEqual(combo_y, label_y, delta=2,
                               msg=f"Combo y={combo_y} != label y={label_y}")

    def test_folder_label_prefix(self):
        """The folder label starts with 'in: '."""
        text = self.dlg._lut_folder_label.text()
        self.assertTrue(text.startswith("in: "),
                        msg=f"Folder label '{text}' does not start with 'in: '")


class TestButtonSizes(K25iTestBase):
    """Rule 7: every visible QAbstractButton: height ≥ row_h−1, width ≥ row_h−1."""

    def test_all_buttons_min_size(self):
        """Every visible button meets the minimum size."""
        for w in self.dlg.findChildren(QAbstractButton):
            if not w.isVisible():
                continue
            h = w.height()
            wd = w.width()
            self.assertGreaterEqual(
                h, self.row_h - 1,
                msg=f"Button '{w.objectName() or w.text()}' height {h} < row_h-1={self.row_h - 1}"
            )
            self.assertGreaterEqual(
                wd, self.row_h - 1,
                msg=f"Button '{w.objectName() or w.text()}' width {wd} < row_h-1={self.row_h - 1}"
            )


class TestColumnGaps(K25iTestBase):
    """Rule 8: distance between consecutive visible children of the column ≤ row_h + 2."""

    def test_gaps_in_scroll_area(self):
        """Gaps between consecutive visible widgets in the scroll area ≤ row_h + 2."""
        scroll = self.dlg._scroll
        layout = scroll.layout()
        if layout is None:
            self.skipTest("No layout on scroll area")
        visible_widgets = []
        for i in range(layout.count()):
            item = layout.itemAt(i)
            w = item.widget() if item else None
            if w is not None and w.isVisible():
                y = w.mapTo(scroll, QPoint(0, 0)).y()
                bottom = y + w.height()
                visible_widgets.append((y, bottom, w))
        visible_widgets.sort()
        for i in range(len(visible_widgets) - 1):
            y1, b1, w1 = visible_widgets[i]
            y2, b2, w2 = visible_widgets[i + 1]
            gap = y2 - b1
            self.assertLessEqual(
                gap, self.row_h + 2,
                msg=f"Gap between '{w1.objectName() or w1.metaObject().className()}' "
                    f"and '{w2.objectName() or w2.metaObject().className()}' = {gap} > row_h+2={self.row_h + 2}"
            )


class TestScopeButtonSquare(K25iTestBase):
    """Rule 6: scope buttons are square with side row_h."""

    def test_scope_buttons_square(self):
        """Scope buttons have side = row_h (±1)."""
        from ref_grade.widgets import ScopeButton
        for btn in self.dlg.findChildren(ScopeButton):
            if not btn.isVisible():
                continue
            self.assertAlmostEqual(btn.width(), btn.height(), delta=1,
                                   msg=f"ScopeButton '{btn.channel}' not square: {btn.width()}×{btn.height()}")
            self.assertAlmostEqual(btn.width(), self.row_h, delta=1,
                                   msg=f"ScopeButton '{btn.channel}' side {btn.width()} != row_h={self.row_h}")


# ── Multi-resolution: run the key checks at all 3 sizes ──────────────────────

class TestMultiResolution(unittest.TestCase):
    """Key checks at 1366×720, 1920×1032, 2560×1392."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _make_dialog(self, W: int, H: int) -> RefGradeDialog:
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        dlg = RefGradeDialog(img)
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()
        for key, (hdr, _rows) in dlg.sections.items():
            if hdr.is_collapsed():
                hdr._title_click()
                self.app.processEvents()
        return dlg

    def _cleanup(self, dlg: RefGradeDialog):
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_labels_fit_all_resolutions(self):
        """At all 3 resolutions: no label text is truncated."""
        for W, H in RESOLUTIONS:
            with self.subTest(W=W, H=H):
                dlg = self._make_dialog(W, H)
                try:
                    for w in dlg.findChildren(QLabel):
                        if not w.isVisible() or not w.text():
                            continue
                        fm = w.fontMetrics()
                        needed = fm.horizontalAdvance(w.text())
                        available = w.contentsRect().width()
                        self.assertLessEqual(
                            needed, available,
                            msg=f"[{W}×{H}] Label '{w.text()}' needs {needed}px, has {available}px"
                        )
                finally:
                    self._cleanup(dlg)

    def test_master_first_row_all_resolutions(self):
        """At all 3 resolutions: MASTER is the first row with the '?' button."""
        for W, H in RESOLUTIONS:
            with self.subTest(W=W, H=H):
                dlg = self._make_dialog(W, H)
                try:
                    master_y = dlg.master.mapTo(dlg.panel, QPoint(0, 0)).y()
                    help_y = dlg._general_help_btn.mapTo(dlg.panel, QPoint(0, 0)).y()
                    self.assertAlmostEqual(master_y, help_y, delta=2,
                                           msg=f"[{W}×{H}] MASTER y={master_y} != help y={help_y}")
                finally:
                    self._cleanup(dlg)

    def test_button_sizes_all_resolutions(self):
        """At all 3 resolutions: all visible buttons meet minimum size."""
        for W, H in RESOLUTIONS:
            with self.subTest(W=W, H=H):
                dlg = self._make_dialog(W, H)
                try:
                    rh = scale.row_h(scale.unit(W, H))
                    for w in dlg.findChildren(QAbstractButton):
                        if not w.isVisible():
                            continue
                        self.assertGreaterEqual(w.height(), rh - 1,
                                                msg=f"[{W}×{H}] Button '{w.objectName()}' h={w.height()} < {rh - 1}")
                        self.assertGreaterEqual(w.width(), rh - 1,
                                                msg=f"[{W}×{H}] Button '{w.objectName()}' w={w.width()} < {rh - 1}")
                finally:
                    self._cleanup(dlg)

    def test_status_strip_all_resolutions(self):
        """At all 3 resolutions: status strip is set and outside the image."""
        for W, H in RESOLUTIONS:
            with self.subTest(W=W, H=H):
                dlg = self._make_dialog(W, H)
                try:
                    self.assertEqual(dlg.view._status_h, 0, msg=f"[{W}×{H}] strip under image")
                    self.assertTrue(dlg._status_label.isVisible(), msg=f"[{W}×{H}] no status")
                finally:
                    self._cleanup(dlg)

    def test_lut_folder_label_all_resolutions(self):
        """At all 3 resolutions: folder label is in the same row as the combo."""
        for W, H in RESOLUTIONS:
            with self.subTest(W=W, H=H):
                dlg = self._make_dialog(W, H)
                try:
                    combo_y = dlg._lut_combo.mapTo(dlg, QPoint(0, 0)).y()
                    label_y = dlg._lut_folder_label.mapTo(dlg, QPoint(0, 0)).y()
                    self.assertAlmostEqual(combo_y, label_y, delta=2,
                                           msg=f"[{W}×{H}] combo y={combo_y} != label y={label_y}")
                finally:
                    self._cleanup(dlg)


if __name__ == "__main__":
    unittest.main()
