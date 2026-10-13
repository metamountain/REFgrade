"""K27a: Parade unten als Ersatz fürs Histogramm; Knopfreihe einzeilig; K26-Test."""

from __future__ import annotations

import ast
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
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

from ref_grade import scale  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade.widgets import Parade, Histogram  # noqa: E402

W, H = 1920, 1152


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K27aTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 1920×1152."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()
        self.s = scale.unit(W, H)
        self.row_h = scale.row_h(self.s)

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestParadePlacement(K27aTestBase):
    """Parade: kein Widget im RGB-Bereich, genau eines im Dialog, Sichtbarkeit, Geometrie."""

    def test_no_parade_in_rgb_section(self):
        """Kein Parade-Widget im Bereich RGB."""
        rgb_content = self.dlg._rgb_content
        parades = [w for w in rgb_content.findChildren(Parade)]
        self.assertEqual(len(parades), 0, "Parade-Widget im RGB-Bereich gefunden")

    def test_exactly_one_parade_in_dialog(self):
        """Genau ein Parade-Widget im gesamten Dialog."""
        parades = self.dlg.findChildren(Parade)
        self.assertEqual(len(parades), 1, f"Erwartet 1 Parade, gefunden: {len(parades)}")

    def test_parade_visible_when_scope_view_parade(self):
        """Bei scope_view='parade' ist Parade sichtbar, Histogramm nicht."""
        self.dlg._set_scope_view("parade")
        self.app.processEvents()
        self.assertTrue(self.dlg._parade.isVisible())
        self.assertFalse(self.dlg.histogram.isVisible())

    def test_histogram_visible_when_scope_view_hist(self):
        """Bei scope_view='hist' ist Histogramm sichtbar, Parade nicht."""
        self.dlg._set_scope_view("hist")
        self.app.processEvents()
        self.assertTrue(self.dlg.histogram.isVisible())
        self.assertFalse(self.dlg._parade.isVisible())

    def test_parade_and_histogram_same_geometry(self):
        """Geometrie von Parade und Histogramm ist identisch."""
        self.app.processEvents()
        h_geo = self.dlg.histogram.geometry()
        p_geo = self.dlg._parade.geometry()
        self.assertEqual(h_geo.x(), p_geo.x())
        self.assertEqual(h_geo.y(), p_geo.y())
        self.assertEqual(h_geo.width(), p_geo.width())
        self.assertEqual(h_geo.height(), p_geo.height())


class TestScopeViewPersistence(K27aTestBase):
    """Umschalten → Wert bleibt nach Schließen/erneutem Öffnen."""

    def test_scope_view_persisted(self):
        self.dlg._set_scope_view("hist")
        self.app.processEvents()
        saved_view = self.dlg._settings.value("scope_view", type=str)
        self.assertEqual(saved_view, "hist")

        # Close and reopen
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()

        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        dlg2 = RefGradeDialog(img)
        dlg2.resize(W, H)
        dlg2.show()
        _wait(self.app, lambda: dlg2.width() >= W - 20)
        self.app.processEvents()
        self.assertEqual(dlg2._scope_view, "hist")
        self.assertTrue(dlg2.histogram.isVisible())
        self.assertFalse(dlg2._parade.isVisible())
        dlg2.close()
        dlg2.deleteLater()
        self.app.processEvents()

    def test_default_scope_view_is_parade(self):
        """Standard ist 'parade'."""
        self.assertEqual(self.dlg._scope_view, "parade")
        self.assertTrue(self.dlg._parade.isVisible())
        self.assertFalse(self.dlg.histogram.isVisible())


class TestScopeGeometry(K27aTestBase):
    """Histogramm- und Parade-Rechteck = bottom-Rechteck (±1)."""

    def test_histogram_fills_bottom(self):
        rects = scale.layout_rects(W, H, self.dlg._effective_mode,
                                   self.dlg._orig_w, self.dlg._orig_h)
        bx, by, bw, bh = rects["bottom"]
        h_geo = self.dlg.histogram.geometry()
        self.assertLessEqual(abs(h_geo.width() - bw), 1)
        self.assertLessEqual(abs(h_geo.height() - bh), 1)

    def test_parade_fills_bottom(self):
        rects = scale.layout_rects(W, H, self.dlg._effective_mode,
                                   self.dlg._orig_w, self.dlg._orig_h)
        bx, by, bw, bh = rects["bottom"]
        p_geo = self.dlg._parade.geometry()
        self.assertLessEqual(abs(p_geo.width() - bw), 1)
        self.assertLessEqual(abs(p_geo.height() - bh), 1)


class TestScopeButtonRow(K27aTestBase):
    """Alle Scope-Knöpfe: gleiche y, innerhalb bottom, x-Reihenfolge, AUTO rechtsbündig."""

    def _get_buttons_in_order(self) -> list[tuple[str, QWidget]]:
        """Return buttons in expected x-order: HIST, PARADE, RGB, R, G, B, L, H, V, AUTO."""
        btns = [
            ("HIST", self.dlg._btn_hist),
            ("PARADE", self.dlg._btn_parade),
            ("RGB", self.dlg._scope_btns["rgb"]),
            ("R", self.dlg._scope_btns["r"]),
            ("G", self.dlg._scope_btns["g"]),
            ("B", self.dlg._scope_btns["b"]),
            ("L", self.dlg._scope_btns["l"]),
            ("H", self.dlg._btn_h),
            ("V", self.dlg._btn_v),
            ("AUTO", self.dlg._btn_auto),
        ]
        return btns

    def test_all_buttons_same_y(self):
        """Alle Knöpfe haben dieselbe y (±2)."""
        btns = self._get_buttons_in_order()
        ys = [b.geometry().y() for _, b in btns]
        y0 = ys[0]
        for i, y in enumerate(ys):
            self.assertAlmostEqual(y, y0, delta=2,
                                   msg=f"Button {btns[i][0]} y={y} != {y0}")

    def test_buttons_in_top_of_bottom(self):
        """Knöpfe liegen im oberen Bereich des bottom-Rechtecks (y − bottom.y ≤ row_h)."""
        rects = scale.layout_rects(W, H, self.dlg._effective_mode,
                                   self.dlg._orig_w, self.dlg._orig_h)
        by = rects["bottom"][1]
        btns = self._get_buttons_in_order()
        for name, b in btns:
            btn_y = b.geometry().y()
            # btn_y is relative to scope_block; bottom.y is relative to dialog
            # scope_block is at bottom position, so btn_y should be ≤ row_h
            self.assertLessEqual(btn_y, self.row_h,
                                 msg=f"Button {name} y={btn_y} > row_h={self.row_h}")

    def test_buttons_visible(self):
        """Alle Knöpfe sind sichtbar."""
        btns = self._get_buttons_in_order()
        for name, b in btns:
            self.assertTrue(b.isVisible(), f"Button {name} not visible")

    def test_x_order(self):
        """x-Reihenfolge exakt: HIST, PARADE, RGB, R, G, B, L, H, V, AUTO."""
        btns = self._get_buttons_in_order()
        xs = [(name, b.geometry().x()) for name, b in btns]
        # Left group should be in increasing x order
        left_group = xs[:7]  # HIST through L
        for i in range(len(left_group) - 1):
            self.assertLess(left_group[i][1], left_group[i + 1][1],
                            msg=f"{left_group[i][0]} x={left_group[i][1]} >= "
                                f"{left_group[i+1][0]} x={left_group[i+1][1]}")
        # Right group: H < V < AUTO
        right_group = xs[7:]
        for i in range(len(right_group) - 1):
            self.assertLess(right_group[i][1], right_group[i + 1][1],
                            msg=f"{right_group[i][0]} x={right_group[i][1]} >= "
                                f"{right_group[i+1][0]} x={right_group[i+1][1]}")
        # All left buttons should be left of all right buttons
        self.assertLess(left_group[-1][1], right_group[0][1],
                        msg="L should be left of H")

    def test_auto_right_edge(self):
        """AUTO rechte Kante = bottom rechte Kante − Rand ±2."""
        rects = scale.layout_rects(W, H, self.dlg._effective_mode,
                                   self.dlg._orig_w, self.dlg._orig_h)
        bx, by, bw, bh = rects["bottom"]
        auto = self.dlg._btn_auto
        auto_right = auto.geometry().x() + auto.geometry().width()
        margin = self.row_h // 4
        expected_right = bw - margin
        self.assertAlmostEqual(auto_right, expected_right, delta=2,
                               msg=f"AUTO right={auto_right}, expected={expected_right}")


class TestLayoutRectsNoParadeKey(unittest.TestCase):
    """scale.layout_rects liefert keinen Schlüssel 'parade' mehr."""

    def test_v_mode_no_parade_key(self):
        r = scale.layout_rects(1920, 1152, "V")
        self.assertNotIn("parade", r)

    def test_h_mode_no_parade_key(self):
        r = scale.layout_rects(1920, 1152, "H", 3000, 1500)
        self.assertNotIn("parade", r)

    def test_h_mode_no_image_no_parade_key(self):
        r = scale.layout_rects(1920, 1152, "H")
        self.assertNotIn("parade", r)


class TestK26ModuleConstraints(unittest.TestCase):
    """K26: keine dlg_*.py > 800 Zeilen, MRO enthält 6 Mixins, keine doppelten Methoden."""

    MIXINS = ("SectionsMixin", "LayoutMixin", "LutMixin",
              "ReferenceMixin", "WorkerMixin", "WhiteBalanceMixin")

    def test_no_module_over_800_lines(self):
        """Kein Modul aus dialog.py und dlg_*.py hat mehr als 800 Zeilen (widgets.py ausgenommen)."""
        files = list(_PLUGIN_DIR.glob("dialog.py")) + list(_PLUGIN_DIR.glob("dlg_*.py"))
        for f in files:
            if f.name == "widgets.py":
                continue
            lines = len(f.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(lines, 800,
                                 f"{f.name} has {lines} lines (> 800)")

    def test_mro_contains_six_mixins(self):
        """RefGradeDialog.__mro__ enthält die sechs Mixins."""
        mro_names = [cls.__name__ for cls in RefGradeDialog.__mro__]
        for mixin in self.MIXINS:
            self.assertIn(mixin, mro_names, f"{mixin} not in MRO")

    def test_no_method_defined_in_two_modules(self):
        """Keine Methode ist in zwei Mixin-Modulen definiert (per ast)."""
        files = list(_PLUGIN_DIR.glob("dialog.py")) + list(_PLUGIN_DIR.glob("dlg_*.py"))
        methods: dict[str, str] = {}  # method_name -> module_name
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in self.MIXINS:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name.startswith("_") and item.name not in methods:
                                methods[item.name] = f.name
                            elif item.name in methods:
                                self.fail(
                                    f"Method '{item.name}' defined in both "
                                    f"{methods[item.name]} and {f.name}"
                                )


if __name__ == "__main__":
    unittest.main()
