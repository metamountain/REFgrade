"""Tests für Paket K18: „?"-Hilfe vor jedem ON-Schalter."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
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
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeK18Tests")

from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog, HelpPopup  # noqa: E402
from ref_grade.ui_help import SECTION_HELP  # noqa: E402

SECTION_KEYS = ("reference", "tone", "color", "hsl", "curves", "wheels", "lut")


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


class FakeClient:
    """Records all calls; returns fixed responses."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        elif path == "/references":
            items = []
            for i in range(3):
                tp = _make_png(self._tmp / f"ref{i}.png", 32, 32, (i * 40, 100, 200))
                items.append({"path": f"/pool/ref{i}.jpg", "name": f"ref{i}",
                              "score": 0.9 - i * 0.05, "thumb": tp})
            return {"items": items}
        elif path == "/crop":
            return {"image": self._png, "scale": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/apply":
            raw = self._tmp / "result.raw"
            raw.write_bytes(b"\x00" * 1024)
            return {"raw": str(raw), "ms": 1.0}
        elif path == "/prerender":
            return {"image": self._png}
        elif path == "/thumb":
            return {"thumb": self._png}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def count(self, path: str) -> int:
        return sum(1 for p, _ in self.calls if p == path)


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _settle(app: QApplication, fake, path: str = "/preview", quiet_ms: int = 300,
            timeout_ms: int = 3000) -> None:
    deadline = time.monotonic() + timeout_ms / 1000
    last, since = fake.count(path), time.monotonic()
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
        n = fake.count(path)
        if n != last:
            last, since = n, time.monotonic()
        elif time.monotonic() - since >= quiet_ms / 1000:
            return


def _make_source(tmp: Path) -> dict:
    return {
        "raw": str(tmp / "source.raw"),
        "width": 100,
        "height": 80,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "session": "fake_session_id",
    }


def _make_image(w: int = 200, h: int = 150) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(80, 120, 160))
    return img


def _expected_display_names() -> dict[str, list[str]]:
    """Display names (from LABELS) that must appear in each section's help text.

    Derived from SECTIONS in core/refgrade/sections.py + LABELS in dialog.py.
    Only sliders with a LABELS entry are checked.
    """
    return {
        "reference": ["Light", "Color", "Protect shadows", "Protect highlights"],
        "tone": ["Exposure", "Contrast", "Highlights", "Shadows", "Whites", "Blacks", "Clarity"],
        "color": ["Temperature", "Tint", "Saturation", "Vibrance"],
        "wheels": ["Blending", "Balance"],
        "hsl": [],
        "curves": [],
        "lut": [],
    }


class K18TestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k18_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()
        self.dlg = RefGradeDialog(
            self.image, None,
            client=self.fake, source=self.source,
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _settle(self.app, self.fake)

    def tearDown(self):
        if self.dlg._help_popup is not None:
            self.dlg._help_popup.close()
            self.dlg._help_popup = None
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestHelpButtonExists(K18TestBase):
    """Jeder Bereich hat help_<key>, quadratisch, links vom ON-Schalter."""

    def test_help_button_exists(self):
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            btn = header.findChild(QPushButton, f"help_{key}")
            self.assertIsNotNone(btn, f"{key}: help_{key} not found")

    def test_help_button_is_square(self):
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            btn = header.findChild(QPushButton, f"help_{key}")
            self.assertAlmostEqual(btn.width(), btn.height(), delta=1,
                                   msg=f"{key}: help button not square "
                                       f"({btn.width()}x{btn.height()})")

    def test_help_button_right_of_on(self):
        """Layout order: AUTO · ↺ · ON · ? (help is rightmost)."""
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            help_btn = header.findChild(QPushButton, f"help_{key}")
            on_btn = header.findChild(QPushButton, f"sec_{key}_on")
            self.assertIsNotNone(help_btn)
            self.assertIsNotNone(on_btn)
            help_geo = help_btn.geometry()
            on_geo = on_btn.geometry()
            self.assertGreaterEqual(help_geo.x(), on_geo.x() + on_geo.width(),
                                 msg=f"{key}: help button not right of ON "
                                     f"(help.x={help_geo.x()} < on.x+{on_geo.width()}="
                                     f"{on_geo.x() + on_geo.width()})")
            self.assertAlmostEqual(help_geo.y(), on_geo.y(), delta=2,
                                   msg=f"{key}: help button not on same row as ON")


class TestHelpPopupOpens(K18TestBase):
    """Klick öffnet das Popup mit dem Text aus SECTION_HELP[key]."""

    def _click_help(self, key: str) -> HelpPopup:
        header = self.dlg.sections[key][0]
        btn = header.findChild(QPushButton, f"help_{key}")
        btn.click()
        self.app.processEvents()
        self.assertIsNotNone(self.dlg._help_popup, f"{key}: popup not opened")
        self.assertTrue(self.dlg._help_popup.isVisible(), f"{key}: popup not visible")
        return self.dlg._help_popup

    def test_popup_shows_section_text(self):
        popup = self._click_help("tone")
        self.assertEqual(popup._section_key, "tone")
        label_text = popup._label.text()
        self.assertIn("Exposure", label_text)
        self.assertIn("Controls", label_text)

    def test_popup_no_state_change(self):
        key = "tone"
        header = self.dlg.sections[key][0]
        on_before = header.is_on()
        collapsed_before = header.is_collapsed()
        values_before = {r.name: r.value() for r in self.dlg.sections[key][1]}
        preview_before = self.fake.count("/preview")

        self._click_help(key)

        self.assertEqual(header.is_on(), on_before, "ON/OFF changed")
        self.assertEqual(header.is_collapsed(), collapsed_before, "Collapse changed")
        values_after = {r.name: r.value() for r in self.dlg.sections[key][1]}
        self.assertEqual(values_before, values_after, "Slider values changed")
        self.assertEqual(self.fake.count("/preview"), preview_before,
                         "A /preview was sent")

    def test_second_click_closes(self):
        key = "tone"
        self._click_help(key)
        self.assertTrue(self.dlg._help_popup.isVisible())

        header = self.dlg.sections[key][0]
        btn = header.findChild(QPushButton, f"help_{key}")
        btn.click()
        self.app.processEvents()

        self.assertFalse(self.dlg._help_popup is not None and self.dlg._help_popup.isVisible(),
                         "Popup still visible after second click")


class TestSectionHelpContent(K18TestBase):
    """SECTION_HELP hat für jeden Bereich den Text; Reglernamen in Controls-Tabelle."""

    def test_all_sections_have_text(self):
        for key in SECTION_KEYS:
            self.assertIn(key, SECTION_HELP, f"SECTION_HELP missing key {key!r}")
            self.assertTrue(SECTION_HELP[key].strip(), f"SECTION_HELP[{key!r}] is empty")

    def test_slider_names_present(self):
        from ref_grade.ui_help import controls_table
        expected = _expected_display_names()
        for key in SECTION_KEYS:
            text = SECTION_HELP[key] + "\n" + controls_table(key)
            for display in expected[key]:
                self.assertIn(display, text,
                              f"SECTION_HELP[{key!r}] missing display name {display!r}")

    def test_controls_heading_present(self):
        from ref_grade.ui_help import controls_table
        for key in SECTION_KEYS:
            table = controls_table(key)
            self.assertIn("**Controls**", table,
                          f"controls_table({key!r}) missing '**Controls**' heading")


class TestHelpPopupGeometry(K18TestBase):
    """Das Popup liegt als Quadrat vollständig über dem Bild (1920 und 1366; Nutzer 2026-10-11)."""

    def _check_geometry(self, width: int) -> None:
        self.dlg.resize(width, 1080)
        self.app.processEvents()
        self.app.processEvents()

        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            btn = header.findChild(QPushButton, f"help_{key}")
            btn.click()
            self.app.processEvents()
            popup = self.dlg._help_popup
            self.assertIsNotNone(popup, f"{key}: popup not opened at {width}px")
            if popup is None:
                continue
            self.assertTrue(popup.isVisible(), f"{key}: popup not visible at {width}px")
            popup_geo = popup.geometry()
            # 2026-10-11 (user): black square over the picture — inside the view, not the column
            view = self.dlg.view
            vpos = view.mapToGlobal(view.rect().topLeft())
            self.assertEqual(popup_geo.width(), popup_geo.height(), f"{key} at {width}px: square")
            self.assertGreaterEqual(popup_geo.x(), vpos.x() - 2,
                                    msg=f"{key} at {width}px: popup left of the picture")
            self.assertLessEqual(popup_geo.x() + popup_geo.width(), vpos.x() + view.width() + 2,
                                 msg=f"{key} at {width}px: popup right of the picture")
            self.assertGreaterEqual(popup_geo.y(), vpos.y() - 2, f"{key} at {width}px: above")
            self.assertLessEqual(popup_geo.y() + popup_geo.height(), vpos.y() + view.height() + 2,
                                 f"{key} at {width}px: below the picture")
            # Close popup before next iteration
            popup.close()
            self.dlg._help_popup = None
            self.app.processEvents()

    def test_geometry_1920(self):
        self._check_geometry(1920)

    def test_geometry_1366(self):
        self._check_geometry(1366)


if __name__ == "__main__":
    unittest.main()
