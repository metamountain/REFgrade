"""Tests für Paket K17: Bereichs-Master als normaler Amount-Regler statt Mini-Slider im Kopf."""

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
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeK17Tests")

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QSlider, QWidget  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

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

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path:
                return payload
        return None

    def payloads(self, path: str) -> list[dict | None]:
        return [payload for p, payload in self.calls if p == path]


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


class K17TestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k17_"))

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

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestAmountSliderExists(K17TestBase):
    """Jeder Bereich hat als erste Inhaltszeile einen Regler mit objectName='amount_<key>'."""

    def test_amount_slider_objectname(self):
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            self.assertIsNotNone(header.amount_slider, f"{key}: amount_slider is None")
            self.assertEqual(header.amount_slider.objectName(), f"amount_{key}")

    def test_amount_slider_is_first_row(self):
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            self.assertIsNotNone(header.amount_slider, f"{key}: amount_slider is None")
            self.assertEqual(header.amount_slider.objectName(), f"amount_{key}")
        # For slider-based sections (reference, tone, color) the amount is the first row
        for key in ("reference", "tone", "color"):
            rows = self.dlg.sections[key][1]
            self.assertTrue(len(rows) > 0, f"{key}: no rows")
            self.assertEqual(rows[0].objectName(), f"amount_{key}",
                             f"{key}: first row is not the amount slider")


class TestNoSliderInHeader(K17TestBase):
    """Im Kopf gibt es keinen QSlider mehr."""

    def test_no_qslider_in_header(self):
        for key in SECTION_KEYS:
            header = self.dlg.sections[key][0]
            sliders = header.findChildren(QSlider)
            self.assertEqual(len(sliders), 0,
                             f"{key}: header still contains {len(sliders)} QSlider(s)")


class TestAmountDrivesPreview(K17TestBase):
    """Amount Tone auf 40 → /preview.mix.weights['tone'] == 0.4."""

    def test_amount_tone_40(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _settle(self.app, self.fake)
        initial = self.fake.count("/preview")

        # Turn tone ON and set amount to 40 %
        self.dlg.sections["tone"][0].on.setChecked(True)
        self.dlg.sections["tone"][0].amount_slider.set_value(0.4)
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertIn("tone", payload["mix"]["weights"])
        self.assertAlmostEqual(payload["mix"]["weights"]["tone"], 0.4, places=3)


class TestResetAmount(K17TestBase):
    """↺ des Bereichs setzt Amount auf 100 zurück."""

    def test_reset_sets_amount_to_100(self):
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _settle(self.app, self.fake)

        # Set amount to 50 %
        self.dlg.sections["tone"][0].on.setChecked(True)
        self.dlg.sections["tone"][0].amount_slider.set_value(0.5)
        self.app.processEvents()

        # Click reset
        self.dlg.sections["tone"][0].undo.click()
        self.app.processEvents()

        self.assertAlmostEqual(self.dlg.sections["tone"][0].amount_slider.value(), 1.0, places=3)


class TestCollapsedLabel(K17TestBase):
    """Zugeklappt bei 60 % steht '60 %' im Kopf, bei 100 % nichts."""

    def test_collapsed_60_percent(self):
        header = self.dlg.sections["tone"][0]
        header.on.setChecked(True)
        header.amount_slider.set_value(0.6)
        header.set_collapsed(True, emit=False)
        self.app.processEvents()
        self.assertEqual(header._collapsed_label.text(), "60 %")

    def test_collapsed_100_percent(self):
        header = self.dlg.sections["tone"][0]
        header.on.setChecked(True)
        header.amount_slider.set_value(1.0)
        header.set_collapsed(True, emit=False)
        self.app.processEvents()
        self.assertEqual(header._collapsed_label.text(), "")

    def test_expanded_no_label(self):
        header = self.dlg.sections["tone"][0]
        header.on.setChecked(True)
        header.amount_slider.set_value(0.6)
        header.set_collapsed(False, emit=False)
        self.app.processEvents()
        self.assertEqual(header._collapsed_label.text(), "")


class TestAmountGeometry(K17TestBase):
    """Geometrie: Der Amount-Regler ist genauso breit wie die anderen SliderRow desselben Bereichs."""

    def _check_widths(self, width: int) -> None:
        self.dlg.resize(width, 1080)
        self.app.processEvents()
        self.app.processEvents()

        for key in ("reference", "tone", "color"):
            rows = self.dlg.sections[key][1]
            if len(rows) < 2:
                continue
            amount_w = rows[0].width()
            for r in rows[1:]:
                if r.name.startswith("amount_"):
                    continue
                self.assertAlmostEqual(amount_w, r.width(), delta=2,
                                       msg=f"{key}: amount width {amount_w} != "
                                           f"{r.name} width {r.width()} at {width}px")

    def test_geometry_1920(self):
        self._check_widths(1920)

    def test_geometry_2560(self):
        self._check_widths(2560)


if __name__ == "__main__":
    unittest.main()
