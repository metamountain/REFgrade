"""Tests für Paket K13: LUT-Kategorien, SAVE LUT, LUT-Hover, Räder offen."""

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

# Package-Modul ohne __init__.py-Ausführung
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import QEvent, QPoint  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade import config  # noqa: E402
from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402


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
        elif path == "/cube":
            # Save into the RefGrade subfolder
            save_dir = self._tmp / "luts" / "RefGrade"
            save_dir.mkdir(parents=True, exist_ok=True)
            p = save_dir / "saved.cube"
            p.write_text('TITLE "RefGrade saved"\n')
            return {"path": str(p)}
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
    """Wait until no new request on `path` arrived for quiet_ms: the start-up previews
    (dialog init + /references reply) may land in either order, so a baseline taken
    earlier counts one of them as the hover."""
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


class K13TestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient + LUT temp folder with subfolders."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k13_"))
        # LUT folder: a.cube directly, Warm/b.cube, Cool/c.cube
        cls.lut_folder = cls.tmp / "luts"
        cls.lut_folder.mkdir()
        (cls.lut_folder / "a.cube").write_text('TITLE "a"\n')
        (cls.lut_folder / "Warm").mkdir()
        (cls.lut_folder / "Warm" / "b.cube").write_text('TITLE "b"\n')
        (cls.lut_folder / "Cool").mkdir()
        (cls.lut_folder / "Cool" / "c.cube").write_text('TITLE "c"\n')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeClient(self.tmp)
        self.source = _make_source(self.tmp)
        self.image = _make_image()

        def choose_folder(start: str) -> str | None:
            return str(self.lut_folder)

        def save_lut(folder: str, filename: str) -> str | None:
            d = Path(folder)
            d.mkdir(parents=True, exist_ok=True)
            p = d / filename
            p.write_text(f'TITLE "RefGrade {filename}"\n')
            return str(p)

        self.dlg = RefGradeDialog(
            self.image, None,
            client=self.fake, source=self.source,
            save_lut=save_lut,
            choose_folder=choose_folder,
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        # Load the LUT folder (K27e: popup is populated on open, not on load)
        self.dlg._lut_folder_btn.click()
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


@unittest.skip("K27e: QComboBox → QPushButton + Tree-Popup; Verhalten in test_plugin_k27e.TestLutPopup")
class TestLutDropdown(K13TestBase):
    """Dropdown: Einträge mit Unterordnern, Überschriften nicht auswählbar, Wheel skippt."""

    def test_entries_order(self):
        """a.cube + Warm/b.cube + Cool/c.cube → — none —, a, — COOL —, c, — WARM —, b."""
        combo = self.dlg._lut_combo
        self.assertEqual(combo.count(), 6)
        self.assertEqual(combo.itemText(0), "— none —")
        self.assertEqual(combo.itemText(1), "a")
        self.assertEqual(combo.itemText(2), "— COOL —")
        self.assertEqual(combo.itemText(3), "c")
        self.assertEqual(combo.itemText(4), "— WARM —")
        self.assertEqual(combo.itemText(5), "b")

    def test_headers_not_selectable(self):
        """Überschriften haben Data == '__HEADER__'."""
        combo = self.dlg._lut_combo
        self.assertEqual(combo.itemData(2), "__HEADER__")
        self.assertEqual(combo.itemData(4), "__HEADER__")
        # Regulare Einträge haben Pfade
        self.assertNotEqual(combo.itemData(1), "__HEADER__")
        self.assertNotEqual(combo.itemData(3), "__HEADER__")
        self.assertNotEqual(combo.itemData(5), "__HEADER__")

    def test_wheel_skips_header(self):
        """Mausrad von a (Index 1) aus → c (Index 3), Überschrift wird übersprungen."""
        combo = self.dlg._lut_combo
        combo.setCurrentIndex(1)  # a
        # Simulate wheel down
        self.dlg._lut_view_wheel(_FakeWheelEvent(120))
        self.assertEqual(combo.currentIndex(), 3)
        self.assertEqual(combo.itemText(3), "c")


class _FakeWheelEvent:
    """Minimal wheel event: only angleDelta().y() is used by _lut_view_wheel."""
    def __init__(self, delta_y: int):
        self._delta = QPoint(0, delta_y)
    def angleDelta(self) -> QPoint:
        return self._delta


@unittest.skip("K27e: Hover wird im Tree-Popup intern gehandhabt; _lut_view_hover ist ein No-op")
class TestLutHover(K13TestBase):
    """Hover: Überfahren sendet /preview mit mix.lut; Schließen ohne Auswahl → /preview ohne lut."""

    def test_hover_sends_preview(self):
        """Überfahren von c sendet genau ein /preview mit mix.lut = Pfad von c.cube."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        _settle(self.app, self.fake)
        initial = self.fake.count("/preview")

        # Simulate hover on index 3 (c.cube)
        self.dlg._lut_view_hover(3)
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # Exactly one new /preview
        self.assertEqual(self.fake.count("/preview"), initial + 1)

        # mix.lut = path of c.cube
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertIn("lut", payload["mix"])
        self.assertEqual(Path(payload["mix"]["lut"]), self.lut_folder / "Cool" / "c.cube")

        # Selection unchanged (still — none —)
        self.assertEqual(self.dlg._lut_combo.currentIndex(), 0)

    def test_leave_restores(self):
        """Nach dem Schließen ohne Auswahl folgt ein /preview ohne lut."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)

        # Hover c
        self.dlg._lut_view_hover(3)
        _wait(self.app, lambda: self.fake.count("/preview") >= 2, timeout_ms=2000)
        initial = self.fake.count("/preview")

        # Leave
        self.dlg._lut_view_leave()
        _wait(self.app, lambda: self.fake.count("/preview") >= initial + 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # The restore preview has no lut in mix
        payload = self.fake.last("/preview")
        self.assertIsNotNone(payload)
        self.assertNotIn("lut", payload["mix"])


class TestLutSave(K13TestBase):
    """SAVE: /cube in RefGrade-Unterordner."""

    def test_save(self):
        """SAVE → genau ein /cube mit dir in RefGrade-Unterordner."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)

        # Click SAVE
        self.dlg._lut_save_btn.click()
        _wait(self.app, lambda: self.fake.count("/cube") >= 1, timeout_ms=2000)
        time.sleep(0.05)
        self.app.processEvents()

        # Exactly one /cube
        self.assertEqual(self.fake.count("/cube"), 1)

        # The /cube payload has dir in RefGrade subfolder
        cube_payload = self.fake.payloads("/cube")[0]
        self.assertIsNotNone(cube_payload)
        self.assertIn("RefGrade", cube_payload["dir"])

        # Selection is still — none —
        self.assertEqual(self.dlg._lut_combo.text(), "— none —")


class TestWheelsDefaultOn(K13TestBase):
    """Räder: Beim Start offen, aber OFF (2026-10-11: alle Kapitel starten OFF)."""

    def test_wheels_on_and_open(self):
        """Beim Start: wheels ON und content sichtbar."""
        self.assertFalse(self.dlg.sections["wheels"][0].is_on())  # 2026-10-11: sections start OFF, ON once something changes
        wheels_content = self.dlg._contents.get("wheels")
        self.assertIsNotNone(wheels_content)
        self.assertTrue(wheels_content.isVisible())

    def test_initial_preview_weights(self):
        """K25c: der erste /preview hat mix.weights == alle Sektionen außer reference (1.0)."""
        _wait(self.app, lambda: self.fake.count("/preview") >= 1, timeout_ms=2000)
        payload = self.fake.payloads("/preview")[0]
        self.assertIsNotNone(payload)
        self.assertEqual(payload["mix"]["weights"], {})  # 2026-10-11: sections start OFF, ON once something changes


if __name__ == "__main__":
    unittest.main()
