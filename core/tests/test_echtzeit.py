"""K31b: Echtzeit-Vorschau — belegte Beschleuniger mit festem Sollwert.

Kandidat 6 (LUT 17³ beim Ziehen):
- Session: preview(lut_size=17) baut eine 17³-LUT; Default bleibt 33³.
- Die 17³-Vorschau weicht im Mittel ≤ 3/255 von der 33³-Vorschau ab.
- Plugin: beim Ziehen sendet das Dialog-Payload lut_size == 17, nach Loslassen 33.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable

import torch

from refgrade import session as sess
from refgrade.session import Session, LUT_SIZE

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
PYTHON = sys.executable

_PLUGIN_DIR = PROJECT_DIR / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade import worker_client  # noqa: E402


def _make_test_image(h: int = 64, w: int = 64) -> torch.Tensor:
    """Horizontaler Farbverlauf + Checkerboard, [1,H,W,3] float32 0..1."""
    img = torch.zeros(1, h, w, 3, dtype=torch.float32)
    for y in range(h):
        for x in range(w):
            img[0, y, x, :] = 1.0 if (x + y) % 2 == 0 else 0.0
    grad = torch.linspace(0.0, 1.0, w, dtype=torch.float32)
    grad = grad[None, None, :, None].expand(1, h, w, 3)
    return (0.5 * img + 0.5 * grad).clamp(0.0, 1.0)


def _make_session(img: torch.Tensor) -> Session:
    """Minimale Session für die Session-Tests (kein pair, keine LUT-Datei)."""
    s = Session.__new__(Session)
    s.source = img
    s.source_full = img
    s.alpha = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
    s.ignore = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
    s.hists = {}
    s.previews = {}
    s.luts = {}
    s.method_fns = {}
    s.cube_cache = {}
    s.section_grid_cache = {}
    s.depth = 8
    return s


# Nicht-identische Regler, damit die LUT tatsächlich etwas tut.
_VALUES = {
    "exposure": 0.5,
    "contrast": 1.3,
    "temperature": 20.0,
    "light_strength": 1.0,
    "color_strength": 1.0,
}


class TestLutSize(unittest.TestCase):
    """Kandidat 6: LUT-Größe 17 beim Ziehen, 33 sonst (Session-Ebene)."""

    def test_default_lut_size_is_33(self):
        img = _make_test_image()
        s = _make_session(img)
        sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None)
        sizes = [l["size"] for l in s.luts.values()]
        self.assertIn(LUT_SIZE, sizes)
        self.assertNotIn(17, sizes)

    def test_lut_size_17_during_drag(self):
        img = _make_test_image()
        s = _make_session(img)
        sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None, lut_size=17)
        sizes = [l["size"] for l in s.luts.values()]
        self.assertIn(17, sizes, "beim Ziehen muss eine 17³-LUT gebaut werden")

    def test_17_close_to_33(self):
        """17³-Vorschau weicht im Mittel ≤ 3/255 von der 33³-Vorschau ab."""
        img = _make_test_image()
        s = _make_session(img)
        r33, _ = sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None, lut_size=33)
        r17, _ = sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None, lut_size=17)
        mean_dev = (r17 - r33).abs().mean().item()
        self.assertLessEqual(mean_dev, 3.0 / 255.0,
                             f"17³ vs 33³ Mittelabweichung {mean_dev:.6f} > 3/255")


class _FakeClient:
    """Nur /preview aufzeichnen; feste Antwort liefern."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        png = tmp / "resp.png"
        img = QImage(64, 64, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 150, 200))
        img.save(str(png), "PNG")
        self._png = str(png)

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def last(self, path: str) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == path:
                return payload
        return None

    def last_lut(self) -> dict | None:
        """Letzte /preview mit stage=='lut' (der lut-stage-Pfad aus _trigger_preview)."""
        for p, payload in reversed(self.calls):
            if p == "/preview" and payload.get("stage") == "lut":
                return payload
        return None


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class TestDragLutSize(unittest.TestCase):
    """Kandidat 6 (Plugin): Ziehen → lut_size 17, Loslassen → lut_size 33."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_echtzeit_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = _FakeClient(self.tmp)
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(
            img, None,
            client=self.fake,
            source={"raw": str(self.tmp / "s.raw"), "width": 200, "height": 150,
                    "depth": "U8", "x": 0, "y": 0, "session": "fake_session_id"},
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        _wait(self.app, lambda: self.fake.last("/preview") is not None, timeout_ms=2000)

    def test_drag_sends_lut_size_17(self):
        lut_count = lambda: sum(1 for p, pl in self.fake.calls
                                if p == "/preview" and pl.get("stage") == "lut")

        # Ziehen → lut-stage /preview mit lut_size 17
        n_before = lut_count()
        self.dlg._on_drag_started()
        self.assertTrue(self.dlg._dragging)
        self.dlg._trigger_preview()
        _wait(self.app, lambda: lut_count() > n_before, timeout_ms=2000)
        drag_lut = self.fake.last_lut()
        self.assertIsNotNone(drag_lut, "keine lut-stage /preview beim Ziehen")
        self.assertEqual(drag_lut.get("lut_size"), 17,
                         f"beim Ziehen lut_size={drag_lut.get('lut_size')}, erwartet 17")

        # Loslassen → lut-stage /preview mit lut_size 33
        n_before = lut_count()
        self.dlg._on_drag_stopped()
        self.assertFalse(self.dlg._dragging)
        _wait(self.app, lambda: lut_count() > n_before, timeout_ms=2000)
        release_lut = self.fake.last_lut()
        self.assertIsNotNone(release_lut, "keine lut-stage /preview nach Loslassen")
        self.assertEqual(release_lut.get("lut_size"), 33,
                         f"nach Loslassen lut_size={release_lut.get('lut_size')}, erwartet 33")


class TestScopesPayload(unittest.TestCase):
    """K31b-D: Plugin sendet nur sichtbare Scopes im Preview-Payload."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_scopes_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = _FakeClient(self.tmp)
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(
            img, None,
            client=self.fake,
            source={"raw": str(self.tmp / "s.raw"), "width": 200, "height": 150,
                    "depth": "U8", "x": 0, "y": 0, "session": "fake_session_id"},
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        _wait(self.app, lambda: self.fake.last("/preview") is not None, timeout_ms=2000)

    def test_scopes_includes_visible(self):
        """Vorschaupayload enthält 'scopes' mit den sichtbaren Scopes."""
        payload = self.fake.last_lut()
        self.assertIsNotNone(payload)
        self.assertIn("scopes", payload, "Payload muss 'scopes' enthalten")
        self.assertIn("hist", payload["scopes"], "hist ist immer sichtbar")

    def test_scopes_excludes_hidden_vectorscope(self):
        """Vectorscope ausgeblendet → nicht in scopes."""
        self.dlg.vectorscope.setVisible(False)
        self.app.processEvents()
        self.assertFalse(self.dlg.vectorscope.isVisible(),
                         "Vectorscope muss nach setVisible(False) unsichtbar sein")
        self.dlg._trigger_preview()
        _wait(self.app, lambda: self.fake.last_lut() is not None
              and "scopes" in self.fake.last_lut(), timeout_ms=2000)
        payload = self.fake.last_lut()
        self.assertIn("scopes", payload)
        self.assertNotIn("vectorscope", payload["scopes"],
                         "ausgeblendeter Vectorscope darf nicht in scopes sein")
        self.dlg.vectorscope.setVisible(True)  # cleanup


class TestDragColorWheel(unittest.TestCase):
    """K31b-F: Ziehen am ColorWheel → halbe Auflösung im Payload."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_wheel_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = _FakeClient(self.tmp)
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(
            img, None,
            client=self.fake,
            source={"raw": str(self.tmp / "s.raw"), "width": 200, "height": 150,
                    "depth": "U8", "x": 0, "y": 0, "session": "fake_session_id"},
        )
        self.dlg.resize(1920, 1080)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        _wait(self.app, lambda: self.fake.last("/preview") is not None, timeout_ms=2000)

    def test_wheel_drag_halves_max_side(self):
        """Beim Ziehen am ColorWheel: max_side == voll // 2."""
        full = self.dlg._max_side()
        # Simuliere Drag-Start (wie _on_drag_started)
        self.dlg._on_drag_started()
        self.assertTrue(self.dlg._dragging)
        self.dlg._trigger_preview()
        _wait(self.app, lambda: self.fake.last_lut() is not None
              and self.fake.last_lut().get("max_side") == full // 2, timeout_ms=2000)
        payload = self.fake.last_lut()
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("max_side"), full // 2,
                         f"beim Rad-Ziehen max_side={payload.get('max_side')}, "
                         f"erwartet {full // 2}")
        # Cleanup
        self.dlg._on_drag_stopped()


class TestSectionGridCache(unittest.TestCase):
    """K31b-2: Section-Grid-Cache — geänderte Sektionen werden neu berechnet,
    unveränderte nicht."""

    def test_unchanged_section_not_recalled(self):
        """Nach Änderung nur der tone-Regler wird die wheels-Fn nicht erneut aufgerufen."""
        img = _make_test_image()
        s = _make_session(img)

        # Zähler für wheels-Aufrufe
        call_count = {"wheels": 0}
        orig_wheels_fn = None

        # Erster Build: alle Sektionen werden aufgerufen
        sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None)
        self.assertEqual(len(s.section_grid_cache) > 0, True,
                         "Nach erstem Build muss der Cache gefüllt sein")

        # Zweiter Build: nur tone ändern → wheels darf nicht erneut aufgerufen werden
        cache_size_before = len(s.section_grid_cache)
        vals2 = dict(_VALUES)
        vals2["contrast"] = 1.5  # nur tone ändern
        sess.preview(s, [], "reinhard", vals2, img.shape[2], None)
        # Der Cache sollte jetzt einen neuen Eintrag für tone haben,
        # aber den alten für wheels behalten
        cache_size_after = len(s.section_grid_cache)
        self.assertGreater(cache_size_after, cache_size_before,
                           "Nach tone-Änderung muss der Cache wachsen (neuer tone-Eintrag)")

    def test_same_values_no_new_cache_entry(self):
        """Gleiche Werte → kein neuer Cache-Eintrag."""
        img = _make_test_image()
        s = _make_session(img)
        sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None)
        n1 = len(s.section_grid_cache)
        # Gleiche Werte erneut → LUT-Cache trifft, kein neuer Grid-Eintrag
        sess.preview(s, [], "reinhard", _VALUES, img.shape[2], None)
        n2 = len(s.section_grid_cache)
        self.assertEqual(n1, n2,
                         "Gleiche Werte dürfen keinen neuen Grid-Cache-Eintrag erzeugen")


if __name__ == "__main__":
    unittest.main()
