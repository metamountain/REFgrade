"""K25j: Reset (↺ pro Bereich und RESET gesamt) setzt Regler auf Neutral und löst /preview aus."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest
import unittest.mock
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

from ref_grade import worker_client  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402


def _make_png(path: Path, w: int = 64, h: int = 64, color=(128, 64, 32)) -> str:
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(*color))
    img.save(str(path), "PNG")
    return str(path)


class FakeClient:
    """Records calls; returns fixed responses."""

    def __init__(self, tmp: Path):
        self.calls: list[tuple[str, dict | None]] = []
        self._tmp = tmp
        self._png = _make_png(tmp / "resp.png", 64, 64, (100, 150, 200))

    def __call__(self, port: int, path: str, payload: dict | None = None) -> dict:
        self.calls.append((path, payload))
        if path == "/preview":
            return {"image": self._png, "histogram": {
                "source": [0.015625] * 64, "result": [0.02] * 64}, "ms": 1.0}
        elif path == "/crop":
            return {"image": self._png, "scale": 1.0}
        elif path == "/fit":
            return {"values": {}}
        elif path == "/apply":
            raw = self._tmp / "result.raw"
            raw.write_bytes(b"\x00" * 1024)
            return {"raw": str(raw), "ms": 1.0}
        elif path == "/references":
            return {"items": []}
        elif path == "/thumb":
            return {"thumb": self._png}
        else:
            raise worker_client.WorkerError(404, f"Unknown path: {path}")

    def last_preview(self) -> dict | None:
        for p, payload in reversed(self.calls):
            if p == "/preview":
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


class ResetTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog with FakeClient."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = Path(tempfile.mkdtemp(prefix="refgrade_k25j_"))

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()
        self.fake = FakeClient(self.tmp)
        self.source = {
            "raw": str(self.tmp / "source.raw"),
            "width": 200,
            "height": 150,
            "depth": "U8",
            "x": 0,
            "y": 0,
            "session": "test_session",
        }
        img = QImage(200, 150, QImage.Format.Format_RGB32)
        img.fill(QColor(80, 120, 160))
        self.dlg = RefGradeDialog(
            img, None,
            client=self.fake, source=self.source,
        )
        self.dlg.resize(1920, 1032)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.isVisible())
        self.app.processEvents()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestResetSection(ResetTestBase):
    """_reset_section(key) setzt alle Regler des Bereichs auf Default."""

    def _set_slider(self, section_key: str, name: str, value: float) -> None:
        """Set a slider in a section to a non-default value."""
        for r in self.dlg.sections[section_key][1]:
            if r.name == name:
                r.set_value(value)
                return
        raise AssertionError(f"Slider {name} not found in section {section_key}")

    def test_reset_tone(self):
        """Tone-Reset: exposure, contrast, highlights, shadows, whites, blacks, clarity → Default."""
        self._set_slider("tone", "exposure", 1.0)
        self._set_slider("tone", "contrast", 1.5)
        self._set_slider("tone", "highlights", -0.5)
        self._set_slider("tone", "shadows", 0.5)
        self._set_slider("tone", "whites", 0.3)
        self._set_slider("tone", "blacks", -0.3)
        self._set_slider("tone", "clarity", 0.5)

        self.dlg._reset_section("tone")
        self.app.processEvents()

        for r in self.dlg.sections["tone"][1]:
            self.assertAlmostEqual(r.value(), r.default, places=5,
                                   msg=f"tone/{r.name}: {r.value()} ≠ default {r.default}")

    def test_reset_rgb(self):
        """RGB-Reset: alle 15 rgb_*-Regler → Default."""
        self._set_slider("rgb", "rgb_m_gamma", 2.0)
        self._set_slider("rgb", "rgb_r_gain", 3.0)
        self._set_slider("rgb", "rgb_b_pedestal", 0.3)
        self._set_slider("rgb", "rgb_use_shadows", 0.0)

        self.dlg._reset_section("rgb")
        self.app.processEvents()

        for r in self.dlg.sections["rgb"][1]:
            self.assertAlmostEqual(r.value(), r.default, places=5,
                                   msg=f"rgb/{r.name}: {r.value()} ≠ default {r.default}")

    def test_reset_color(self):
        """Color-Reset: temperature, tint, saturation, vibrance → Default."""
        self._set_slider("color", "temperature", 0.5)
        self._set_slider("color", "tint", -0.3)
        self._set_slider("color", "saturation", 2.0)
        self._set_slider("color", "vibrance", 0.8)

        self.dlg._reset_section("color")
        self.app.processEvents()

        for r in self.dlg.sections["color"][1]:
            self.assertAlmostEqual(r.value(), r.default, places=5,
                                   msg=f"color/{r.name}: {r.value()} ≠ default {r.default}")

    def test_reset_reference(self):
        """Reference-Reset: light_strength, color_strength, protect_* → Default."""
        self._set_slider("reference", "light_strength", 0.3)
        self._set_slider("reference", "color_strength", 0.5)
        self._set_slider("reference", "protect_shadows", 0.7)
        self._set_slider("reference", "protect_highlights", 0.4)

        self.dlg._reset_section("reference")
        self.app.processEvents()

        for r in self.dlg.sections["reference"][1]:
            self.assertAlmostEqual(r.value(), r.default, places=5,
                                   msg=f"reference/{r.name}: {r.value()} ≠ default {r.default}")


class TestResetAll(ResetTestBase):
    """_reset_all() setzt ALLE Regler auf Default und löst /preview aus."""

    def test_reset_all_sliders_default(self):
        """Nach _reset_all(): alle sichtbaren Regler stehen auf Default."""
        # Set some sliders to non-default values
        self._set_slider("tone", "exposure", 2.0)
        self._set_slider("color", "saturation", 0.0)
        self._set_slider("rgb", "rgb_m_gain", 3.0)
        self._set_slider("reference", "light_strength", 0.1)

        self.dlg._reset_all()
        self.app.processEvents()

        for key, (_h, rows) in self.dlg.sections.items():
            for r in rows:
                if r.name == "master" or r.name.startswith("amount_"):
                    continue
                self.assertAlmostEqual(r.value(), r.default, places=5,
                                       msg=f"{key}/{r.name}: {r.value()} ≠ default {r.default}")

    def test_reset_all_triggers_preview(self):
        """Nach _reset_all(): letzter /preview-Aufruf hat Neutralwerte."""
        self._set_slider("tone", "exposure", 2.0)
        self._set_slider("color", "saturation", 0.0)

        self.dlg._reset_all()
        # Wait for the debounced preview to fire
        _wait(self.app, lambda: self.fake.last_preview() is not None, timeout_ms=2000)

        payload = self.fake.last_preview()
        self.assertIsNotNone(payload, "Kein /preview-Aufruf nach _reset_all()")
        vals = payload.get("values", {})
        # Key values should be at default
        self.assertAlmostEqual(vals.get("exposure", 0.0), 0.0, places=5,
                               msg=f"exposure={vals.get('exposure')} ≠ 0.0 nach Reset")
        self.assertAlmostEqual(vals.get("saturation", 1.0), 1.0, places=5,
                               msg=f"saturation={vals.get('saturation')} ≠ 1.0 nach Reset")

    def _set_slider(self, section_key: str, name: str, value: float) -> None:
        for r in self.dlg.sections[section_key][1]:
            if r.name == name:
                r.set_value(value)
                return
        raise AssertionError(f"Slider {name} not found in section {section_key}")


class TestPythonExe(unittest.TestCase):
    """python_exe() überspringt WindowsApps-Stubs und findet LOCALAPPDATA-Python."""

    def test_skips_windowsapps_finds_localappdata(self):
        from ref_grade import config as cfg
        with tempfile.TemporaryDirectory() as td:
            pydir = Path(td) / "Programs" / "Python" / "Python312"
            pydir.mkdir(parents=True)
            exe = pydir / "python.exe"
            exe.write_text("")
            env = {"LOCALAPPDATA": td, "REFGRADE_PYTHON": ""}
            with unittest.mock.patch(
                "shutil.which",
                return_value=r"C:\Users\user\AppData\Local\Microsoft\WindowsApps\PythonManager\python.exe",
            ):
                with unittest.mock.patch.dict(os.environ, env):
                    result = cfg.python_exe()
            self.assertEqual(result, str(exe))

    def test_refgrade_python_wins(self):
        from ref_grade import config as cfg
        with unittest.mock.patch.dict(os.environ, {"REFGRADE_PYTHON": "/custom/python"}):
            result = cfg.python_exe()
        self.assertEqual(result, "/custom/python")


if __name__ == "__main__":
    unittest.main()
