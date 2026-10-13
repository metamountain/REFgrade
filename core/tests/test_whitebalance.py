"""White balance (user 2026-10-08): AUTO and eyedropper set Temperature/Tint so that the
neutral areas end up neutral — exactly the inverse of grade._temp_tint."""

import os
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from refgrade import color, whitebalance  # noqa: E402
from refgrade.grade import grade  # noqa: E402


def _cast_scene(da=4.0, db=12.0):
    g = torch.rand(1, 64, 64, 3, generator=torch.Generator().manual_seed(0)) * 0.3 + 0.4
    g[...] = g.mean(-1, keepdim=True)                      # neutral greys
    g[:, :20, :20] = torch.tensor([0.8, 0.1, 0.1])         # a colourful object
    lab = color.rgb_to_lab(g)
    lab[..., 1] += da
    lab[..., 2] += db
    return color.lab_to_rgb(lab).clamp(0, 1)


class TestWhiteBalanceCore(unittest.TestCase):

    def test_auto_values(self):
        """Gelbstich (+b*) → kühler (Richtung, 2026-10-13: von Kries; Tint hängt beim Mischstich
        von der Temperatur ab — die Werte selbst prüft test_auto_neutralises_greys über den Rückweg)."""
        v = whitebalance.estimate(_cast_scene())
        self.assertLess(v["temperature"], -0.1)

    def test_auto_neutralises_greys(self):
        img = _cast_scene()
        lab = color.rgb_to_lab(grade(img, whitebalance.estimate(img)))[:, 30:, 30:]
        self.assertLess(abs(float(lab[..., 1].median())), 0.5)
        self.assertLess(abs(float(lab[..., 2].median())), 0.5)

    def test_neutral_image_gives_zero(self):
        g = torch.full((1, 16, 16, 3), 0.5)
        self.assertEqual(whitebalance.estimate(g), {"temperature": 0.0, "tint": 0.0})

    def test_pick_patch(self):
        """Pipette: der gepickte Fleck wird neutral (Rückweg über grade, 2026-10-13: von Kries)."""
        img = _cast_scene()
        v = whitebalance.sample(img, 50, 50)
        lab = color.rgb_to_lab(grade(img, v))[:, 46:55, 46:55]
        self.assertLess(abs(float(lab[..., 1].mean())), 0.5)
        self.assertLess(abs(float(lab[..., 2].mean())), 0.5)

    def test_pick_outside_raises(self):
        with self.assertRaises(ValueError):
            whitebalance.sample(_cast_scene(), 500, 5)


class TestWhiteBalanceDialog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
        pkg = types.ModuleType("ref_grade")
        pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
        sys.modules.setdefault("ref_grade", pkg)
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _dialog(self, calls):
        from PyQt6.QtGui import QImage
        from ref_grade.dialog import RefGradeDialog

        def client(_port, path, payload=None):
            calls.append((path, payload))
            if path == "/whitebalance":
                return {"temperature": -0.4, "tint": 0.1}
            return {"items": []} if path == "/references" else {}
        img = QImage(200, 100, QImage.Format.Format_RGB32)
        img.fill(0x808080)
        d = RefGradeDialog(img, None, client=client,
                           source={"width": 2000, "height": 1000, "session": "s1"})
        d.resize(1920, 1080)
        d.show()
        self.app.processEvents()
        return d

    def _pump(self, cond):
        import time
        t = time.time()
        while time.time() - t < 2 and not cond():
            self.app.processEvents()
            time.sleep(0.01)

    def test_auto_sets_sliders_and_turns_color_on(self):
        calls = []
        d = self._dialog(calls)
        d._wb_auto_btn.click()
        self._pump(lambda: any(p == "/whitebalance" for p, _ in calls))
        rows = {r.name: r for r in d.sections["color"][1]}
        self._pump(lambda: abs(rows["temperature"].value() + 0.4) < 1e-6)
        self.assertAlmostEqual(rows["temperature"].value(), -0.4, places=6)
        self.assertAlmostEqual(rows["tint"].value(), 0.1, places=6)
        self.assertTrue(d.sections["color"][0].is_on())
        d.close()

    def test_pick_click_sends_full_image_point(self):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest
        calls = []
        d = self._dialog(calls)
        d._wb_pick_btn.setChecked(True)
        v = d.view
        target = (500.0, 250.0)                               # full-image px
        pos = QPoint(round((target[0] - v.offset.x()) * v.zoom),
                     round((target[1] - v.offset.y()) * v.zoom))
        QTest.mouseClick(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos)
        self._pump(lambda: any(p == "/whitebalance" for p, _ in calls))
        payload = next(pl for p, pl in calls if p == "/whitebalance")
        self.assertEqual(payload["mode"], "pick")
        self.assertAlmostEqual(payload["point"][0], target[0], delta=2 / v.zoom)
        self.assertAlmostEqual(payload["point"][1], target[1], delta=2 / v.zoom)
        self.assertFalse(d._wb_pick_btn.isChecked())          # one click, then off
        d.close()


if __name__ == "__main__":
    unittest.main()
