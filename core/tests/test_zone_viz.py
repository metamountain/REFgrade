"""Tests für die ZoneVisualizer (B/W-Zonenkarte unter den Wheels, 2026-10-18).

Prüft:
1. Die Maskenberechnung passt exakt zur Kern-Formel in ``refgrade/grade.py::_zone_masks``
   (Cross-Check: die Schattenmaske wird über ein bekanntes ``lift_r`` aus dem echten
   Kern extrahiert und mit ``ZoneVisualizer._masks`` verglichen).
2. Grosse Lücken im Standard (0.1/0.1): in den Lücken wirkt keine Maske.
3. Gain vergrößert die Kerne, Falloff die weichen Ränder.
4. Gain + Falloff schließen die Lücken.
5. Das Widget rendert ohne Crash.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

# Package-Modul ohne __init__.py-Ausführung (krita fehlt im Systempython)
_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import torch  # noqa: E402  (vor PyQt6, DLL-Konflikt)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.zone_viz import ZoneVisualizer, _ss  # noqa: E402


class TestSmoothstep(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(_ss(0.0, 1.0, -1.0), 0.0)
        self.assertEqual(_ss(0.0, 1.0, 0.0), 0.0)
        self.assertEqual(_ss(0.0, 1.0, 1.0), 1.0)
        self.assertEqual(_ss(0.0, 1.0, 2.0), 1.0)
        self.assertAlmostEqual(_ss(0.0, 1.0, 0.5), 0.5, places=6)

    def test_monotonic(self):
        vals = [_ss(0.2, 0.8, i / 100) for i in range(101)]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b)


class TestMasks(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.v = ZoneVisualizer()

    def test_big_gaps_at_default(self):
        """Standard (0.1/0.1): in Y=0.25 und Y=0.75 wirkt keine Maske (grosse Lücken)."""
        for y in (0.25, 0.75):
            s, m, h = self.v._masks(y)
            self.assertLess(s, 0.05, f"Y={y}: S={s}")
            self.assertLess(h, 0.05, f"Y={y}: H={h}")
        # Mittenkern: Y=0.5 voll M, keine S/H
        s, m, h = self.v._masks(0.5)
        self.assertGreater(m, 0.9)
        self.assertLess(s, 0.05)
        self.assertLess(h, 0.05)

    def test_zone_cores_at_default(self):
        """Standard: Kern-Zentren voll weiß — Y=0.05 (S), Y=0.5 (M), Y=0.95 (H)."""
        s, m, h = self.v._masks(0.05)
        self.assertGreater(s, 0.95)
        self.assertLess(m, 0.05)
        self.assertLess(h, 0.05)
        s, m, h = self.v._masks(0.95)
        self.assertGreater(h, 0.95)
        self.assertLess(s, 0.05)
        self.assertLess(m, 0.05)

    def test_gain_grows_cores(self):
        """gain=0.5, falloff=0: Lichterkern [0.5,1] → Y=0.6 voll H; gain=0.1 nicht."""
        self.v.set_gain(0.5)
        self.v.set_falloff(0.0)
        _, _, h_big = self.v._masks(0.6)
        self.v.set_gain(0.1)
        _, _, h_small = self.v._masks(0.6)
        self.assertGreater(h_big, 0.95)
        self.assertLess(h_small, 0.05)

    def test_falloff_widens_edges(self):
        """falloff=0.3: Lichter-Falloff reicht bis Y=0.6; falloff=0.1 nur bis 0.8."""
        self.v.set_gain(0.1)
        self.v.set_falloff(0.3)
        _, _, h_wide = self.v._masks(0.65)
        self.v.set_falloff(0.1)
        _, _, h_narrow = self.v._masks(0.65)
        self.assertGreater(h_wide, 0.05)
        self.assertLess(h_narrow, 0.05)

    def test_gain_and_falloff_close_gaps(self):
        """gain=0.3, falloff=0.3: keine Lücke mehr — Y=0.25 und Y=0.75 werden getroffen."""
        self.v.set_gain(0.3)
        self.v.set_falloff(0.3)
        s, _, _ = self.v._masks(0.25)
        _, _, h = self.v._masks(0.75)
        self.assertGreater(s, 0.05, f"Y=0.25: S={s}")
        self.assertGreater(h, 0.05, f"Y=0.75: H={h}")

    def test_matches_core_shadow_mask(self):
        """Die Schattenmaske muss den echten Kern grade.py::_zone_masks matchen (via lift_r)."""
        core_dir = Path(__file__).resolve().parents[1]
        if str(core_dir) not in sys.path:
            sys.path.insert(0, str(core_dir))
        from refgrade.grade import grade, _defaults
        N = 128
        y = torch.linspace(0.0, 1.0, N)
        # channel-last [B,H,W,3] (luma() liest die letzte Dimension): jedes Pixel ist neutral grau
        img = y.unsqueeze(-1).repeat(1, 3).unsqueeze(0).unsqueeze(0)   # (1,1,N,3)
        d = _defaults()
        L = 0.3
        # (0.0, 0.0) fällt weg: Kern = einzelner Punkt bei Y=0, den der Master-Clamp (EPS)
        # verlässt — degenerierter Fall, dort wirkt das Rad ohnehin nicht.
        for gain, falloff in ((0.1, 0.1), (0.1, 0.0), (0.3, 0.3), (0.5, 0.5), (0.0, 0.4)):
            vals = {**d, "wheel_gain": gain, "wheel_falloff": falloff, "lift_r": L}
            out_red = grade(img, vals)[0, 0, :, 0]   # roter Kanal über die N Pixel
            ch = y
            s_core = ((out_red - ch) / (L * (1.0 - ch).clamp_min(1e-6))).clamp(0.0, 1.0)
            self.v.set_gain(gain)
            self.v.set_falloff(falloff)
            # Extraction ist nur gültig wo (1-ch) nicht ≈0: Lift-Term lift·(1-ch)·S wird bei
            # ch→1 zu 0, S ist dort nicht mehr rückgewinnbar. Vergleich nur für y < 0.9.
            valid = [i for i in range(N) if y[i].item() < 0.9]
            max_diff = max(abs(self.v._masks(i / (N - 1))[0] - s_core[i].item())
                           for i in valid)
            self.assertLess(max_diff, 0.02,
                            f"gain={gain} falloff={falloff}: max_diff={max_diff}")


class TestRender(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.v = ZoneVisualizer()

    def test_size(self):
        self.assertEqual(self.v.height(), ZoneVisualizer.BAR_H * 3 + ZoneVisualizer.GAP * 2)

    def test_renders_without_crash(self):
        self.v.resize(300, self.v.height())
        self.v.set_gain(0.3)
        self.v.set_falloff(0.2)
        pix = self.v.grab()
        self.assertTrue(pix.width() >= 300)
        self.assertTrue(pix.height() >= ZoneVisualizer.BAR_H * 3)


if __name__ == "__main__":
    unittest.main()
