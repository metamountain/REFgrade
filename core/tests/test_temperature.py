"""Regression 2026-10-08 (user in Krita: "temperature not working correct"): the slider
changed brightness — it wrote L* instead of b*. Temperature must warm/cool, not lighten."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade import color  # noqa: E402
from refgrade.grade import grade  # noqa: E402


class TestTemperature(unittest.TestCase):

    def _dlab(self, rgb, values):
        x = torch.tensor(rgb).view(1, 1, 1, 3)
        return (color.rgb_to_lab(grade(x, values)) - color.rgb_to_lab(x))[0, 0, 0].tolist()

    def test_warm_is_yellow_not_brighter(self):
        dl, da, db = self._dlab([0.5, 0.5, 0.5], {"temperature": 0.5})
        # von Kries seit 2026-10-13: Richtung und Helligkeit prüfen, nicht die alte Zahl (b* +20·t)
        self.assertGreater(db, 5.0)
        self.assertAlmostEqual(dl, 0.0, delta=0.5)
        self.assertLess(abs(da), 0.5 * abs(db))   # warm = bernstein, vor allem gelb

    def test_cool_is_blue(self):
        dl, _da, db = self._dlab([0.5, 0.5, 0.5], {"temperature": -0.5})
        self.assertLess(db, -5.0)
        self.assertAlmostEqual(dl, 0.0, delta=0.5)

    def test_warm_raises_red_over_blue(self):
        x = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 1, 3)
        r, _g, b = grade(x, {"temperature": 0.5})[0, 0, 0].tolist()
        self.assertGreater(r, b)


if __name__ == "__main__":
    unittest.main()
