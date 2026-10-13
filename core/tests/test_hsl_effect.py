"""Regression 2026-10-08 (user in Krita: "hsl not working"): selective colour had almost no
effect — unwrapped hue difference + a cosine window repeating every 120° made the bands cancel,
and the result was scaled by the pixel's saturation a second time."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.grade import _rgb_to_hsv, grade  # noqa: E402


def px(r, g, b):
    return torch.tensor([r, g, b], dtype=torch.float32).view(1, 1, 1, 3)


class TestHSLEffect(unittest.TestCase):

    def test_blue_desaturate(self):
        out = grade(px(0.2, 0.2, 0.8), {"sel_blue_sat": -1.0})   # hue exactly 240° (band centre)
        _h, s, _v = _rgb_to_hsv(out)
        self.assertLess(float(s), 0.05)                   # blue → grey

    def test_other_band_untouched(self):
        green = px(0.2, 0.7, 0.25)
        out = grade(green, {"sel_blue_sat": -1.0})
        self.assertLess(float((out - green).abs().max()), 1e-4)

    def test_hue_shift_degrees(self):
        out = grade(px(0.2, 0.7, 0.2), {"sel_green_hue": 30.0})  # pure green hue 120°
        h, _s, _v = _rgb_to_hsv(out)
        self.assertAlmostEqual(float(h), 150.0, delta=2.0)

    def test_red_wraps_at_360(self):
        near_red = px(0.8, 0.2, 0.3)                      # hue ≈ 350°
        out = grade(near_red, {"sel_red_sat": -1.0})
        _h, s, _v = _rgb_to_hsv(out)
        self.assertLess(float(s), 0.3)                    # mostly in the red band

    def test_grey_unchanged(self):
        grey = px(0.5, 0.5, 0.5)
        out = grade(grey, {"sel_blue_sat": -1.0, "sel_red_hue": 30.0})
        self.assertLess(float((out - grey).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
