"""K29a: HSL-Sättigung luminanzerhaltend (Rec. 709) statt HSV-Skalierung.

Verifiziert die festen Sollwerte aus spec/73_K29a_hsl_saettigung.md.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.grade import grade  # noqa: E402


def px(r, g, b):
    return torch.tensor([r, g, b], dtype=torch.float32).view(1, 1, 1, 3)


def luma(t):
    return 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]


class TestHSLumaSaturation(unittest.TestCase):

    def test_red_sat_minus1_gives_luma_grey(self):
        """sel_red_sat=-1 auf (0.8, 0.2, 0.2): alle Kanäle = Y ± 0.01."""
        out = grade(px(0.8, 0.2, 0.2), {"sel_red_sat": -1.0})
        Y = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.2  # 0.32756
        for i in range(3):
            self.assertAlmostEqual(float(out[0, 0, 0, i]), Y, delta=0.01,
                                   msg=f"Kanal {i}: {float(out[0, 0, 0, i]):.4f} ≠ Y={Y:.4f}")

    def test_red_sat_plus05_preserves_luma(self):
        """sel_red_sat=+0.5 auf (0.6, 0.4, 0.4): Luma gleich ± 0.01, R steigt, G/B sinken."""
        img = px(0.6, 0.4, 0.4)
        out = grade(img, {"sel_red_sat": 0.5})
        Y_before = float(luma(img)[0, 0, 0])
        Y_after = float(luma(out)[0, 0, 0])
        self.assertAlmostEqual(Y_after, Y_before, delta=0.01,
                               msg=f"Luma: {Y_before:.4f} → {Y_after:.4f}")
        self.assertGreater(float(out[0, 0, 0, 0]), 0.6, "R muss steigen")
        self.assertLess(float(out[0, 0, 0, 1]), 0.4, "G muss sinken")
        self.assertLess(float(out[0, 0, 0, 2]), 0.4, "B muss sinken")

    def test_red_hue_plus30_unchanged(self):
        """sel_red_hue=+30 auf (0.8, 0.2, 0.2) → (0.8, 0.5, 0.2) ± 0.01 (unverändert)."""
        out = grade(px(0.8, 0.2, 0.2), {"sel_red_hue": 30.0})
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 0.8, delta=0.01)
        self.assertAlmostEqual(float(out[0, 0, 0, 1]), 0.5, delta=0.01)
        self.assertAlmostEqual(float(out[0, 0, 0, 2]), 0.2, delta=0.01)

    def test_blue_sat_minus1_leaves_red_untouched(self):
        """sel_blue_sat=-1 lässt (0.8, 0.2, 0.2) unverändert (± 1e-6)."""
        img = px(0.8, 0.2, 0.2)
        out = grade(img, {"sel_blue_sat": -1.0})
        self.assertLess(float((out - img).abs().max()), 1e-6)

    def test_grey_unchanged_at_all_sat(self):
        """Graues Pixel (0.5, 0.5, 0.5) bei sat=±1 unverändert (± 1e-6)."""
        img = px(0.5, 0.5, 0.5)
        for sat in (-1.0, 1.0):
            out = grade(img, {"sel_red_sat": sat, "sel_blue_sat": sat})
            self.assertLess(float((out - img).abs().max()), 1e-6,
                            msg=f"sat={sat}: max diff {float((out - img).abs().max())}")


if __name__ == "__main__":
    unittest.main()
