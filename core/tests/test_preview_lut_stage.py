"""K27f: LUT-stage preview matches full-stage within 2/255 (no clarity)."""

from __future__ import annotations

import unittest

import torch

from refgrade import session as sess
from refgrade.session import Session


def _make_test_image(h: int = 64, w: int = 64) -> torch.Tensor:
    """Checkerboard (1-px lines) + horizontal gradient, [1,H,W,3] float32 0..1."""
    img = torch.zeros(1, h, w, 3, dtype=torch.float32)
    # Checkerboard: 1-px black/white lines
    for y in range(h):
        for x in range(w):
            img[0, y, x, :] = 1.0 if (x + y) % 2 == 0 else 0.0
    # Add a horizontal gradient on top (blend 50 %)
    grad = torch.linspace(0.0, 1.0, w, dtype=torch.float32)
    grad = grad[None, None, :, None].expand(1, h, w, 3)
    img = 0.5 * img + 0.5 * grad
    return img.clamp(0.0, 1.0)


def _make_session(img: torch.Tensor) -> Session:
    """Create a minimal session for testing."""
    s = Session.__new__(Session)
    s.source = img
    s.source_full = img
    s.alpha = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
    s.ignore = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)
    s.hists = {}
    s.previews = {}
    s.luts = {}
    s.method_fns = {}
    s.section_grid_cache = {}
    s.depth = 8
    return s


class TestLutStage(unittest.TestCase):
    """lut-stage preview matches full-stage (clarity=0) within 2/255."""

    def test_lut_matches_full(self):
        img = _make_test_image()
        s = _make_session(img)
        refs = []
        method = "reinhard"
        values = {
            "exposure": 0.5,
            "contrast": 1.3,
            "temperature": 20.0,
            "clarity": 0.0,
            "light_strength": 1.0,
            "color_strength": 1.0,
        }
        # RGB curve with shadow lift (spec: "Kurve RGB mit Schatten-Anhebung")
        mix = {
            "weights": {"curves": 1.0},
            "master": 1.0,
            "solo": None,
            "curves": {"rgb": [[0.0, 0.1], [0.5, 0.5], [1.0, 1.0]]},
        }
        max_side = img.shape[2]  # no downscale

        result_lut, _ = sess.preview(s, refs, method, values, max_side, mix, stage="lut")
        result_full, _ = sess.preview(s, refs, method, values, max_side, mix, stage="full")

        # Same size
        self.assertEqual(result_lut.shape, result_full.shape)
        # Max deviation ≤ 2/255
        max_dev = (result_lut - result_full).abs().max().item()
        self.assertLessEqual(max_dev, 2.0 / 255.0,
                             f"lut vs full deviation {max_dev:.6f} > 2/255")

    def test_lut_skips_clarity(self):
        """With clarity ≠ 0, lut and full differ (clarity is spatial)."""
        img = _make_test_image()
        s = _make_session(img)
        refs = []
        method = "reinhard"
        values = {
            "exposure": 0.0,
            "contrast": 1.0,
            "clarity": 0.5,
            "light_strength": 1.0,
            "color_strength": 1.0,
        }
        mix = None
        max_side = img.shape[2]

        result_lut, _ = sess.preview(s, refs, method, values, max_side, mix, stage="lut")
        result_full, _ = sess.preview(s, refs, method, values, max_side, mix, stage="full")

        # They should differ because clarity is spatial
        max_dev = (result_lut - result_full).abs().max().item()
        self.assertGreater(max_dev, 2.0 / 255.0,
                           "clarity should make lut and full differ")


if __name__ == "__main__":
    unittest.main()
