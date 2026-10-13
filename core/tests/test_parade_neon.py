"""Neon-Parade (Nutzerwahl 2026-10-11: glow 6 + „Plasma-Weiß“): dicht glüht weiß-heiß,
dünn bleibt dunkel und farbig, leer bleibt schwarz, Schein neben den Spuren."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.scope import PARADE_COLORS, parade_neon  # noqa: E402


def _counts() -> torch.Tensor:
    """256×256: Zeile 200 dicht, Spalte 31 extrem (selten, < 0,5 % der
    belegten Zellen, wie ein echter Spitzenwert), Zeile 100 dünn, Rest leer."""
    c = torch.zeros(256, 256)
    c[200, :] = 1.0
    c[200, 31] = 4.0           # extreme Dichte (1 Zelle) → einziger Ort für Weiß
    c[100, :] = 0.02
    return c / c.max()


class TestParadeNeon(unittest.TestCase):
    def setUp(self):
        self.rgb = parade_neon(_counts(), PARADE_COLORS["r"])   # [256,256,3], Zeile 0 oben

    def _px(self, level: int, col: int) -> torch.Tensor:
        return self.rgb[255 - level, col]

    def test_shape_range(self):
        self.assertEqual(tuple(self.rgb.shape), (256, 256, 3))
        self.assertGreaterEqual(float(self.rgb.min()), 0.0)
        self.assertLessEqual(float(self.rgb.max()), 1.0)
        self.assertFalse(torch.isnan(self.rgb).any())

    def test_dense_brighter_than_sparse(self):
        dense, sparse = self._px(200, 128), self._px(100, 128)
        self.assertGreater(float(dense[0]), 0.6)
        self.assertLess(float(sparse[0]), 0.35)
        self.assertGreater(float(dense[0]), 2 * float(sparse[0]))

    def test_empty_stays_black(self):
        self.assertLess(float(self._px(20, 128).max()), 0.02)

    def test_glow_beside_dense_trace(self):
        """Halo: neben der dichten Zeile leuchtet es (keine Daten dort), weit weg nicht."""
        beside, far = float(self._px(203, 128)[0]), float(self._px(240, 128)[0])
        self.assertGreater(beside, 0.03)
        self.assertGreater(beside, 5 * far)

    def test_plasma_white_on_dense_trace_only(self):
        """Dichte Spur: weiß-heißer Kern (G deutlich angehoben, R bleibt vorn); dünne Spur: Kanalfarbe."""
        dense, sparse = self._px(200, 128), self._px(100, 128)
        self.assertGreater(float(dense[1]), 0.4)
        self.assertGreaterEqual(float(dense[0]), float(dense[1]))
        self.assertLess(float(sparse[1]), 0.1)
        self.assertGreater(float(sparse[0]), 2 * float(sparse[1]))

    def test_outlier_does_not_set_scale(self):
        """Ein einzelner Ausreißer darf die dichte Zeile nicht abdunkeln (99,5-%-Perzentil)."""
        c = _counts()
        c[10, 10] = 1000.0
        rgb = parade_neon(c / c.max(), PARADE_COLORS["r"])
        self.assertGreater(float(rgb[255 - 200, 128, 0]), 0.6)


if __name__ == "__main__":
    unittest.main()
