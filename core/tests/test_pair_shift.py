"""K28c: pair_shift — Phasenkorrelations-Verschiebung der Kantenbilder."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.fit_pair import pair_shift


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _noise_image(dev: torch.device, h: int = 768, w: int = 512, seed: int = 0) -> torch.Tensor:
    """Zufallsrauschen [1,h,w,3] 0..1 — scharfe Kanten, eindeutig korrelierbar."""
    torch.manual_seed(seed)
    return torch.rand(1, h, w, 3, device=dev)


class TestPairShift(unittest.TestCase):
    """pair_shift: Verschiebungserkennung per Phasenkorrelation."""

    def test_shift_12_5(self):
        """Bild und dasselbe Bild um (12, 5) px verschoben → 13 ±1."""
        dev = _get_device()
        img = _noise_image(dev).to(dev)
        after = torch.roll(img, shifts=(5, 12), dims=(1, 2))
        self.assertAlmostEqual(pair_shift(img, after), 13.0, delta=1.0)

    def test_identical(self):
        """Identisches Bild → < 0.5 px."""
        dev = _get_device()
        img = _noise_image(dev).to(dev)
        self.assertLess(pair_shift(img, img), 0.5)

    def test_color_change(self):
        """Gleiches Bild mit Farbänderung (Gain 1.3 auf R) → < 1 px."""
        dev = _get_device()
        img = _noise_image(dev).to(dev)
        after = img.clone()
        after[..., 0] = (after[..., 0] * 1.3).clamp(0, 1)
        self.assertLess(pair_shift(img, after), 1.0)


if __name__ == "__main__":
    unittest.main()
