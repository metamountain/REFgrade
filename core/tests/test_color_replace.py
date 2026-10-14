"""Tests für Color Replace: HSL-Distanz-Maske, mehrere Instanzen."""

from __future__ import annotations

import unittest

import torch

from refgrade.color_replace import apply, MAX_INSTANCES
from refgrade.sliders import SLIDERS


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _red_image() -> torch.Tensor:
    """[1,4,4,3] rotes Bild (Hue=0°, S=1, V=1)."""
    return torch.tensor([[[[1.0, 0.0, 0.0]] * 4] * 4])


def _blue_image() -> torch.Tensor:
    """[1,4,4,3] blaues Bild (Hue=240°, S=1, V=1)."""
    return torch.tensor([[[[0.0, 0.0, 1.0]] * 4] * 4])


def _green_image() -> torch.Tensor:
    """[1,4,4,3] grünes Bild (Hue=120°, S=1, V=1)."""
    return torch.tensor([[[[0.0, 1.0, 0.0]] * 4] * 4])


def _grey_image() -> torch.Tensor:
    """[1,4,4,3] graues Bild (S=0)."""
    return torch.tensor([[[[0.5, 0.5, 0.5]] * 4] * 4])


class TestColorReplace(unittest.TestCase):

    def test_default_is_identity(self):
        """Alle Werte auf Default → Bild unverändert."""
        x = _red_image()
        y = apply(x, _defaults())
        torch.testing.assert_close(y, x)

    def test_replace_red_to_blue(self):
        """Rot → Blau: rotes Bild wird blau."""
        x = _red_image()
        vals = _defaults()
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_src_sat"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_tgt_sat"] = 1.0
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_range"] = 30.0
        y = apply(x, vals)
        # Ergebnis sollte blau sein
        self.assertGreater(y[0, 0, 0, 2].item(), 0.9)
        self.assertLess(y[0, 0, 0, 0].item(), 0.1)

    def test_replace_out_of_range_unchanged(self):
        """Grün ist außerhalb des Rot-Bereichs → unverändert."""
        x = _green_image()
        vals = _defaults()
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_src_sat"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_tgt_sat"] = 1.0
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_range"] = 30.0
        y = apply(x, vals)
        torch.testing.assert_close(y, x)

    def test_grey_unchanged(self):
        """Graues Bild (S=0) wird nicht berührt."""
        x = _grey_image()
        vals = _defaults()
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_src_sat"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_tgt_sat"] = 1.0
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_range"] = 180.0
        y = apply(x, vals)
        torch.testing.assert_close(y, x)

    def test_partial_intensity(self):
        """Intensity 0.5 → halb gemischt."""
        x = _red_image()
        vals = _defaults()
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_src_sat"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_tgt_sat"] = 1.0
        vals["cr_0_intensity"] = 0.5
        vals["cr_0_range"] = 30.0
        y = apply(x, vals)
        # R sollte halbiert, B halbiert erhöht sein
        r = y[0, 0, 0, 0].item()
        b = y[0, 0, 0, 2].item()
        self.assertAlmostEqual(r, 0.5, delta=0.1)
        self.assertAlmostEqual(b, 0.5, delta=0.1)

    def test_multiple_instances(self):
        """Zwei Instanzen: Rot→Blau und Grün→Gelb."""
        # Bild mit rot (links) und grün (rechts)
        x = torch.tensor([
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ])
        vals = _defaults()
        # Instanz 0: Rot → Blau
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_range"] = 30.0
        # Instanz 1: Grün → Gelb
        vals["cr_1_src_hue"] = 120.0
        vals["cr_1_intensity"] = 1.0
        vals["cr_1_tgt_hue"] = 60.0
        vals["cr_1_range"] = 30.0
        y = apply(x, vals)
        # Links (rot) → blau
        self.assertGreater(y[0, 0, 0, 2].item(), 0.9)
        # Rechts (grün) → gelb (R hoch, G hoch, B niedrig)
        self.assertGreater(y[0, 0, 2, 0].item(), 0.9)
        self.assertGreater(y[0, 0, 2, 1].item(), 0.9)
        self.assertLess(y[0, 0, 2, 2].item(), 0.1)

    def test_wide_range_covers_more(self):
        """Range 180° deckt die Hälfte des Hue-Kreises ab."""
        # Cyan (Hue=180°) ist genau 180° von Rot (0°) entfernt
        x = torch.tensor([[[[0.0, 1.0, 1.0]] * 4] * 4])  # cyan
        vals = _defaults()
        vals["cr_0_src_hue"] = 0.0
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_tgt_hue"] = 240.0
        vals["cr_0_range"] = 180.0
        y = apply(x, vals)
        # Cyan sollte teilweise beeinflusst sein (Grenze)
        # Mit Range=179° wäre es unverändert
        vals2 = _defaults()
        vals2["cr_0_src_hue"] = 0.0
        vals2["cr_0_intensity"] = 1.0
        vals2["cr_0_tgt_hue"] = 240.0
        vals2["cr_0_range"] = 179.0
        y2 = apply(x, vals2)
        torch.testing.assert_close(y2, x)

    def test_max_instances(self):
        """MAX_INSTANCES = 8."""
        self.assertEqual(MAX_INSTANCES, 8)


if __name__ == "__main__":
    unittest.main()
