"""Tests für sort_luts.py: synthetische LUTs klassifizieren, --apply verschiebt."""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.lut import build as lut_build, write_cube as lut_write_cube
from tools.sort_luts import _classify_one, sort_luts


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _desaturate(x: torch.Tensor) -> torch.Tensor:
    """Entsättigung: Luma auf alle Kanäle."""
    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return luma.unsqueeze(-1).expand_as(x)


def _warm(x: torch.Tensor) -> torch.Tensor:
    """Wärme: b* +10 → im sRGB-Code: b-Kanal leicht senken, r leicht erhöhen."""
    out = x.clone()
    out[..., 0] = (out[..., 0] + 0.05).clamp(0, 1)
    out[..., 2] = (out[..., 2] - 0.05).clamp(0, 1)
    return out


def _cool(x: torch.Tensor) -> torch.Tensor:
    """Kühl: b* -10 → r-Kanal senken, b-Kanal erhöhen."""
    out = x.clone()
    out[..., 0] = (out[..., 0] - 0.05).clamp(0, 1)
    out[..., 2] = (out[..., 2] + 0.05).clamp(0, 1)
    return out


def _hue_shift_40(x: torch.Tensor) -> torch.Tensor:
    """Farbtonverschiebung um 40° im Lab-Raum."""
    from refgrade.color import rgb_to_lab, lab_to_rgb
    lab = rgb_to_lab(x)
    a, b = lab[..., 1], lab[..., 2]
    angle = math.radians(40.0)
    ca, sa = math.cos(angle), math.sin(angle)
    a_new = a * ca - b * sa
    b_new = a * sa + b * ca
    lab_new = torch.stack([lab[..., 0], a_new, b_new], dim=-1)
    return lab_to_rgb(lab_new).clamp(0, 1)


class TestClassifyOne(unittest.TestCase):
    """_classify_one auf synthetischen LUTs."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")

    def _build_and_classify(self, fn, name: str, title: str = "") -> tuple[str, str]:
        lut = lut_build(fn, size=17, title=title, device=self.device)
        return _classify_one(lut, name, title, self.device)

    def test_desaturation_bw(self):
        cat, reason = self._build_and_classify(_desaturate, "desat.cube")
        self.assertEqual(cat, "B&W")

    def test_warm(self):
        cat, reason = self._build_and_classify(_warm, "warm_test.cube")
        self.assertEqual(cat, "Warm")

    def test_cool(self):
        cat, reason = self._build_and_classify(_cool, "cool_test.cube")
        self.assertEqual(cat, "Cool")

    def test_identity_natural(self):
        cat, reason = self._build_and_classify(_identity, "identity.cube")
        self.assertEqual(cat, "Natural")

    def test_hue_shift_creative(self):
        cat, reason = self._build_and_classify(_hue_shift_40, "hue40.cube")
        self.assertEqual(cat, "Creative")


class TestSortLutsApply(unittest.TestCase):
    """sort_luts mit --apply: Dateien liegen in den Ordnern."""

    def test_apply_moves_files(self):
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_sort_"))
        try:
            self.device = torch.device("cpu")

            # Synthetische LUTs schreiben
            identity = lut_build(_identity, size=17, title="id", device=self.device)
            lut_write_cube(identity, str(tmp / "identity.cube"))

            desat = lut_build(_desaturate, size=17, title="desat", device=self.device)
            lut_write_cube(desat, str(tmp / "desat.cube"))

            warm = lut_build(_warm, size=17, title="warm", device=self.device)
            lut_write_cube(warm, str(tmp / "warm.cube"))

            # Ohne --apply: nichts verschoben
            results = sort_luts(tmp, apply=False)
            self.assertTrue((tmp / "identity.cube").exists())
            self.assertTrue((tmp / "desat.cube").exists())
            self.assertTrue((tmp / "warm.cube").exists())
            self.assertFalse((tmp / "B&W").exists())
            self.assertFalse((tmp / "Warm").exists())
            self.assertFalse((tmp / "Natural").exists())

            # Mit --apply: Dateien verschoben
            results = sort_luts(tmp, apply=True)
            self.assertFalse((tmp / "identity.cube").exists())
            self.assertFalse((tmp / "desat.cube").exists())
            self.assertFalse((tmp / "warm.cube").exists())

            # identity → Natural
            self.assertTrue((tmp / "Natural" / "identity.cube").exists())
            # desat → B&W
            self.assertTrue((tmp / "B&W" / "desat.cube").exists())
            # warm → Warm
            self.assertTrue((tmp / "Warm" / "warm.cube").exists())

            # Log existiert
            self.assertTrue((tmp / "sort_luts.log").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
