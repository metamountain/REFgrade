"""AUTO-Neuentwurf (refgrade.auto2, Opus 2026-10-13): Ziel erreichen, gute Bilder in Ruhe lassen,
nichts doppelt korrigieren. Feste Sollwerte, synthetische Bilder (unabhängig von FiveK)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refgrade import auto as auto2  # noqa: E402
from refgrade.color import luma, rgb_to_lab  # noqa: E402
from refgrade.curves import apply as curves_apply  # noqa: E402
from refgrade.grade import grade  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402

_D = {s.name: s.default for s in SLIDERS}
_Q = torch.tensor(auto2.Q)


def _from_quantiles(qv: list[float], n: int = 128) -> torch.Tensor:
    """Graues Bild, dessen Luma-Quantile bei auto2.Q genau qv sind (stückweise linear)."""
    xs = [0.0] + list(auto2.Q) + [1.0]
    ys = [max(0.0, qv[0] - 0.02)] + list(qv) + [min(1.0, qv[-1] + 0.02)]
    t = torch.linspace(0, 1, n * n)
    v = torch.zeros_like(t)
    for i in range(len(xs) - 1):
        m = (t >= xs[i]) & (t <= xs[i + 1])
        v[m] = ys[i] + (t[m] - xs[i]) / (xs[i + 1] - xs[i]) * (ys[i + 1] - ys[i])
    return v.reshape(1, n, n, 1).expand(1, n, n, 3).clone()


def _quantiles(img: torch.Tensor) -> list[float]:
    return torch.quantile(luma(img).flatten(), _Q).tolist()


def _cast(img: torch.Tensor) -> float:
    lab = rgb_to_lab(img)
    m = (lab[..., 0] > 15) & (lab[..., 0] < 95)
    return float(torch.sqrt(lab[..., 1][m].mean() ** 2 + lab[..., 2][m].mean() ** 2))


_MID = [(lo + hi) / 2 for lo, hi in zip(auto2.BAND_LO, auto2.BAND_HI)]
_DARK = [v * 0.45 for v in _MID]


class TestTone(unittest.TestCase):
    def test_dark_image_lands_in_band(self):
        x = _from_quantiles(_DARK)
        v = auto2.auto(x, "tone")
        q = _quantiles(grade(x, {**_D, **v}).clamp(0, 1))
        self.assertGreaterEqual(q[3], auto2.BAND_LO[3] - 0.03, f"Median {q[3]:.3f}")
        self.assertLessEqual(q[3], auto2.BAND_HI[3] + 0.03, f"Median {q[3]:.3f}")

    def test_image_in_band_is_left_alone(self):
        v = auto2.auto(_from_quantiles(_MID), "tone")
        for n, val in v.items():
            self.assertAlmostEqual(val, _D[n], places=3, msg=n)


class TestColor(unittest.TestCase):
    def test_cast_removed_down_to_tolerance(self):
        x = _from_quantiles(_MID) * torch.tensor([1.12, 1.0, 0.82])
        self.assertGreater(_cast(x.clamp(0, 1)), 8.0)
        v = auto2.auto(x.clamp(0, 1), "color")
        self.assertLessEqual(_cast(grade(x.clamp(0, 1), {**_D, **v}).clamp(0, 1)), auto2.CAST_OK + 0.6)

    def test_neutral_image_is_left_alone(self):
        v = auto2.auto(_from_quantiles(_MID), "color")
        for n, val in v.items():
            self.assertAlmostEqual(val, _D[n], places=3, msg=n)


class TestCurves(unittest.TestCase):
    def test_dark_image_median_into_band(self):
        x = _from_quantiles(_DARK)
        c = auto2.auto(x, "curves")["curves"]
        q = _quantiles(curves_apply(x, c).clamp(0, 1))
        self.assertGreaterEqual(q[3], auto2.BAND_LO[3] - 0.03)

    def test_very_bright_or_dark_image_gives_a_valid_curve(self):
        """Bench 2026-10-13: 98-%-Quantil rundete auf 1,0 → doppeltes x → PCHIP teilte durch null."""
        from refgrade.curves import validate
        for qv in ([0.6, 0.8, 0.9, 0.97, 0.99, 0.9999, 1.0], [0.0, 0.0, 0.001, 0.01, 0.02, 0.03, 0.05]):
            c = auto2.auto(_from_quantiles(qv), "curves")["curves"]
            validate(c)
            xs = [p[0] for p in c["rgb"]]
            self.assertTrue(all(b > a for a, b in zip(xs, xs[1:])), xs)
            curves_apply(_from_quantiles(qv), c)

    def test_image_in_band_gives_identity(self):
        c = auto2.auto(_from_quantiles(_MID), "curves")["curves"]
        self.assertEqual(c["rgb"], [[0.0, 0.0], [1.0, 1.0]])


class TestNoDoubleCorrection(unittest.TestCase):
    def test_curves_after_tone_do_almost_nothing(self):
        """Erst TONE-AUTO, dann CURVES-AUTO auf dem Ergebnis (Kette) → Kurve fast Identität."""
        x = _from_quantiles(_DARK)
        y = grade(x, {**_D, **auto2.auto(x, "tone")}).clamp(0, 1)
        pts = auto2.auto(y, "curves")["curves"]["rgb"]
        self.assertLess(max(abs(px - py) for px, py in pts), 0.04, pts)


class TestErrors(unittest.TestCase):
    def test_wheels_and_rgb_not_automatic(self):
        """Gemessen 2026-10-13: beide schaden schon guten Bildern (ΔE 5–30) ohne Nutzen."""
        self.assertEqual(auto2.AUTO_SECTIONS, ("color", "tone", "curves"))
        for sec in ("wheels", "rgb"):
            with self.assertRaises(ValueError):
                auto2.auto(_from_quantiles(_MID), sec)


if __name__ == "__main__":
    unittest.main()
