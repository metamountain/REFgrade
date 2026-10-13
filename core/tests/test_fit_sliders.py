"""Tests für refgrade.fit_sliders: Regler-Fit per autograd."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.fit_sliders import fit_sliders
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from refgrade.metrics import delta_e00
from slow import slow  # noqa: E402


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _make_test_image(dev: torch.device, seed: int = 42) -> torch.Tensor:
    """Testbild: [1, 64, 64, 3] mit Struktur."""
    torch.manual_seed(seed)
    img = torch.rand(1, 64, 64, 3, device=dev)
    # Leichte Struktur hinzufügen
    x = torch.linspace(0, 1, 64, device=dev)
    grad = x[None, :, None].expand(1, 64, 64, 1)
    img = img * 0.7 + grad * 0.3
    return img.clamp(0, 1)


class TestFitSliders(unittest.TestCase):

    @slow

    def test_known_sliders_recovery(self):
        """Vorlage = Bild mit bekannten Reglerwerten → ΔE00 < 1.5 (Entscheidung 2026-10-07)."""
        dev = _get_device()
        torch.manual_seed(123)
        img = _make_test_image(dev)

        # Bekannte Reglerwerte anwenden
        vals = _defaults()
        vals["exposure"] = 0.3
        vals["temperature"] = 0.15
        vals["contrast"] = 1.2
        vals["saturation"] = 1.1
        reference = grade(img, vals).clamp(0, 1)

        # Fit
        fitted, log = fit_sliders(img, reference, max_steps=200)

        # Ergebnis mit gefitteten Werten
        result = grade(img, fitted).clamp(0, 1)

        # ΔE00 zwischen Ergebnis und Vorlage
        mean_de, p95_de = delta_e00(result, reference)
        self.assertLess(mean_de, 1.5, f"ΔE00 zu hoch: {mean_de:.3f}")

    def test_identity_reference(self):
        """Vorlage = Bild selbst → Regler nahe Default."""
        dev = _get_device()
        torch.manual_seed(456)
        img = _make_test_image(dev)

        fitted, log = fit_sliders(img, img, max_steps=100)

        # Regler sollten nahe am Default sein
        d = _defaults()
        for name, val in fitted.items():
            if name in d:
                s = next(s for s in SLIDERS if s.name == name)
                range_ = s.max - s.min
                self.assertLess(
                    abs(val - d[name]), range_ * 0.1,
                    f"{name}: {val:.4f} vs Default {d[name]:.4f}"
                )

    @slow

    def test_deterministic(self):
        """Zwei Läufe → bitgleiches Ergebnis."""
        dev = _get_device()
        torch.manual_seed(789)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 0.2}).clamp(0, 1)

        fitted1, _ = fit_sliders(img, ref, max_steps=100)
        fitted2, _ = fit_sliders(img, ref, max_steps=100)

        for name in fitted1:
            self.assertAlmostEqual(fitted1[name], fitted2[name], places=6,
                                   msg=f"{name} nicht deterministisch")

    @slow

    def test_values_in_range(self):
        """Alle gefitteten Werte im SLIDERS-Bereich."""
        dev = _get_device()
        torch.manual_seed(321)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 0.5, "temperature": -0.3}).clamp(0, 1)

        fitted, _ = fit_sliders(img, ref, max_steps=150)

        for s in SLIDERS:
            if s.group == "match":
                continue
            if s.name in fitted:
                self.assertGreaterEqual(fitted[s.name], s.min - 1e-6,
                                        f"{s.name} unter min")
                self.assertLessEqual(fitted[s.name], s.max + 1e-6,
                                     f"{s.name} über max")

    @slow

    def test_loss_decreases(self):
        """Verlust nimmt über die Schritte ab (mindestens am Anfang)."""
        dev = _get_device()
        torch.manual_seed(555)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 0.4, "contrast": 1.3}).clamp(0, 1)

        _, log = fit_sliders(img, ref, max_steps=100)

        # Verlust sollte am Ende niedriger sein als am Anfang
        self.assertLess(log[-1], log[0],
                        f"Verlust nimmt nicht ab: start={log[0]:.4f}, end={log[-1]:.4f}")

    @slow

    def test_no_nan(self):
        """Kein NaN/Inf im Ergebnis."""
        dev = _get_device()
        torch.manual_seed(777)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 1.0, "saturation": 2.0}).clamp(0, 1)

        fitted, log = fit_sliders(img, ref, max_steps=100)

        for name, val in fitted.items():
            self.assertFalse(torch.isnan(torch.tensor(val)), f"{name} ist NaN")
            self.assertFalse(torch.isinf(torch.tensor(val)), f"{name} ist Inf")
        for v in log:
            self.assertFalse(math.isnan(v), "Verlust ist NaN")
            self.assertFalse(math.isinf(v), "Verlust ist Inf")

    def test_nan_mid_fit_keeps_last_finite_values(self):
        """Ein NaN im Verlauf (Verlust/Gradient) darf nicht als NaN-Regler herauskommen."""
        import refgrade.fit_sliders as fs
        dev = _get_device()
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 1.0}).clamp(0, 1)
        calls = {"n": 0}

        def poisoned(x, values):
            calls["n"] += 1
            y = grade(x, values)
            return y * float("nan") if calls["n"] > 5 else y

        original = fs.grade
        fs.grade = poisoned
        try:
            fitted, log = fit_sliders(img, ref, max_steps=50)
        finally:
            fs.grade = original
        self.assertEqual(len(log), 5)
        for name, val in fitted.items():
            self.assertTrue(math.isfinite(val), f"{name} ist {val}")
        for v in log:
            self.assertTrue(math.isfinite(v))


import math


if __name__ == "__main__":
    unittest.main()
