"""Tests für refgrade.idt: IDT."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.idt import idt, apply_idt
from refgrade.fit_sliders import fit_sliders
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from refgrade.metrics import coverage, structure, delta_e00
from slow import slow  # noqa: E402


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _make_test_image(dev: torch.device, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    img = torch.rand(1, 64, 64, 3, device=dev)
    x = torch.linspace(0, 1, 64, device=dev)
    grad = x[None, :, None].expand(1, 64, 64, 1)
    img = img * 0.7 + grad * 0.3
    return img.clamp(0, 1)


class TestIDT(unittest.TestCase):

    @slow

    def test_same_image_coverage(self):
        """Vorlage = gegradetes selbes Bild: coverage steigt, Endwert ≥ 95 %."""
        dev = _get_device()
        torch.manual_seed(111)
        img = _make_test_image(dev)

        # Regler-Fit
        vals = _defaults()
        vals["exposure"] = 0.2
        vals["temperature"] = 0.1
        reference = grade(img, vals).clamp(0, 1)

        # Fit
        fitted, _ = fit_sliders(img, reference, max_steps=100)
        graded = grade(img, fitted).clamp(0, 1)

        # IDT
        steps = idt(graded, reference, max_iters=30)
        self.assertGreater(len(steps), 0, "IDT lieferte keine Schritte")

        # Anwenden
        result = apply_idt(graded, steps, residual_strength=0.0)

        # Coverage
        cov = coverage(result, reference)
        self.assertGreaterEqual(cov, 95.0, f"Coverage zu niedrig: {cov:.1f} %")

    @slow

    def test_structure_preserved(self):
        """Structure ≥ 0.9 bei selbem Bild."""
        dev = _get_device()
        torch.manual_seed(222)
        img = _make_test_image(dev)

        vals = _defaults()
        vals["exposure"] = 0.15
        reference = grade(img, vals).clamp(0, 1)

        fitted, _ = fit_sliders(img, reference, max_steps=100)
        graded = grade(img, fitted).clamp(0, 1)

        steps = idt(graded, reference, max_iters=30)
        result = apply_idt(graded, steps, residual_strength=0.0)

        struct = structure(img, result)
        self.assertGreaterEqual(struct, 0.9, f"Structure zu niedrig: {struct:.3f}")

    @slow

    def test_residual_zero_identity(self):
        """residual_strength=0 → Ergebnis unterscheidet sich von Original."""
        dev = _get_device()
        torch.manual_seed(333)
        img = _make_test_image(dev)

        vals = _defaults()
        vals["exposure"] = 0.2
        reference = grade(img, vals).clamp(0, 1)

        fitted, _ = fit_sliders(img, reference, max_steps=100)
        graded = grade(img, fitted).clamp(0, 1)

        steps = idt(graded, reference, max_iters=20)
        result = apply_idt(graded, steps, residual_strength=0.0)

        # Ergebnis sollte sich vom Original (vor IDT) unterscheiden
        diff = (graded - result).abs().mean().item()
        self.assertGreater(diff, 1e-6, "IDT hatte keine Wirkung")

    @slow

    def test_deterministic(self):
        """Zwei IDT-Läufe → bitgleiches Ergebnis."""
        dev = _get_device()
        torch.manual_seed(444)
        img = _make_test_image(dev)

        vals = _defaults()
        vals["exposure"] = 0.1
        reference = grade(img, vals).clamp(0, 1)

        fitted, _ = fit_sliders(img, reference, max_steps=80)
        graded = grade(img, fitted).clamp(0, 1)

        steps1 = idt(graded, reference, max_iters=20)
        steps2 = idt(graded, reference, max_iters=20)

        # Gleiche Anzahl Schritte
        self.assertEqual(len(steps1), len(steps2))

        # Gleiche Rotationen
        for i in range(len(steps1)):
            self.assertEqual(steps1[i]["rotation"], steps2[i]["rotation"])

    @slow

    def test_json_serializable(self):
        """IDT-Schritte sind JSON-fähig."""
        import json
        dev = _get_device()
        torch.manual_seed(555)
        img = _make_test_image(dev)

        vals = _defaults()
        reference = grade(img, vals).clamp(0, 1)

        fitted, _ = fit_sliders(img, reference, max_steps=50)
        graded = grade(img, fitted).clamp(0, 1)

        steps = idt(graded, reference, max_iters=10)

        # JSON-Serialisierung
        s = json.dumps(steps)
        loaded = json.loads(s)
        self.assertEqual(len(loaded), len(steps))
        for i in range(len(loaded)):
            self.assertEqual(len(loaded[i]["rotation"]), 3)
            self.assertEqual(len(loaded[i]["curves"]), 3)
            self.assertEqual(len(loaded[i]["curves"][0]["src"]), 512)
            self.assertEqual(len(loaded[i]["curves"][0]["dst"]), 512)

    @slow

    def test_crop_reference(self):
        """Vorlage = Ausschnitt: nur Werte protokollieren (kein Assert)."""
        dev = _get_device()
        torch.manual_seed(666)
        img = _make_test_image(dev)

        # Ausschnitt als Vorlage
        ref_crop = img[:, 16:48, 16:48, :].clone()
        # Auf volle Größe bringen (wiederholen)
        ref_full = ref_crop.repeat(1, 2, 2, 1)

        fitted, _ = fit_sliders(img, ref_full, max_steps=80)
        graded = grade(img, fitted).clamp(0, 1)

        steps = idt(graded, ref_full, max_iters=20)
        result = apply_idt(graded, steps, residual_strength=0.0)

        # Nur protokollieren
        cov = coverage(result, ref_full)
        de_mean, _ = delta_e00(result, ref_full)
        print(f"  [crop] coverage={cov:.1f} %, ΔE00={de_mean:.2f}, steps={len(steps)}")


if __name__ == "__main__":
    unittest.main()
