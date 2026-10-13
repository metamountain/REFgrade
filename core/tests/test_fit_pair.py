"""Tests für refgrade.fit_pair: LUT aus Vorher/Nachher."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from refgrade.fit_pair import fit_pair
from refgrade.lut import apply as lut_apply, build
from refgrade.grade import grade
from refgrade.sliders import SLIDERS


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


class TestFitPair(unittest.TestCase):

    def test_identity_pair(self):
        """Identitätspaar → Identitäts-LUT."""
        dev = _get_device()
        torch.manual_seed(100)
        img = _make_test_image(dev)

        result = fit_pair(img, img, size=17)

        # LUT sollte nahe Identität sein
        table = result["table"].cpu().numpy()  # [b][g][r][3]
        N = result["size"]
        # Identität: table[b,g,r] = (r, g, b) / (N-1)
        r_idx = np.arange(N) / (N - 1)
        g_idx = np.arange(N) / (N - 1)
        b_idx = np.arange(N) / (N - 1)
        rr, gg, bb = np.meshgrid(r_idx, g_idx, b_idx, indexing='ij')
        identity = np.stack([rr, gg, bb], axis=-1)
        # table ist [b][g][r][3], identity ist [r][g][b][3]
        identity_bgr = identity.transpose(2, 1, 0, 3)
        diff = np.abs(table - identity_bgr).mean()
        self.assertLess(diff, 0.01, f"Identitätspaar: mittlere Abweichung {diff:.4f}")

    def test_known_lut_recovery(self):
        """Paar aus bekannter LUT → Tabelle zurück (mittlerer Fehler < 3e-3)."""
        dev = _get_device()
        torch.manual_seed(200)

        # Bekannte LUT: leichte Farbverschiebung
        size = 17
        r_vals = torch.linspace(0, 1, size, device=dev)
        g_vals = torch.linspace(0, 1, size, device=dev)
        b_vals = torch.linspace(0, 1, size, device=dev)

        r_3d = r_vals[None, None, :].expand(size, size, size)
        g_3d = g_vals[None, :, None].expand(size, size, size)
        b_3d = b_vals[:, None, None].expand(size, size, size)
        grid = torch.stack([r_3d, g_3d, b_3d], dim=-1)  # [N,N,N,3]

        # Farbverschiebung: R +0.05, B -0.03
        known_table = grid.clone()
        known_table[..., 0] = (known_table[..., 0] + 0.05).clamp(0, 1)
        known_table[..., 2] = (known_table[..., 2] - 0.03).clamp(0, 1)

        known_lut = {"size": size, "table": known_table, "title": "test"}

        # Vorher/Nachher erzeugen
        img = _make_test_image(dev)
        before = img
        after = lut_apply(img, known_lut)

        # Fit
        result = fit_pair(before, after, size=size)

        # Vergleich
        table = result["table"].cpu().numpy()  # [b][g][r][3]
        known = known_table.cpu().numpy()  # [b][g][r][3]

        # Mittlere Abweichung
        diff = np.abs(table - known).mean()
        self.assertLess(diff, 3e-3, f"Known LUT: mittlere Abweichung {diff:.6f}")

    def test_rb_axis(self):
        """R/B-Achsentest: reine R- oder B-Änderung wird korrekt erkannt."""
        dev = _get_device()
        torch.manual_seed(300)

        # Reine R-Änderung: R → R*0.8
        img = _make_test_image(dev)
        before = img
        after = img.clone()
        after[..., 0] = (after[..., 0] * 0.8).clamp(0, 1)

        result = fit_pair(before, after, size=17)
        table = result["table"].cpu().numpy()  # [b][g][r][3]

        # R-Achse: bei r=1, g=0, b=0 → R sollte ~0.8 sein
        # table[b=0, g=0, r=N-1, 0] ≈ 0.8
        N = result["size"]
        r_val = table[0, 0, N - 1, 0]
        self.assertAlmostEqual(r_val, 0.8, delta=0.05,
                               msg=f"R-Achse: erwartet ~0.8, erhalten {r_val:.3f}")

    def test_residual_reported(self):
        """Restfehler und leere Zellen werden gemeldet."""
        dev = _get_device()
        torch.manual_seed(400)
        img = _make_test_image(dev)

        result = fit_pair(img, img, size=17)
        self.assertIn("residual", result)
        self.assertIn("empty_fraction", result)
        self.assertGreaterEqual(result["residual"], 0.0)
        self.assertGreaterEqual(result["empty_fraction"], 0.0)
        self.assertLessEqual(result["empty_fraction"], 1.0)

    def test_deterministic(self):
        """Zwei Läufe → bitgleiches Ergebnis."""
        dev = _get_device()
        torch.manual_seed(500)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 0.2}).clamp(0, 1)

        r1 = fit_pair(img, ref, size=17)
        r2 = fit_pair(img, ref, size=17)

        t1 = r1["table"].cpu().numpy()
        t2 = r2["table"].cpu().numpy()
        self.assertTrue(np.allclose(t1, t2, atol=1e-7),
                        "fit_pair nicht deterministisch")

    def test_no_nan(self):
        """Kein NaN/Inf in der LUT."""
        dev = _get_device()
        torch.manual_seed(600)
        img = _make_test_image(dev)
        ref = grade(img, {**_defaults(), "exposure": 0.5, "saturation": 1.5}).clamp(0, 1)

        result = fit_pair(img, ref, size=17)
        table = result["table"]
        self.assertFalse(torch.isnan(table).any(), "LUT enthält NaN")
        self.assertFalse(torch.isinf(table).any(), "LUT enthält Inf")
        self.assertGreaterEqual(table.min(), 0.0)
        self.assertLessEqual(table.max(), 1.0)


if __name__ == "__main__":
    unittest.main()
