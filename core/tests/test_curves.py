"""Tests für Paket K9a: Kurven (PCHIP), HSL-Bereich, Vibrance, Sections, Worker."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from refgrade.curves import apply as curves_apply, is_identity as curves_is_identity, validate as curves_validate
from refgrade.sections import SECTIONS, VISIBLE, FITTABLE
from refgrade.session import (
    Session,
    apply_full,
    is_identity,
    preview,
    values_with_defaults,
)
from refgrade.sliders import SLIDERS
from slow import slow  # noqa: E402

_CORE_DIR = Path(__file__).resolve().parents[1]
_POOL = _CORE_DIR.parent / "bench" / "pool"
_REF = str(_POOL / "01.jpg")


def _make_session(source_full: torch.Tensor, device: torch.device | None = None) -> Session:
    """Session direkt aus Tensor bauen (ohne Rohdatei)."""
    if device is None:
        device = torch.device("cpu")
    h, w = source_full.shape[1], source_full.shape[2]
    from refgrade.session import PREVIEW_SIDE
    scale = min(1.0, PREVIEW_SIDE / max(h, w))
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        source = F.interpolate(
            source_full.permute(0, 3, 1, 2), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).clamp(0.0, 1.0)
    else:
        source = source_full
    source = source.to(device)
    alpha = np.ones((h, w), dtype=np.uint8) * 255
    return Session(
        source_full=source_full,
        alpha=alpha,
        depth="U8",
        source=source,
        ignore=None,
    )


class TestPCHIP(unittest.TestCase):
    """PCHIP-Interpolation: Feste Sollwerte."""

    def test_value_at_control_point(self):
        """PCHIP auf [[0,0],[0.5,0.25],[1,1]]: Wert bei 0.5 = 0.25 (± 1e-6)."""
        curves = {"rgb": [[0, 0], [0.5, 0.25], [1, 1]]}
        x = torch.full((1, 1, 1, 3), 0.5, dtype=torch.float32)
        result = curves_apply(x, curves)
        self.assertAlmostEqual(result[0, 0, 0, 0].item(), 0.25, places=6)

    def test_monotone(self):
        """Kurve ist auf 1001 Stützstellen monoton steigend."""
        curves = {"rgb": [[0, 0], [0.5, 0.25], [1, 1]]}
        x = torch.linspace(0.0, 1.0, 1001, dtype=torch.float32).view(1, 1, 1001, 1).expand(1, 1, 1001, 3)
        result = curves_apply(x, curves)
        vals = result[0, 0, :, 0]
        diffs = vals[1:] - vals[:-1]
        self.assertGreaterEqual(diffs.min().item(), -1e-6, "Kurve nicht monoton steigend")

    def test_no_overshoot(self):
        """Kein Wert außerhalb [0,1]."""
        curves = {"rgb": [[0, 0], [0.5, 0.25], [1, 1]]}
        x = torch.linspace(0.0, 1.0, 1001, dtype=torch.float32).view(1, 1, 1001, 1).expand(1, 1, 1001, 3)
        result = curves_apply(x, curves)
        self.assertGreaterEqual(result.min().item(), 0.0)
        self.assertLessEqual(result.max().item(), 1.0)


class TestCurvesIdentity(unittest.TestCase):
    """Identität: apply(x, {}) und apply(x, {"rgb": [[0,0],[1,1]]}) == x."""

    def test_empty_curves(self):
        """apply(x, {}) == x (max. 1e-6)."""
        x = torch.rand(1, 8, 8, 3, dtype=torch.float32)
        result = curves_apply(x, {})
        diff = (result - x).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_identity_curve(self):
        """apply(x, {"rgb": [[0,0],[1,1]]}) == x (max. 1e-6)."""
        x = torch.rand(1, 8, 8, 3, dtype=torch.float32)
        result = curves_apply(x, {"rgb": [[0, 0], [1, 1]]})
        diff = (result - x).abs().max().item()
        self.assertLess(diff, 1e-6)


class TestCurvesChannel(unittest.TestCase):
    """Kanal: {"r": [[0,0],[1,0.5]]} auf Weiß → (0.5, 1, 1) ± 1e-4."""

    def test_red_channel(self):
        white = torch.full((1, 1, 1, 3), 1.0, dtype=torch.float32)
        result = curves_apply(white, {"r": [[0, 0], [1, 0.5]]})
        self.assertAlmostEqual(result[0, 0, 0, 0].item(), 0.5, places=4)
        self.assertAlmostEqual(result[0, 0, 0, 1].item(), 1.0, places=4)
        self.assertAlmostEqual(result[0, 0, 0, 2].item(), 1.0, places=4)


class TestCurvesOrder(unittest.TestCase):
    """Reihenfolge: rgb zuerst, dann Kanal-Kurven."""

    def test_rgb_then_channel(self):
        """{"rgb": [[0,0],[1,0.5]], "g": [[0,0],[0.5,1],[1,1]]} auf Weiß → G=1.0, R=B=0.5."""
        white = torch.full((1, 1, 1, 3), 1.0, dtype=torch.float32)
        curves = {"rgb": [[0, 0], [1, 0.5]], "g": [[0, 0], [0.5, 1], [1, 1]]}
        result = curves_apply(white, curves)
        # rgb(1.0) = 0.5 für alle Kanäle
        # dann g(0.5) = 1.0
        self.assertAlmostEqual(result[0, 0, 0, 0].item(), 0.5, places=4)  # R
        self.assertAlmostEqual(result[0, 0, 0, 1].item(), 1.0, places=4)  # G
        self.assertAlmostEqual(result[0, 0, 0, 2].item(), 0.5, places=4)  # B


class TestCurvesMix(unittest.TestCase):
    """Mix: curves w=0.5, rgb=[[0,0],[1,0.5]] auf Grau 0.8 → 0.6 (± 3/255)."""

    def test_mix_weight_half(self):
        h, w = 32, 32
        source = torch.full((1, h, w, 3), 0.8, dtype=torch.float32)
        s = _make_session(source, device=torch.device("cpu"))
        vals = values_with_defaults({})
        mix = {"weights": {"curves": 0.5}, "master": 1.0, "solo": None,
               "curves": {"rgb": [[0, 0], [1, 0.5]]}}
        result, _ = preview(s, [_REF], "reinhard", vals, 1536, mix)
        # Erwartet: 0.8 + 0.5·(0.4 − 0.8) = 0.6
        expected = 0.8 + 0.5 * (0.4 - 0.8)  # = 0.6
        diff = abs(result[0, 0, 0, 0].item() - expected)
        self.assertLessEqual(diff, 3.0 / 255.0,
                             f"Ergebnis {result[0,0,0,0].item():.4f} ≠ 0.6 (±3/255)")


class TestCurvesValidate(unittest.TestCase):
    """validate: ValueError bei Formatverletzung."""

    def test_x_not_increasing(self):
        with self.assertRaises(ValueError):
            curves_validate({"rgb": [[0.5, 0], [0.2, 1]]})

    def test_one_point(self):
        with self.assertRaises(ValueError):
            curves_validate({"rgb": [[0.5, 0.5]]})

    def test_seventeen_points(self):
        pts = [[i / 16, i / 16] for i in range(17)]
        with self.assertRaises(ValueError):
            curves_validate({"rgb": pts})

    def test_y_out_of_range(self):
        with self.assertRaises(ValueError):
            curves_validate({"rgb": [[0, 0], [1, 1.2]]})

    def test_valid_curves_no_error(self):
        curves_validate({"rgb": [[0, 0], [0.5, 0.25], [1, 1]]})
        curves_validate({"r": [[0, 0], [1, 0.5]], "g": [[0, 0], [1, 1]]})
        curves_validate({})


class TestCurvesIsIdentity(unittest.TestCase):
    """is_identity: True für leere/Identitäts-Kurven."""

    def test_none(self):
        self.assertTrue(curves_is_identity(None))

    def test_empty(self):
        self.assertTrue(curves_is_identity({}))

    def test_identity(self):
        self.assertTrue(curves_is_identity({"rgb": [[0, 0], [1, 1]]}))

    def test_non_identity(self):
        self.assertFalse(curves_is_identity({"rgb": [[0, 0], [1, 0.5]]}))

    def test_partial_identity(self):
        self.assertFalse(curves_is_identity({"rgb": [[0, 0], [1, 1]], "r": [[0, 0], [1, 0.8]]}))


class TestSectionsK9a(unittest.TestCase):
    """Sections: VISIBLE=47, FITTABLE=22, hsl-Namen."""

    def test_visible_count(self):
        self.assertEqual(len(VISIBLE), 95)

    def test_fittable_count(self):
        self.assertEqual(len(FITTABLE), 22)

    def test_hsl_names_match_sliders(self):
        """SECTIONS["hsl"] enthält genau die 18 sel_* aus SLIDERS."""
        sel_from_sliders = {s.name for s in SLIDERS if s.name.startswith("sel_")}
        sel_from_sections = set(SECTIONS["hsl"])
        self.assertEqual(sel_from_sections, sel_from_sliders)
        self.assertEqual(len(sel_from_sections), 18)

    def test_vibrance_in_color(self):
        self.assertIn("vibrance", SECTIONS["color"])

    def test_curves_in_sections(self):
        self.assertIn("curves", SECTIONS)
        self.assertEqual(SECTIONS["curves"], ())


class TestSessionFitK9a(unittest.TestCase):
    """session.fit liefert für jeden sel_*-Regler den Standardwert."""

    @slow

    def test_fit_returns_defaults_for_sel(self):
        from refgrade.session import fit as sess_fit
        torch.manual_seed(42)
        img = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        from refgrade import rawio
        rawio.save_png(img, str(ref_file))
        try:
            s = _make_session(img, device=torch.device("cpu"))
            result = sess_fit(s, [str(ref_file)], "reinhard")
            for sl in SLIDERS:
                if sl.name.startswith("sel_"):
                    self.assertEqual(result[sl.name], sl.default,
                                     f"{sl.name} ≠ Standard {sl.default}")
        finally:
            if ref_file.exists():
                os.unlink(ref_file)


class TestCurvesIsIdentitySession(unittest.TestCase):
    """is_identity mit curves-Bereich."""

    def test_curves_no_data_is_identity(self):
        """curves w=1 ohne Daten → Identität."""
        vals = values_with_defaults({})
        mix = {"weights": {"curves": 1.0}, "master": 1.0, "solo": None}
        self.assertTrue(is_identity(vals, mix))

    def test_curves_identity_data_is_identity(self):
        """curves w=1 mit Identitätskurven → Identität."""
        vals = values_with_defaults({})
        mix = {"weights": {"curves": 1.0}, "master": 1.0, "solo": None,
               "curves": {"rgb": [[0, 0], [1, 1]]}}
        self.assertTrue(is_identity(vals, mix))

    def test_curves_non_identity_not_identity(self):
        """curves w=1 mit nicht-identischen Kurven → keine Identität."""
        vals = values_with_defaults({})
        mix = {"weights": {"curves": 1.0}, "master": 1.0, "solo": None,
               "curves": {"rgb": [[0, 0], [1, 0.5]]}}
        self.assertFalse(is_identity(vals, mix))


if __name__ == "__main__":
    unittest.main()
