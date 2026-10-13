"""Tests für Paket K8: LUT-Bereich im additiven Mix (Kern)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from refgrade.lut import build as lut_build, write_cube as lut_write_cube
from refgrade.sections import SECTIONS, VISIBLE, FITTABLE
from refgrade.session import (
    Session,
    apply_full,
    is_identity,
    preview,
    values_with_defaults,
)

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


class TestSectionsTableLut(unittest.TestCase):
    """SECTIONS hat lut mit leerer Reglerliste; VISIBLE und FITTABLE unverändert."""

    def test_lut_in_sections(self):
        self.assertIn("lut", SECTIONS)
        self.assertEqual(SECTIONS["lut"], ())

    def test_visible_count(self):
        self.assertEqual(len(VISIBLE), 65)

    def test_fittable_count(self):
        self.assertEqual(len(FITTABLE), 22)


class TestLutSectionIdentity(unittest.TestCase):
    """Identitäts-LUT (w=1) → Vorschau gleich Quelle."""

    def test_identity_lut(self):
        """Identitäts-LUT mit w=1: Ergebnis = Quelle (max. 2/255)."""
        rng = np.random.default_rng(42)
        h, w = 64, 96
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        source = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

        # Identitäts-LUT bauen und schreiben
        identity_fn = lambda x: x
        identity_lut = lut_build(identity_fn, size=17, title="identity", device=torch.device("cpu"))
        tmp = tempfile.mkdtemp(prefix="refgrade_lut_test_")
        cube_path = Path(tmp) / "identity.cube"
        lut_write_cube(identity_lut, str(cube_path))

        try:
            s = _make_session(source, device=torch.device("cpu"))
            vals = values_with_defaults({})
            mix = {"weights": {"lut": 1.0}, "master": 1.0, "solo": None,
                   "lut": str(cube_path)}
            result, _ = preview(s, [_REF], "reinhard", vals, 1536, mix)
            diff = (result - source).abs().max().item()
            self.assertLessEqual(diff, 2.0 / 255.0, f"Abweichung {diff:.6f} > 2/255")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLutSectionInvert(unittest.TestCase):
    """Invertierungs-LUT (x→1−x) mit w=0.5 auf konstantes Grau 0.2 → 0.5."""

    def test_invert_lut_half(self):
        """x=0.2, LUT: 1−x=0.8, w=0.5 → 0.2 + 0.5·(0.8−0.2) = 0.5 (± 3/255)."""
        h, w = 32, 32
        source = torch.full((1, h, w, 3), 0.2, dtype=torch.float32)

        # Invertierungs-LUT: f(x) = 1 − x
        invert_fn = lambda x: 1.0 - x
        invert_lut = lut_build(invert_fn, size=17, title="invert", device=torch.device("cpu"))
        tmp = tempfile.mkdtemp(prefix="refgrade_lut_test_")
        cube_path = Path(tmp) / "invert.cube"
        lut_write_cube(invert_lut, str(cube_path))

        try:
            s = _make_session(source, device=torch.device("cpu"))
            vals = values_with_defaults({})
            mix = {"weights": {"lut": 0.5}, "master": 1.0, "solo": None,
                   "lut": str(cube_path)}
            result, _ = preview(s, [_REF], "reinhard", vals, 1536, mix)
            expected = 0.2 + 0.5 * (0.8 - 0.2)  # = 0.5
            diff = abs(result[0, 0, 0, 0].item() - expected)
            self.assertLessEqual(diff, 3.0 / 255.0, f"Ergebnis {result[0,0,0,0].item():.4f} ≠ 0.5 (±3/255)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLutSectionNoPath(unittest.TestCase):
    """Gewicht lut=1 ohne Pfad → Ergebnis gleich Mix ohne lut (bytegleich)."""

    def test_no_path_same_as_without(self):
        """apply_full mit lut w=1 ohne Pfad == apply_full ohne lut."""
        rng = np.random.default_rng(7)
        h, w = 32, 48
        rgba = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            from refgrade.session import open_session
            s = open_session(path, w, h, "U8", device=torch.device("cpu"))
            vals = values_with_defaults({"exposure": 0.3})

            # Mix ohne lut
            mix_no_lut = {"weights": {"tone": 1.0}, "master": 1.0, "solo": None}
            r1 = apply_full(s, [_REF], "reinhard", vals, mix_no_lut)

            # Mix mit lut w=1 aber ohne Pfad
            mix_with_lut = {"weights": {"tone": 1.0, "lut": 1.0}, "master": 1.0, "solo": None}
            r2 = apply_full(s, [_REF], "reinhard", vals, mix_with_lut)

            self.assertTrue(torch.equal(r1, r2),
                            "Mit lut w=1 ohne Pfad ≠ ohne lut (bytegleich erwartet)")
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestLutSectionIsIdentity(unittest.TestCase):
    """is_identity mit lut-Bereich."""

    def test_lut_no_path_is_identity(self):
        """lut w=1 ohne Pfad → Identität."""
        vals = values_with_defaults({})
        mix = {"weights": {"lut": 1.0}, "master": 1.0, "solo": None}
        self.assertTrue(is_identity(vals, mix))

    def test_lut_with_path_not_identity(self):
        """lut w=1 mit Pfad → keine Identität."""
        vals = values_with_defaults({})
        mix = {"weights": {"lut": 1.0}, "master": 1.0, "solo": None,
               "lut": "/fake/path.cube"}
        self.assertFalse(is_identity(vals, mix))


if __name__ == "__main__":
    unittest.main()
