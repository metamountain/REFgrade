"""Tests für refgrade.sections: SECTIONS, VISIBLE, chain_fn, Identität, Kette."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from refgrade import rawio
from refgrade.color import rgb_to_lab
from refgrade.grade import grade
from refgrade.sections import SECTIONS, VISIBLE, FITTABLE, section_values, chain_fn
from refgrade.session import (
    Session,
    apply_full,
    build_lut,
    is_identity,
    open_session,
    preview,
    values_with_defaults,
)
from refgrade.sliders import SLIDERS

_CORE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_DIR = _CORE_DIR.parent
_DIALOG_PATH = _PROJECT_DIR / "plugin" / "ref_grade" / "dlg_common.py"
_POOL_DIR = str(_PROJECT_DIR / "bench" / "pool")


def _tensor_from_array(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] → [1,H,W,3] float32 0..1."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


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


def _lab_mean(tensor: torch.Tensor) -> tuple[float, float, float]:
    """Mittelwerte L*, a*, b*."""
    lab = rgb_to_lab(tensor)
    return (
        lab[..., 0].mean().item(),
        lab[..., 1].mean().item(),
        lab[..., 2].mean().item(),
    )


def _dialog_visible_names() -> set[str]:
    """VISIBLE-Regler aus dialog.py per ast lesen (ohne Qt)."""
    source = _DIALOG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    col1_names: list[str] = []
    col2_names: list[str] = []
    wheels_names: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "COLUMN_1":
                # (("Reference", ("light_strength", ...)), ("Tone", (...)))
                for elt in node.value.elts:  # tuple of tuples
                    names = elt.elts[1]  # second element is the names tuple
                    for n in names.elts:
                        col1_names.append(n.value)
            elif target.id == "COLUMN_2":
                for elt in node.value.elts:
                    names = elt.elts[1]
                    for n in names.elts:
                        col2_names.append(n.value)
            elif target.id == "WHEELS":
                # (("lift", "Lift · shadows"), ("gamma", ...), ("gain", ...))
                for elt in node.value.elts:
                    kind = elt.elts[0].value
                    for c in "mrgb":
                        wheels_names.append(f"{kind}_{c}")

    return set(col1_names + col2_names + wheels_names)


class TestSectionsTable(unittest.TestCase):
    """SECTIONS und VISIBLE stimmen mit dialog.py überein."""

    def test_visible_matches_dialog(self):
        """Alle Dialog-Namen sind in VISIBLE; VISIBLE enthält zusätzlich hsl + vibrance (K9b)."""
        dialog_names = _dialog_visible_names()
        # Dialog-Namen ⊆ VISIBLE
        missing = dialog_names - set(VISIBLE)
        self.assertEqual(missing, set(), f"Dialog-Namen nicht in VISIBLE: {missing}")
        # VISIBLE = Dialog + hsl + vibrance + wheel_* + color_replace (curves/lut haben keine Regler)
        new_names = set(SECTIONS["hsl"]) | set(SECTIONS["color_replace"]) | {"vibrance", "wheel_blending", "wheel_balance", "wheel_intensity"}
        expected = dialog_names | new_names
        self.assertEqual(set(VISIBLE), expected,
                         f"SECTIONS ≠ dialog+neu: nur SECTIONS={set(VISIBLE) - expected}, "
                         f"nur dialog+neu={expected - set(VISIBLE)}")

    def test_visible_count(self):
        """VISIBLE hat genau 129 Einträge."""
        self.assertEqual(len(VISIBLE), 129)

    def test_fittable_count(self):
        """FITTABLE hat genau 22 Einträge (tone + color + wheels, ohne clarity)."""
        self.assertEqual(len(FITTABLE), 22)

    def test_fittable_excludes_reference(self):
        """Kein reference-Regler in FITTABLE."""
        for name in SECTIONS["reference"]:
            self.assertNotIn(name, FITTABLE)

    def test_section_values(self):
        """section_values: Bereichsregler aus values, Rest auf Standard."""
        vals = values_with_defaults({"exposure": 0.5, "light_strength": 0.3})
        v_tone = section_values(vals, "tone")
        self.assertEqual(v_tone["exposure"], 0.5)
        self.assertEqual(v_tone["contrast"], 1.0)  # Standard
        self.assertEqual(v_tone["highlights"], 0.0)  # Standard
        # reference-Regler sind nicht im tone-Bereich
        self.assertEqual(v_tone["light_strength"], 1.0)  # Standard

    def test_section_values_unknown_key(self):
        """Unbekannter Bereich → KeyError."""
        with self.assertRaises(KeyError):
            section_values({}, "bogus")


class TestMixFn(unittest.TestCase):
    """chain_fn: Kette (2026-10-12, vorher additiv), SOLO, Master."""

    def test_chain(self):
        """x = 0.4, tone exposure=+0.5, color saturation=1.5, beide w=1, master=1 → tone(color(x))."""
        x = torch.full((1, 4, 4, 3), 0.4, dtype=torch.float32)

        # tone: exposure=+0.5
        v_tone = {s.name: s.default for s in SLIDERS}
        v_tone["exposure"] = 0.5
        f_tone = lambda t: grade(t, v_tone)

        # color: saturation=1.5
        v_color = {s.name: s.default for s in SLIDERS}
        v_color["saturation"] = 1.5
        f_color = lambda t: grade(t, v_color)

        # reference: Identität (Stärken 0)
        f_ref = lambda t: t

        # wheels: Identität
        f_wheels = lambda t: t

        source_fns = {"reference": f_ref, "tone": f_tone, "color": f_color, "wheels": f_wheels}
        mix = {"weights": {"tone": 1.0, "color": 1.0}, "master": 1.0, "solo": None}
        fn = chain_fn(source_fns, mix)
        result = fn(x)

        # Erwartet: Kette in DEFAULT_ORDER (color vor tone)
        expected = f_tone(f_color(x).clamp(0.0, 1.0)).clamp(0.0, 1.0)

        diff = (result - expected).abs().max().item()
        self.assertLess(diff, 2e-3, f"Abweichung {diff:.6f} > 2e-3")

    def test_master_halves(self):
        """master=0.5 halbiert die Abweichung (solange nichts geklemmt wird)."""
        x = torch.full((1, 4, 4, 3), 0.4, dtype=torch.float32)

        v_tone = {s.name: s.default for s in SLIDERS}
        v_tone["exposure"] = 0.5
        f_tone = lambda t: grade(t, v_tone)
        f_identity = lambda t: t

        source_fns = {"tone": f_tone, "reference": f_identity,
                      "color": f_identity, "wheels": f_identity}

        mix_full = {"weights": {"tone": 1.0}, "master": 1.0, "solo": None}
        mix_half = {"weights": {"tone": 1.0}, "master": 0.5, "solo": None}

        r_full = chain_fn(source_fns, mix_full)(x)
        r_half = chain_fn(source_fns, mix_half)(x)

        dev_full = (r_full - x).abs()
        dev_half = (r_half - x).abs()
        # |R_half - x| ≈ 0.5 * |R_full - x|
        ratio = dev_half / dev_full.clamp_min(1e-8)
        self.assertLess((ratio - 0.5).abs().max().item(), 2e-3)

    def test_solo_equals_weight(self):
        """SOLO tone == {"weights": {"tone": 1}, "master": 1}."""
        x = torch.rand(1, 8, 8, 3, dtype=torch.float32)

        v_tone = {s.name: s.default for s in SLIDERS}
        v_tone["exposure"] = 0.3
        v_tone["contrast"] = 1.2
        f_tone = lambda t: grade(t, v_tone)
        f_identity = lambda t: t

        source_fns = {"tone": f_tone, "reference": f_identity,
                      "color": f_identity, "wheels": f_identity}

        r_solo = chain_fn(source_fns, {"weights": {}, "master": 1.0, "solo": "tone"})(x)
        r_weight = chain_fn(source_fns, {"weights": {"tone": 1.0}, "master": 1.0, "solo": None})(x)

        diff = (r_solo - r_weight).abs().max().item()
        self.assertLess(diff, 1e-6, f"SOLO ≠ Gewicht: {diff:.2e}")


class TestIdentityWithMix(unittest.TestCase):
    """Identität mit Mix-Param."""

    def test_master_zero(self):
        """master=0 → Identität, egal welche Regler."""
        vals = values_with_defaults({"exposure": 1.0, "light_strength": 1.0})
        mix = {"weights": {"tone": 1.0, "reference": 1.0}, "master": 0.0, "solo": None}
        self.assertTrue(is_identity(vals, mix))

    def test_all_weights_zero(self):
        """Alle Gewichte 0 → Identität."""
        vals = values_with_defaults({"exposure": 1.0})
        mix = {"weights": {"tone": 0.0, "color": 0.0, "wheels": 0.0, "reference": 0.0},
               "master": 1.0, "solo": None}
        self.assertTrue(is_identity(vals, mix))

    def test_reference_strengthened_not_identity(self):
        """reference mit light_strength=1 → keine Identität."""
        vals = values_with_defaults({"light_strength": 1.0, "color_strength": 1.0})
        mix = {"weights": {"reference": 1.0}, "master": 1.0, "solo": None}
        self.assertFalse(is_identity(vals, mix))

    def test_reference_zero_strength_identity(self):
        """reference mit Stärken 0 + tone auf Standard → Identität."""
        vals = values_with_defaults({"light_strength": 0.0, "color_strength": 0.0})
        mix = {"weights": {"reference": 1.0, "tone": 1.0}, "master": 1.0, "solo": None}
        self.assertTrue(is_identity(vals, mix))

    def test_none_mix_defaults(self):
        """mix=None → alle Gewichte 1, master 1 → wie bisher."""
        vals = values_with_defaults({"light_strength": 0, "color_strength": 0})
        self.assertTrue(is_identity(vals, None))
        vals2 = values_with_defaults({"light_strength": 1, "color_strength": 1})
        self.assertFalse(is_identity(vals2, None))


class TestIdentityByteEqualWithMix(unittest.TestCase):
    """Identität bytegleich mit Mix: U16-Quelle."""

    def test_u16_master_zero(self):
        """master=0 → apply_full + write_raw bytegleich zur U16-Quelle."""
        rng = np.random.default_rng(0)
        h, w = 32, 48
        rgba = rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            s = open_session(path, w, h, "U16", device=torch.device("cpu"))
            mix = {"weights": {"tone": 1.0, "reference": 1.0}, "master": 0.0, "solo": None}
            result = apply_full(s, [], "reinhard", {"exposure": 1.0}, mix)
            out_path = path + ".out"
            rawio.write_raw(out_path, result, s.alpha, "U16")
            with open(path, "rb") as f:
                original = f.read()
            with open(out_path, "rb") as f:
                written = f.read()
            self.assertEqual(original, written)
        finally:
            for p in (path, path + ".out"):
                if os.path.exists(p):
                    os.unlink(p)

    def test_u16_all_weights_zero(self):
        """Alle Gewichte 0 → apply_full + write_raw bytegleich."""
        rng = np.random.default_rng(1)
        h, w = 32, 48
        rgba = rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            s = open_session(path, w, h, "U16", device=torch.device("cpu"))
            mix = {"weights": {"tone": 0.0, "color": 0.0, "wheels": 0.0, "reference": 0.0},
                   "master": 1.0, "solo": None}
            result = apply_full(s, [], "reinhard", {"exposure": 1.0}, mix)
            out_path = path + ".out"
            rawio.write_raw(out_path, result, s.alpha, "U16")
            with open(path, "rb") as f:
                original = f.read()
            with open(out_path, "rb") as f:
                written = f.read()
            self.assertEqual(original, written)
        finally:
            for p in (path, path + ".out"):
                if os.path.exists(p):
                    os.unlink(p)


class TestReinhardReferenceOnly(unittest.TestCase):
    """Reinhard astronaut → coffee, nur reference w=1, übrige auf Standard."""

    @classmethod
    def setUpClass(cls):
        from skimage import data
        astronaut = data.astronaut()  # [512,512,3] uint8
        coffee = data.coffee()  # [300,400,3] uint8
        cls.source = _tensor_from_array(astronaut)
        cls.ref_file = Path(tempfile.mktemp(suffix=".png"))
        from PIL import Image
        Image.fromarray(coffee).save(str(cls.ref_file), format="PNG")

    @classmethod
    def tearDownClass(cls):
        if cls.ref_file.exists():
            os.unlink(cls.ref_file)

    def test_lab_means_reference_only(self):
        """Nur reference w=1: L*≤0.5, a*≤1.0, b*≤3.5 Abweichung."""
        s = _make_session(self.source, device=torch.device("cpu"))
        vals = values_with_defaults({"light_strength": 1, "color_strength": 1})
        mix = {"weights": {"reference": 1.0}, "master": 1.0, "solo": None}
        result, _ = preview(s, [str(self.ref_file)], "reinhard", vals, 1536, mix)
        ref_img = rawio.load_image(str(self.ref_file), 768)
        L_r, a_r, b_r = _lab_mean(ref_img)
        L_p, a_p, b_p = _lab_mean(result)
        self.assertLessEqual(abs(L_p - L_r), 0.5)
        self.assertLessEqual(abs(a_p - a_r), 1.0)
        self.assertLessEqual(abs(b_p - b_r), 3.5)


class TestFitSlidersNames(unittest.TestCase):
    """fit_sliders mit names-Parameter."""

    def test_fit_single_slider(self):
        """names=['exposure']: nur exposure ≠ 0, Rest auf Standard."""
        from refgrade.fit_sliders import fit_sliders
        torch.manual_seed(42)
        img = torch.rand(1, 32, 32, 3, dtype=torch.float32)
        # Vorlage mit bekannter Belichtung
        v = {s.name: s.default for s in SLIDERS}
        v["exposure"] = 0.4
        ref = grade(img, v).clamp(0, 1)

        fitted, _ = fit_sliders(img, ref, max_steps=100, names=["exposure"])
        # exposure sollte nicht auf Standard sein
        self.assertNotAlmostEqual(fitted["exposure"], 0.0, places=2)
        # Nur exposure in den Ergebnissen
        self.assertEqual(set(fitted.keys()), {"exposure"})


class TestSessionFit(unittest.TestCase):
    """session.fit liefert für Regler außerhalb der 18 den Standardwert."""

    def test_fit_returns_defaults_for_reference(self):
        """session.fit: reference-Regler auf Standard."""
        from refgrade.session import fit as sess_fit
        torch.manual_seed(42)
        img = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(img, str(ref_file))
        try:
            s = _make_session(img, device=torch.device("cpu"))
            result = sess_fit(s, [str(ref_file)], "reinhard")
            # reference-Regler müssen auf Standard stehen
            self.assertEqual(result["light_strength"], 1.0)
            self.assertEqual(result["color_strength"], 1.0)
            self.assertEqual(result["protect_shadows"], 0.0)
            self.assertEqual(result["protect_highlights"], 0.0)
            # Alle SLIDERS-Regler müssen vorhanden sein
            for sl in SLIDERS:
                self.assertIn(sl.name, result)
        finally:
            if ref_file.exists():
                os.unlink(ref_file)


if __name__ == "__main__":
    unittest.main()
