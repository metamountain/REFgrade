"""Tests für refgrade.grade: Identität, Einzelregler, Gradienten."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.grade import grade
from refgrade.sliders import SLIDERS


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


class TestGradeIdentity(unittest.TestCase):
    """Alle Regler auf Standard → bitgleich."""

    def test_all_defaults_identity(self):
        dev = _get_device()
        torch.manual_seed(42)
        img = torch.rand(2, 32, 48, 3, device=dev)
        vals = _defaults()
        out = grade(img, vals)
        self.assertTrue(torch.equal(out, img), "Alle Standards: nicht bitgleich")

    def test_empty_dict_identity(self):
        dev = _get_device()
        torch.manual_seed(42)
        img = torch.rand(1, 16, 16, 3, device=dev)
        out = grade(img, {})
        self.assertTrue(torch.equal(out, img), "Leeres dict: nicht bitgleich")


class TestGradeIndividualSlider(unittest.TestCase):
    """Jeder Regler vom Standard weg → Bild ändert sich."""

    # Regler, die nicht von grade() verwendet werden
    SKIP = {s.name for s in SLIDERS if s.group in ("match", "rgb")} | {"clarity", "texture", "dehaze", "highlight_reconstruct"} \
        | {f"cr_{i}_src_sat" for i in range(8)}   # 2026-10-18: src_sat skaliert die Maske nicht mehr (reine Quellfarb-Beschreibung)
    # Regler, die nur in Kombination wirken: name → zusätzliche Werte
    COMBO = {
        "pivot": {"contrast": 1.5},
        "shadow_hue": {"shadow_amount": 0.5},
        "highlight_hue": {"highlight_amount": 0.5},
        "wheel_gain": {"gain_r": 1.5},
        "wheel_falloff": {"gain_r": 1.5},
        "wheel_intensity": {"gain_r": 1.5},   # scales the wheel tint: no tint, no effect
    }
    # cr_*-Regler brauchen intensity > 0, um etwas zu tun
    for _i in range(8):
        for _k in ("src_hue", "src_bright", "tgt_hue", "tgt_sat", "tgt_bright", "range", "sat_floor"):
            COMBO[f"cr_{_i}_{_k}"] = {f"cr_{_i}_intensity": 0.5}

    def _test_single_slider(self, name: str, offset: float):
        dev = _get_device()
        torch.manual_seed(7)
        img = torch.rand(1, 24, 24, 3, device=dev)
        vals = _defaults()
        s = next(s for s in SLIDERS if s.name == name)
        new_val = s.default + offset
        new_val = max(s.min, min(s.max, new_val))
        if new_val == s.default:
            new_val = s.default + 0.01
        vals[name] = new_val
        # Kombi-Regler: zusätzliche Werte setzen
        if name in self.COMBO:
            vals.update(self.COMBO[name])
        out = grade(img, vals)
        changed = (out - img).abs().max().item()
        self.assertGreater(changed, 1e-6, f"Regler {name}: Bild nicht verändert (Δ={changed})")

    def test_all_sliders(self):
        for s in SLIDERS:
            if s.name in self.SKIP:
                continue
            with self.subTest(slider=s.name):
                if s.default == s.min:
                    offset = 0.1 * (s.max - s.min)
                elif s.default == s.max:
                    offset = -0.1 * (s.max - s.min)
                else:
                    offset = 0.1 * (s.max - s.min)
                self._test_single_slider(s.name, offset)


class TestGradeGradient(unittest.TestCase):
    """Gradient nach jedem Regler existiert und ist endlich."""

    # Regler, die nicht von grade() verwendet werden
    SKIP = {s.name for s in SLIDERS if s.group in ("match", "rgb")} | {"clarity", "texture", "dehaze", "highlight_reconstruct"} \
        | {f"cr_{i}_src_sat" for i in range(8)}   # 2026-10-18: src_sat skaliert die Maske nicht mehr (kein Gradient)

    def test_gradients(self):
        dev = _get_device()
        torch.manual_seed(99)
        img = torch.rand(1, 16, 16, 3, device=dev, requires_grad=False)

        # Alle Regler als Tensoren mit requires_grad
        vals: dict[str, torch.Tensor] = {}
        for s in SLIDERS:
            if s.name in self.SKIP:
                continue
            if s.default == s.min:
                v = s.default + 0.1 * (s.max - s.min)
            elif s.default == s.max:
                v = s.default - 0.1 * (s.max - s.min)
            else:
                v = s.default + 0.1 * (s.max - s.min)
            v = max(s.min, min(s.max, v))
            vals[s.name] = torch.tensor(v, device=dev, requires_grad=True)

        out = grade(img, vals)
        loss = out.sum()
        loss.backward()

        for s in SLIDERS:
            if s.name in self.SKIP:
                continue
            g = vals[s.name].grad
            with self.subTest(slider=s.name):
                self.assertIsNotNone(g, f"Gradient für {s.name} fehlt")
                self.assertTrue(torch.isfinite(g).all(),
                                f"Gradient für {s.name} nicht endlich: {g}")


class TestGradeNoFinalClamp(unittest.TestCase):
    """Kein finales clamp: Werte außerhalb [0,1] möglich."""

    def test_no_clamp(self):
        dev = _get_device()
        torch.manual_seed(1)
        img = torch.rand(1, 8, 8, 3, device=dev)
        vals = _defaults()
        vals["exposure"] = 3.0  # Stark aufhellen
        out = grade(img, vals)
        self.assertGreater(out.max().item(), 1.0, "Erwartet Werte > 1 (kein clamp)")


class TestGradeInputUnchanged(unittest.TestCase):
    """Eingabetensor unverändert."""

    def test_input_unchanged(self):
        dev = _get_device()
        torch.manual_seed(5)
        img = torch.rand(1, 16, 16, 3, device=dev)
        img_copy = img.clone()
        vals = _defaults()
        vals["exposure"] = 1.0
        vals["saturation"] = 1.5
        grade(img, vals)
        self.assertTrue(torch.equal(img, img_copy), "Eingabe wurde verändert")


class TestGradeFlatImages(unittest.TestCase):
    """Flach/schwarz/weiß → keine NaN/Inf."""

    def test_flat(self):
        dev = _get_device()
        vals = _defaults()
        vals["exposure"] = 1.0
        vals["temperature"] = 0.5
        vals["contrast"] = 1.5
        vals["saturation"] = 2.0
        vals["shadow_amount"] = 0.5
        vals["highlight_amount"] = 0.5
        vals["sel_red_hue"] = 15.0
        vals["sel_blue_sat"] = 0.5

        cases = [
            torch.zeros(1, 8, 8, 3, device=dev),
            torch.ones(1, 8, 8, 3, device=dev),
            torch.full((1, 8, 8, 3), 0.5, device=dev),
        ]
        for i, img in enumerate(cases):
            out = grade(img, vals)
            with self.subTest(case=i):
                self.assertTrue(torch.isfinite(out).all(),
                                f"Fall {i}: NaN/Inf in Ausgabe")


class TestPresets(unittest.TestCase):
    """PRESETS: jeder Schlüssel existiert in SLIDERS, jeder Wert im Bereich."""

    def test_all_keys_valid(self):
        from refgrade.presets import PRESETS
        slider_map = {s.name: s for s in SLIDERS}
        for name, vals in PRESETS.items():
            for key, val in vals.items():
                with self.subTest(preset=name, key=key):
                    self.assertIn(key, slider_map,
                                  f"Presets[{name!r}]: {key!r} nicht in SLIDERS")
                    s = slider_map[key]
                    self.assertGreaterEqual(val, s.min,
                                            f"Presets[{name!r}]: {key}={val} < min={s.min}")
                    self.assertLessEqual(val, s.max,
                                         f"Presets[{name!r}]: {key}={val} > max={s.max}")


if __name__ == "__main__":
    unittest.main()
