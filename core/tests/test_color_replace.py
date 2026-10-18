"""Tests für Color Replace: HSL-Distanz-Maske, mehrere Instanzen."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path

import torch

from refgrade.color_replace import apply, MAX_INSTANCES
from refgrade.sliders import SLIDERS
from slow import slow  # noqa: E402

# ── Qt widget setup (for Bug-fix tests 1-3) ──────────────────────────────────
os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules.setdefault("ref_grade", _pkg)

from PyQt6.QtWidgets import QApplication  # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)

from ref_grade.dlg_color_replace import ColorReplaceSection  # noqa: E402
from ref_grade import style  # noqa: E402


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

    @slow
    def test_default_is_identity(self):
        """Alle Werte auf Default → Bild unverändert."""
        x = _red_image()
        y = apply(x, _defaults())
        torch.testing.assert_close(y, x)

    def test_replace_red_to_blue(self):
        """Rot → Blau: rotes Bild wird blau."""
        x = _red_image()
        vals = _defaults()
        # V=1 images, as if picked: source and target bars at V=1 (defaults 0.5 since 2026-10-17)
        for _k in [k for k in vals if k.endswith(("_src_bright", "_tgt_bright"))]:
            vals[_k] = 1.0
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
        # V=1 images, as if picked: source and target bars at V=1 (defaults 0.5 since 2026-10-17)
        for _k in [k for k in vals if k.endswith(("_src_bright", "_tgt_bright"))]:
            vals[_k] = 1.0
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
        # V=1 images, as if picked: source and target bars at V=1 (defaults 0.5 since 2026-10-17)
        for _k in [k for k in vals if k.endswith(("_src_bright", "_tgt_bright"))]:
            vals[_k] = 1.0
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

    def test_bright_ratio_target_over_source(self):
        """2026-10-17: output V scales with tgt/src; defaults 0.5/0.5 keep V."""
        x = torch.tensor([[[[0.0, 0.0, 0.6]] * 4] * 4])        # blue, V=0.6
        vals = _defaults()
        self.assertEqual((vals["cr_0_src_bright"], vals["cr_0_tgt_bright"]), (0.5, 0.5))
        vals.update({"cr_0_src_hue": 240.0, "cr_0_src_sat": 1.0, "cr_0_tgt_hue": 0.0,
                     "cr_0_tgt_sat": 1.0, "cr_0_intensity": 1.0, "cr_0_range": 90.0})
        y = apply(x, vals)
        self.assertGreater(y[0, 0, 0, 0].item(), 0.5)           # red, V kept (88 % mix: V 0.1 off the 0.5 gate)
        vals.update({"cr_0_src_bright": 0.6, "cr_0_tgt_bright": 0.3})
        y = apply(x, vals)
        self.assertAlmostEqual(y[0, 0, 0, 0].item(), 0.3, delta=0.01)   # half as bright

    def test_src_bright_weak_modulation(self):
        """src_bright modulated nur schwach (0.7..1.0): Midtone voll, Highlight 70%."""
        vals = _defaults()
        vals["cr_0_src_hue"] = 240.0
        vals["cr_0_src_sat"] = 1.0
        vals["cr_0_src_bright"] = 0.5
        # 2026-10-17: output brightness ∝ tgt/src → tgt 0.5 keeps V (was 1.0 with V × tgt)
        vals["cr_0_tgt_hue"] = 0.0
        vals["cr_0_tgt_sat"] = 1.0
        vals["cr_0_tgt_bright"] = 0.5
        vals["cr_0_intensity"] = 1.0
        vals["cr_0_range"] = 90.0
        # Midtone-Blau (V=0.5) → bright_mod=1.0 → voll ersetzt (rot mit V=0.5)
        x_mid = torch.tensor([[[[0.0, 0.0, 0.5]] * 4] * 4])
        y_mid = apply(x_mid, vals)
        self.assertAlmostEqual(y_mid[0, 0, 0, 0].item(), 0.5, delta=0.01)
        self.assertLess(y_mid[0, 0, 0, 2].item(), 0.01)
        # Highlight-Blau (V=1.0) → bright_mod=0.7 → 70% ersetzt (nicht mehr unverändert)
        x_hi = torch.tensor([[[[0.0, 0.0, 1.0]] * 4] * 4])
        y_hi = apply(x_hi, vals)
        # 70%-Mix Blau→Rot: R≈0.7, B≈0.3
        self.assertAlmostEqual(y_hi[0, 0, 0, 0].item(), 0.7, delta=0.01)
        self.assertAlmostEqual(y_hi[0, 0, 0, 2].item(), 0.3, delta=0.01)

    def test_tgt_bright_scales_output_brightness(self):
        """tgt_bright skaliert OUTPUT-Helligkeit: 0.5=dunkler, 1.0=normal, 1.5=geclampt."""
        x = torch.tensor([[[[0.0, 0.0, 1.0]] * 4] * 4])  # reines Blau, V=1

        def _red(tb: float) -> float:
            vals = _defaults()
            vals["cr_0_src_hue"] = 240.0
            vals["cr_0_src_sat"] = 1.0
            vals["cr_0_src_bright"] = 1.0
            vals["cr_0_tgt_hue"] = 0.0
            vals["cr_0_tgt_sat"] = 1.0
            vals["cr_0_tgt_bright"] = tb
            vals["cr_0_intensity"] = 1.0
            vals["cr_0_range"] = 90.0
            return apply(x, vals)[0, 0, 0, 0].item()

        self.assertAlmostEqual(_red(0.5), 0.5, delta=0.01)
        self.assertAlmostEqual(_red(1.0), 1.0, delta=0.01)
        self.assertAlmostEqual(_red(1.5), 1.0, delta=0.01)  # V_tgt=1.5 → geclampt
        self.assertLess(_red(0.5), _red(1.0))

    def test_max_instances(self):
        """MAX_INSTANCES = 8."""
        self.assertEqual(MAX_INSTANCES, 8)

    def test_sat_floor_default_protects_grey(self):
        """2026-10-18: Default sat_floor (0.15) keeps the old guard — grey (S=0) untouched."""
        x = _grey_image()
        vals = _defaults()
        self.assertEqual(vals["cr_0_sat_floor"], 0.15)
        vals.update({"cr_0_src_hue": 240.0, "cr_0_src_sat": 1.0, "cr_0_tgt_hue": 0.0,
                     "cr_0_tgt_sat": 1.0, "cr_0_intensity": 1.0, "cr_0_range": 180.0})
        y = apply(x, vals)
        torch.testing.assert_close(y, x)

    def test_sat_floor_open_affects_desaturated(self):
        """2026-10-18: sat_floor=0 lifts the guard, so a desaturated pixel is hit harder
        than at the default (0.15)."""
        # Pale blue, S≈0.08 (desaturated), hue≈210° (inside ±90° of the 240° source).
        x = torch.tensor([[[[0.92, 0.96, 1.0]] * 4] * 4])
        base = {"cr_0_src_hue": 240.0, "cr_0_src_sat": 1.0, "cr_0_tgt_hue": 0.0,
                "cr_0_tgt_sat": 1.0, "cr_0_intensity": 1.0, "cr_0_range": 90.0,
                "cr_0_src_bright": 1.0, "cr_0_tgt_bright": 1.0}

        def _diff(sat_floor: float) -> float:
            vals = _defaults()
            vals.update(base)
            vals["cr_0_sat_floor"] = sat_floor
            return (apply(x, vals) - x).abs().max().item()

        self.assertGreater(_diff(0.0), _diff(0.15),
                           "opening the guard (0.0) should affect the desaturated pixel more")

    def test_low_src_sat_full_effect(self):
        """2026-10-18: src_sat scales the mask no more — picking a desaturated source
        (e.g. a black aircraft, S≈0.32) replaces at full strength (was 32 % before)."""
        x = torch.tensor([[[[0.17, 0.17, 0.25]] * 4] * 4])  # blue-grey, hue 240°, S=0.32, V=0.25
        vals = _defaults()
        vals.update({"cr_0_src_hue": 240.0, "cr_0_src_sat": 0.32, "cr_0_src_bright": 0.25,
                     "cr_0_tgt_hue": 0.0, "cr_0_tgt_sat": 1.0, "cr_0_tgt_bright": 1.0,
                     "cr_0_intensity": 1.0, "cr_0_range": 30.0})
        y = apply(x, vals)
        # full replacement → red at V_tgt = 0.25 × (1.0/0.25) = 1.0
        self.assertGreater(y[0, 0, 0, 0].item(), 0.9)
        self.assertLess(y[0, 0, 0, 2].item(), 0.1)


class TestColorReplaceWidget(unittest.TestCase):
    """PyQt6-Widget-Tests: die drei Wheels-Fixes (Removal, Size-Decision, Resize)."""

    def test_close_removes_instance(self):
        """Fix 1: ×-Button entfernt die Instanz wirklich aus self._instances."""
        section = ColorReplaceSection()
        section._on_add()  # 2 Instanzen
        self.assertEqual(len(section._instances), 2)
        inst = section._instances[1]
        inst._on_close()
        self.assertEqual(len(section._instances), 1)
        self.assertNotIn(inst, section._instances)

    def test_close_first_instance_noop(self):
        """Fix 1: Instanz 0 kann nicht entfernt werden (kein ×-Button)."""
        section = ColorReplaceSection()
        section._on_add()
        section._remove_instance(0)  # no-op
        self.assertEqual(len(section._instances), 2)

    def test_remove_reindexes_remaining(self):
        """Fix 1: Nach Removal werden verbleibende Instanzen neu indiziert."""
        section = ColorReplaceSection()
        section._on_add()
        section._on_add()  # 3 Instanzen: 0, 1, 2
        section._remove_instance(1)  # entferne mittlere
        self.assertEqual(len(section._instances), 2)
        self.assertEqual(section._instances[0].index, 0)
        self.assertEqual(section._instances[1].index, 1)

    def test_size_decision_visible_only(self):
        """Fix 2: Nur sichtbare Instanzen zählen für die Größenentscheidung."""
        section = ColorReplaceSection()
        s = style.current_s()
        natural = round(0.85 * s * 0.7)
        compact = min(100, natural)
        self.assertGreater(natural, compact, "Test-Prämisse: natural > compact")

        # 1 Instanz → natural
        self.assertEqual(section._instances[0].src_wheel.width(), natural)

        # 3 Instanzen → compact
        section._on_add()
        section._on_add()
        self.assertEqual(section._instances[0].src_wheel.width(), compact)

        # 1 entfernen → 2 sichtbar → zurück zu natural
        section._remove_instance(2)
        self.assertEqual(len(section._instances), 2)
        self.assertEqual(section._instances[0].src_wheel.width(), natural)

    def test_resize_updates_wheel_sizes(self):
        """Fix 3: _update_wheel_sizes reagiert auf Scale-Change (was resizeEvent triggert)."""
        section = ColorReplaceSection()
        section._on_add()
        section._on_add()  # 3 Instanzen → compact
        old_s = style.current_s()
        try:
            new_s = old_s * 1.5
            style.set_scale(new_s)
            section._update_wheel_sizes()
            new_natural = round(0.85 * new_s * 0.7)
            new_compact = min(100, new_natural)
            self.assertEqual(section._instances[0].src_wheel.width(), new_compact)
        finally:
            style.set_scale(old_s)

    def test_resize_grows_wheels(self):
        """Fix 3: Vergrößerung des Fensters vergrößert die Wheels."""
        section = ColorReplaceSection()
        old_s = style.current_s()
        try:
            large_s = old_s * 2.0
            style.set_scale(large_s)
            section._update_wheel_sizes()
            large_natural = round(0.85 * large_s * 0.7)
            self.assertEqual(section._instances[0].src_wheel.width(), large_natural)
        finally:
            style.set_scale(old_s)


if __name__ == "__main__":
    unittest.main()
