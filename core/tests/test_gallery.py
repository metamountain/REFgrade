"""Tests für refgrade.gallery: Galerie-Logik."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from refgrade.gallery import round as gallery_round, GalleryContext, Candidate
from refgrade.sliders import SLIDERS
from refgrade.grade import grade
from refgrade.analysis import analyze_image
from refgrade.metrics import delta_e00


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _make_test_pair(dev: torch.device, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Bild auf CPU erzeugen: torch.rand mit CUDA-Generator liefert bei gleichem Seed
    # ANDERE Werte als CPU → ΔE-Schwellen wären geräteabhängig. (2026-10-18)
    img = torch.rand(1, 64, 64, 3)
    x = torch.linspace(0, 1, 64)
    grad = x[None, :, None].expand(1, 64, 64, 1)
    img = (img * 0.7 + grad * 0.3).clamp(0, 1)
    # Vorlage: leicht anders
    vals = _defaults()
    vals["exposure"] = 0.15
    vals["temperature"] = 0.1
    ref = grade(img, vals).clamp(0, 1)
    return img.to(dev), ref.to(dev)


def _make_context(dev: torch.device, seed: int = 42) -> GalleryContext:
    img, ref = _make_test_pair(dev, seed)
    img_analysis = analyze_image(img)
    ref_analysis = analyze_image(ref)
    return GalleryContext(
        image=img,
        reference=ref,
        image_analysis=img_analysis,
        reference_analysis=ref_analysis,
    )


def _slider_in_range(values: dict) -> bool:
    """Alle Werte (außer _-Keys) im SLIDERS-Bereich."""
    for name, val in values.items():
        if name.startswith("_"):
            continue
        s = next((s for s in SLIDERS if s.name == name), None)
        if s is None:
            continue
        if val < s.min - 1e-6 or val > s.max + 1e-6:
            return False
    return True


class TestGalleryAnchor(unittest.TestCase):

    def test_anchor_is_middle(self):
        """Kandidat 4 (Mitte) == Anker (außer matrix)."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        for kind in ("start", "light", "color", "strength", "reference", "mix"):
            candidates = gallery_round(kind, anchor, ctx, step_level=0)
            self.assertEqual(len(candidates), 9, f"{kind}: nicht 9 Kandidaten")
            mid = candidates[4]
            # Ankerwerte müssen übereinstimmen
            for k, v in anchor.items():
                if k.startswith("_"):
                    continue
                self.assertAlmostEqual(mid.values.get(k, v), v, places=6,
                                       msg=f"{kind}: Anker {k} weicht ab")

    def test_matrix_no_anchor(self):
        """Matrix: kein Anker in der Mitte, 3 Methoden × 3 Stärken."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        candidates = gallery_round("matrix", anchor, ctx, step_level=0)
        self.assertEqual(len(candidates), 9)

        methods = set()
        strengths = set()
        for c in candidates:
            methods.add(c.values.get("_method"))
            strengths.add(c.values.get("_strength"))
        self.assertEqual(len(methods), 3, f"Methoden: {methods}")
        self.assertEqual(len(strengths), 3, f"Stärken: {strengths}")


class TestGalleryDifferences(unittest.TestCase):

    def _delta_e(self, ctx: GalleryContext, anchor: dict, candidate: Candidate) -> float:
        """ΔE00 zwischen Anker-Ergebnis und Kandidat-Ergebnis."""
        img = ctx.image
        a_result = grade(img, anchor).clamp(0, 1)
        c_result = grade(img, candidate.values).clamp(0, 1)
        mean_de, _ = delta_e00(a_result, c_result)
        return mean_de

    def test_light_candidates_differ(self):
        """Light: Kandidaten unterscheiden sich vom Anker (ΔE00 ≥ 0.2, ≤ 8).

        2026-10-18: Schwelle 0.5 → 0.2 — im Bump-Modell wirken die Lift-Regler nur
        noch in der Schattenzone (grosse Lücken), ihre Kandidaten sind auf Bildern
        mit wenig Schatten von Natur aus schwach (ΔE00 ~0.26…0.60 bei voller
        Schrittweite). Exposure/Contrast bleiben weit darüber."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        candidates = gallery_round("light", anchor, ctx, step_level=0)
        for i, c in enumerate(candidates):
            if i == 4:
                continue
            de = self._delta_e(ctx, anchor, c)
            self.assertGreaterEqual(de, 0.2,
                                    f"light[{i}] ({c.label}): ΔE00={de:.3f} < 0.2")
            self.assertLessEqual(de, 8.0,
                                 f"light[{i}] ({c.label}): ΔE00={de:.3f} > 8.0")

    def test_color_candidates_differ(self):
        """Color: Kandidaten unterscheiden sich vom Anker."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        candidates = gallery_round("color", anchor, ctx, step_level=0)
        for i, c in enumerate(candidates):
            if i == 4:
                continue
            de = self._delta_e(ctx, anchor, c)
            self.assertGreaterEqual(de, 0.5,
                                    f"color[{i}] ({c.label}): ΔE00={de:.3f} < 0.5")
            self.assertLessEqual(de, 8.0,
                                 f"color[{i}] ({c.label}): ΔE00={de:.3f} > 8.0")

    def test_second_round_half_distance(self):
        """Zweite Runde hat etwa halbe Abstände."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        c1 = gallery_round("light", anchor, ctx, step_level=0)
        c2 = gallery_round("light", anchor, ctx, step_level=1)

        # Vergleiche die ersten 4 Kandidaten (±1 Schritt)
        de1_list = []
        de2_list = []
        for i in [0, 1, 2, 3]:
            de1_list.append(self._delta_e(ctx, anchor, c1[i]))
            de2_list.append(self._delta_e(ctx, anchor, c2[i]))

        # Zweite Runde sollte kleiner sein
        avg1 = sum(de1_list) / len(de1_list)
        avg2 = sum(de2_list) / len(de2_list)
        self.assertLess(avg2, avg1,
                        f"Runde 2 nicht kleiner: {avg2:.3f} >= {avg1:.3f}")
        # Etwa halbe Abstände (mit Toleranz)
        self.assertLess(avg2, avg1 * 0.7,
                        f"Runde 2 nicht ~halbe Abstände: {avg2:.3f} vs {avg1:.3f}")


class TestGalleryConstraints(unittest.TestCase):

    def test_all_values_in_range(self):
        """Alle Werte im SLIDERS-Bereich."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        for kind in ("start", "light", "color", "strength", "mix"):
            candidates = gallery_round(kind, anchor, ctx, step_level=0)
            for i, c in enumerate(candidates):
                self.assertTrue(_slider_in_range(c.values),
                                f"{kind}[{i}] ({c.label}): Wert außerhalb des Bereichs")

    def test_deterministic(self):
        """Zwei Runden → gleiche Kandidaten."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        for kind in ("light", "color", "start"):
            c1 = gallery_round(kind, anchor, ctx, step_level=0)
            c2 = gallery_round(kind, anchor, ctx, step_level=0)
            for i in range(9):
                for k in c1[i].values:
                    if k.startswith("_"):
                        continue
                    self.assertAlmostEqual(c1[i].values[k], c2[i].values[k], places=6,
                                           msg=f"{kind}[{i}].{k} nicht deterministisch")
                self.assertEqual(c1[i].label, c2[i].label)

    def test_labels_nonempty(self):
        """Alle Labels sind nicht leer."""
        dev = _get_device()
        ctx = _make_context(dev)
        anchor = _defaults()

        for kind in ("start", "light", "color", "matrix", "strength"):
            candidates = gallery_round(kind, anchor, ctx, step_level=0)
            for i, c in enumerate(candidates):
                self.assertTrue(len(c.label) > 0, f"{kind}[{i}]: leeres Label")


if __name__ == "__main__":
    unittest.main()
