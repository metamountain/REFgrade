"""Tests für tools/shootout.py: Rastergröße, Blindtest, Scoring."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refgrade.methods import get as get_method
from refgrade.methods import classic  # noqa: F401
from refgrade.metrics import delta_e00, structure, banding
from refgrade.grade import grade
from refgrade.sliders import SLIDERS

# shootout-Module importieren
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from shootout import (
    _get_device,
    _tensor_to_bgr,
    _resize_cell,
    _add_label,
    make_case_grid,
    make_overall_grid,
    Case,
    Candidate,
    _random_slider_values,
    run_score,
    _SHOOTOUT_DIR,
    _MAX_COLS,
    _CELL_SIZE,
)


def _make_test_image(h: int = 64, w: int = 96, seed: int = 0) -> torch.Tensor:
    """Deterministisches Testbild [1,H,W,3]."""
    rng = np.random.default_rng(seed)
    arr = rng.random((1, h, w, 3), dtype=np.float32)
    return torch.from_numpy(arr)


def _make_slider_values(seed: int = 0) -> dict[str, float]:
    """Einfache Reglerwerte für Tests."""
    return _random_slider_values(seed)


class TestGridSize(unittest.TestCase):
    """Rastergröße und Spaltenzahl."""

    def setUp(self):
        self.device = _get_device()
        self.img = _make_test_image(64, 96, seed=0).to(self.device)
        self.ref = _make_test_image(64, 96, seed=1).to(self.device)
        self.truth = _make_test_image(64, 96, seed=2).to(self.device)

        # Ein simpler Kandidat (Reinhard)
        self.case = Case(
            name="test",
            image=self.img,
            reference=self.ref,
            truth=self.truth,
        )
        fit_fn = get_method("reinhard")
        fn = fit_fn(self.img, self.ref)
        self.candidates = [Candidate(label="reinhard", kind="method", value="reinhard", fn=fn)]

    def test_grid_columns(self):
        """Original + Vorlage + Wahrheit + 1 Kandidat = 4 Spalten."""
        grid, labels = make_case_grid(self.case, self.candidates, blind=False)
        # 4 Zellen, je _CELL_SIZE breit (lange Seite)
        expected_w = 4 * _CELL_SIZE
        self.assertEqual(grid.shape[1], expected_w)
        self.assertEqual(len(labels), 4)

    def test_grid_max_columns(self):
        """Max 8 Spalten: mehr Kandidaten werden abgeschnitten."""
        # 7 Kandidaten + 3 fixe = 10 → abgeschnitten auf 8
        cands = []
        for i in range(7):
            fn = get_method("reinhard")(self.img, self.ref)
            cands.append(Candidate(label=f"c{i}", kind="method", value=f"c{i}", fn=fn))
        grid, labels = make_case_grid(self.case, cands, blind=False)
        self.assertLessEqual(len(labels), _MAX_COLS)
        # Breite = max 8 * _CELL_SIZE
        self.assertLessEqual(grid.shape[1], _MAX_COLS * _CELL_SIZE)

    def test_grid_no_reference(self):
        """Ohne Vorlage: nur Original + Wahrheit + Kandidaten."""
        case_no_ref = Case(name="no_ref", image=self.img, truth=self.truth)
        grid, labels = make_case_grid(case_no_ref, self.candidates, blind=False)
        # Original + Wahrheit + 1 Kandidat = 3
        self.assertEqual(len(labels), 3)

    def test_grid_crop(self):
        """Mit Crop: Zellen sind höher (zweite Zeile)."""
        grid_no_crop, _ = make_case_grid(self.case, self.candidates, blind=False, crop=None)
        grid_crop, _ = make_case_grid(self.case, self.candidates, blind=False, crop=(10, 10, 32, 32))
        self.assertGreater(grid_crop.shape[0], grid_no_crop.shape[0])

    def test_overall_grid(self):
        """Gesamtraster: Zeilen = Fälle."""
        grid1, _ = make_case_grid(self.case, self.candidates, blind=False)
        case2 = Case(name="t2", image=self.img, reference=self.ref, truth=self.truth)
        grid2, _ = make_case_grid(case2, self.candidates, blind=False)
        overall = make_overall_grid([grid1, grid2], ["t1", "t2"])
        self.assertEqual(overall.shape[0], grid1.shape[0] + grid2.shape[0])


class TestBlindTest(unittest.TestCase):
    """Blindtest: Zelle unter Buchstabe X bitgleich mit Kandidat laut key.json."""

    def setUp(self):
        self.device = _get_device()
        self.img = _make_test_image(64, 96, seed=10).to(self.device)
        self.ref = _make_test_image(64, 96, seed=11).to(self.device)
        self.truth = _make_test_image(64, 96, seed=12).to(self.device)

        self.case = Case(
            name="blind_test",
            image=self.img,
            reference=self.ref,
            truth=self.truth,
        )

        # Zwei Kandidaten mit unterschiedlichen Ergebnissen
        fn1 = get_method("reinhard")(self.img, self.ref)
        fn2 = get_method("mkl")(self.img, self.ref)
        self.candidates = [
            Candidate(label="reinhard", kind="method", value="reinhard", fn=fn1),
            Candidate(label="mkl", kind="method", value="mkl", fn=fn2),
        ]

    def test_blind_labels_are_letters(self):
        """Im Blindtest sind Kandidaten-Beschriftungen Buchstaben."""
        grid, labels = make_case_grid(self.case, self.candidates, blind=True, seed=42)
        # Original, Vorlage, Wahrheit, A, B
        self.assertEqual(labels[-1], "B")
        self.assertEqual(labels[-2], "A")

    def test_blind_deterministic(self):
        """Gleicher Seed → gleiche Reihenfolge."""
        grid1, labels1 = make_case_grid(self.case, self.candidates, blind=True, seed=42)
        grid2, labels2 = make_case_grid(self.case, self.candidates, blind=True, seed=42)
        self.assertEqual(labels1, labels2)
        # Bitgleich
        self.assertTrue(np.array_equal(grid1, grid2))

    def test_blind_different_seeds(self):
        """Verschiedene Seeds → (wahrscheinlich) verschiedene Reihenfolge."""
        # Mit nur 2 Kandidaten: 2 mögliche Permutationen
        _, labels1 = make_case_grid(self.case, self.candidates, blind=True, seed=42)
        _, labels2 = make_case_grid(self.case, self.candidates, blind=True, seed=43)
        # Die Labels sind immer A, B — aber die Zuordnung zu den Kandidaten
        # kann anders sein. Wir prüfen über key.json-Logik.
        # (Hier nur: Labels sind immer Buchstaben)
        self.assertIn("A", labels1)
        self.assertIn("A", labels2)

    def test_blind_cell_content(self):
        """Zelle A im Blindtest = Ergebnis des Kandidaten, der laut Key A ist."""
        seed = 42
        grid, labels = make_case_grid(self.case, self.candidates, blind=True, seed=seed)

        # Key nachmachen (wie in run_shootout)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.candidates))
        # indices[0] → Kandidat A, indices[1] → Kandidat B

        # Zelle A = erste Kandidaten-Zelle im Raster
        # Raster: Original | Vorlage | Wahrheit | A | B
        cell_a_x = 3 * _CELL_SIZE  # 4. Spalte
        cell_a = grid[:, cell_a_x:cell_a_x + _CELL_SIZE, :]

        # Erwartet: Ergebnis von candidates[indices[0]]
        expected_img = self.candidates[indices[0]].fn(self.img).clamp(0.0, 1.0)
        expected_bgr = _tensor_to_bgr(expected_img)
        expected_resized = _resize_cell(expected_bgr)

        # Die Zelle enthält: Bild + Label darunter
        # Wir vergleichen nur den Bildteil (ohne Label)
        img_h = expected_resized.shape[0]
        cell_img_part = cell_a[:img_h, :, :]

        # cv2.putText ändert nur die Label-Zeile, nicht das Bild
        # Aber _resize_cell kann die Größe ändern — wir vergleichen den Bildbereich
        # Da beide _resize_cell durchlaufen, sollten sie identisch sein
        self.assertTrue(np.array_equal(cell_img_part, expected_resized),
                        f"Zelle A != Kandidat {self.candidates[indices[0]].label}")


class TestScoring(unittest.TestCase):
    """--score mit erfundener votes.csv ergibt erwartete Rangfolge."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="shootout_test_")
        self.device = _get_device()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_key(self, key: dict[str, list[str]]):
        Path(self.tmpdir, "key.json").write_text(
            json.dumps(key, indent=2), encoding="utf-8"
        )

    def _write_votes(self, votes: list[tuple[str, str, int]]):
        with open(Path(self.tmpdir, "votes.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fall", "buchstabe", "rang"])
            for case, letter, rank in votes:
                writer.writerow([case, letter, rank])

    def _write_results(self, results: dict):
        Path(self.tmpdir, "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    def test_score_ranking(self):
        """Drei Kandidaten, klare Rangfolge."""
        key = {
            "case1": ["reinhard", "mkl", "histogram"],
            "case2": ["reinhard", "mkl", "histogram"],
        }
        # A=reinhard, B=mkl, C=histogram
        # Stimmen: reinhard gewinnt beide, mkl zweiter, histogram dritter
        votes = [
            ("case1", "A", 1), ("case1", "B", 2), ("case1", "C", 3),
            ("case2", "A", 1), ("case2", "B", 2), ("case2", "C", 3),
        ]
        results = {
            "cases": [
                {"name": "case1", "metrics": [
                    {"candidate": "reinhard", "delta_e00": 1.0, "structure": 0.95, "banding": 0.01},
                    {"candidate": "mkl", "delta_e00": 2.0, "structure": 0.90, "banding": 0.02},
                    {"candidate": "histogram", "delta_e00": 3.0, "structure": 0.85, "banding": 0.03},
                ]},
                {"name": "case2", "metrics": [
                    {"candidate": "reinhard", "delta_e00": 1.5, "structure": 0.93, "banding": 0.01},
                    {"candidate": "mkl", "delta_e00": 2.5, "structure": 0.88, "banding": 0.02},
                    {"candidate": "histogram", "delta_e00": 3.5, "structure": 0.83, "banding": 0.03},
                ]},
            ]
        }
        self._write_key(key)
        self._write_votes(votes)
        self._write_results(results)

        lb_path = run_score(self.tmpdir)
        self.assertTrue(lb_path.exists())

        content = lb_path.read_text(encoding="utf-8")
        # Reinhard muss Rang 1 sein
        lines = content.split("\n")
        table_lines = [l for l in lines if l.startswith("| 1 ")]
        self.assertTrue(any("reinhard" in l for l in table_lines),
                        f"Reinhard nicht auf Rang 1: {table_lines}")

    def test_score_contradiction(self):
        """Kandidat gewinnt Stimmen, hat aber schlechtes ΔE00 → Widerspruch."""
        key = {
            "case1": ["good_votes", "bad_votes"],
        }
        # good_votes gewinnt, hat aber schlechtes ΔE00
        votes = [
            ("case1", "A", 1),  # good_votes Rang 1
            ("case1", "B", 2),  # bad_votes Rang 2
        ]
        results = {
            "cases": [
                {"name": "case1", "metrics": [
                    {"candidate": "good_votes", "delta_e00": 10.0, "structure": 0.50, "banding": 0.10},
                    {"candidate": "bad_votes", "delta_e00": 1.0, "structure": 0.95, "banding": 0.01},
                ]},
            ]
        }
        self._write_key(key)
        self._write_votes(votes)
        self._write_results(results)

        lb_path = run_score(self.tmpdir)
        content = lb_path.read_text(encoding="utf-8")
        # good_votes hat Rang 1, aber ΔE00=10 >> Median(5.5) → Widerspruch
        self.assertIn("good_votes", content)
        # Widerspruch markiert
        contradiction_lines = [l for l in content.split("\n") if "⚠️" in l and "good_votes" in l]
        self.assertTrue(len(contradiction_lines) > 0,
                        f"Widerspruch nicht markiert für good_votes:\n{content}")

    def test_score_no_votes_file(self):
        """Ohne votes.csv → Exit 1."""
        with self.assertRaises(SystemExit):
            run_score(self.tmpdir)


class TestHelperFunctions(unittest.TestCase):
    """Hilfsfunktionen."""

    def test_tensor_to_bgr_shape(self):
        img = _make_test_image(32, 48, seed=0)
        bgr = _tensor_to_bgr(img)
        self.assertEqual(bgr.shape, (32, 48, 3))
        self.assertEqual(bgr.dtype, np.uint8)

    def test_resize_cell(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        resized = _resize_cell(img, 512)
        self.assertEqual(resized.shape[1], 512)  # lange Seite
        self.assertEqual(resized.shape[0], 256)  # kurze Seite

    def test_add_label(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        labeled = _add_label(img, "test")
        self.assertGreater(labeled.shape[0], 100)
        self.assertEqual(labeled.shape[1], 100)

    def test_random_slider_values_deterministic(self):
        v1 = _random_slider_values(42)
        v2 = _random_slider_values(42)
        self.assertEqual(v1, v2)

    def test_random_slider_values_different_seeds(self):
        v1 = _random_slider_values(42)
        v2 = _random_slider_values(43)
        self.assertNotEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
