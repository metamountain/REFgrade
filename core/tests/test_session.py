"""Tests für session: Sitzungslogik, LUT, Vorschau, Apply, Fit, Rank, Cube."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from refgrade import rawio
from refgrade.color import rgb_to_lab
from refgrade.lut import read_cube
from refgrade.session import (
    APPLY_CHUNK,
    HIST_BINS,
    LUT_SIZE,
    PREVIEW_SIDE,
    REF_SIDE,
    Session,
    apply_full,
    build_lut,
    cube_path,
    fit,
    is_identity,
    open_session,
    preview,
    rank,
    thumbnail,
    values_with_defaults,
    write_cube,
)

_CORE_DIR = Path(__file__).resolve().parents[1]
_POOL_DIR = str(_CORE_DIR.parent / "bench" / "pool")


def _tensor_from_array(arr: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] → [1,H,W,3] float32 0..1."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _make_session(source_full: torch.Tensor, device: torch.device | None = None) -> Session:
    """Session direkt aus Tensor bauen (ohne Rohdatei)."""
    if device is None:
        device = torch.device("cpu")
    h, w = source_full.shape[1], source_full.shape[2]
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


class TestValuesWithDefaults(unittest.TestCase):
    """values_with_defaults und is_identity."""

    def test_identity_true(self):
        vals = values_with_defaults({"light_strength": 0, "color_strength": 0})
        self.assertTrue(is_identity(vals))

    def test_identity_false_saturation(self):
        vals = values_with_defaults({"light_strength": 0, "color_strength": 0, "saturation": 1.5})
        self.assertFalse(is_identity(vals))

    def test_unknown_slider(self):
        with self.assertRaises(KeyError):
            values_with_defaults({"nonexistent_slider": 1.0})


class TestIdentityByteEqual(unittest.TestCase):
    """Identität bytegleich: U16-Quelle mit Stärke 0 durch apply_full + write_raw."""

    def test_u16_identity(self):
        rng = np.random.default_rng(0)
        h, w = 32, 48
        rgba = rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            path = f.name
        try:
            with open(path, "wb") as f:
                f.write(rgba.tobytes())
            s = open_session(path, w, h, "U16", device=torch.device("cpu"))
            vals = values_with_defaults({"light_strength": 0, "color_strength": 0})
            result = apply_full(s, [], "reinhard", vals)
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


class TestReinhard(unittest.TestCase):
    """Reinhard: Quelle=astronaut, Vorlage=coffee, Stärken 1."""

    @classmethod
    def setUpClass(cls):
        from skimage import data
        astronaut = data.astronaut()  # [512,512,3] uint8
        coffee = data.coffee()  # [300,400,3] uint8
        cls.source = _tensor_from_array(astronaut)
        cls.ref_path = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(cls.source, str(cls.ref_path))
        cls.ref_file = Path(tempfile.mktemp(suffix=".png"))
        from PIL import Image
        Image.fromarray(coffee).save(str(cls.ref_file), format="PNG")

    @classmethod
    def tearDownClass(cls):
        for p in (cls.ref_path, cls.ref_file):
            if p.exists():
                os.unlink(p)

    def test_lab_means(self):
        """Mittelwerte der Vorschau weichen vom Vorlagenmittel ab: L*≤0.5, a*≤1.0, b*≤3.5."""
        s = _make_session(self.source, device=torch.device("cpu"))
        vals = values_with_defaults({"light_strength": 1, "color_strength": 1})
        result, _ = preview(s, [str(self.ref_file)], "reinhard", vals, max_side=1536)
        ref_img = rawio.load_image(str(self.ref_file), REF_SIDE)
        L_r, a_r, b_r = _lab_mean(ref_img)
        L_p, a_p, b_p = _lab_mean(result)
        self.assertLessEqual(abs(L_p - L_r), 0.5)
        self.assertLessEqual(abs(a_p - a_r), 1.0)
        self.assertLessEqual(abs(b_p - b_r), 3.5)

    def test_self_reference_delta_e(self):
        """Vorlage = Quelle: mittleres ΔE00 < 0.1."""
        from colour.difference import delta_E_CIE2000
        s = _make_session(self.source, device=torch.device("cpu"))
        vals = values_with_defaults({"light_strength": 1, "color_strength": 1})
        result, _ = preview(s, [str(self.ref_path)], "reinhard", vals, max_side=1536)
        # ΔE00 zwischen Vorschau und Quelle (beide sRGB 0..1)
        rgb1 = result.squeeze(0).numpy()
        rgb2 = self.source.squeeze(0).numpy()
        # Auf ein vernünftiges Subsample begrenzen
        n = min(50000, rgb1.size // 3)
        idx = np.random.default_rng(0).choice(rgb1.shape[0] * rgb1.shape[1], n, replace=False)
        idx_r, idx_c = np.unravel_index(idx, (rgb1.shape[0], rgb1.shape[1]))
        c1 = rgb1[idx_r, idx_c]
        c2 = rgb2[idx_r, idx_c]
        de = delta_E_CIE2000(c1, c2)
        self.assertLess(de.mean(), 0.1)


class TestApplyFullVsPreview(unittest.TestCase):
    """apply_full gegen preview bei 1200×900: mittlere Abweichung < 1/255."""

    def test_consistency(self):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 256, size=(900, 1200, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        # Referenz: einfach ein anderes Bild
        ref_arr = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(_tensor_from_array(ref_arr), str(ref_file))
        try:
            vals = values_with_defaults({"light_strength": 0.8, "color_strength": 0.8})
            full_result = apply_full(s, [str(ref_file)], "reinhard", vals)
            prev_result, _ = preview(s, [str(ref_file)], "reinhard", vals, max_side=1536)
            # Beide sollten gleiche Größe haben (1200×900 < 1536)
            self.assertEqual(full_result.shape, prev_result.shape)
            diff = (full_result - prev_result).abs().mean().item()
            self.assertLess(diff, 1.0 / 255.0)
        finally:
            os.unlink(ref_file)


class TestHistograms(unittest.TestCase):
    """Histogramme: drei Listen mit je 64 Werten, Summe je 1 ± 1e-6."""

    def test_histograms(self):
        rng = np.random.default_rng(2)
        arr = rng.integers(0, 256, size=(100, 150, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        ref_arr = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(_tensor_from_array(ref_arr), str(ref_file))
        try:
            vals = values_with_defaults({"light_strength": 1, "color_strength": 1})
            _, hists = preview(s, [str(ref_file)], "reinhard", vals, max_side=1536)
            self.assertIn("source", hists)
            self.assertIn("reference", hists)
            self.assertIn("result", hists)
            for name in ("source", "reference", "result"):
                self.assertEqual(len(hists[name]), HIST_BINS)
                self.assertAlmostEqual(sum(hists[name]), 1.0, places=6)
        finally:
            os.unlink(ref_file)


class TestCubePath(unittest.TestCase):
    """cube_path: eindeutige Namen, keine Überschreibung."""

    def test_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = "Haus G.jpg"
            p1 = cube_path(tmpdir, ref)
            self.assertEqual(p1.name, "Haus G.cube")
            # Erste Datei erstellen
            p1.write_text("dummy")
            p2 = cube_path(tmpdir, ref)
            self.assertEqual(p2.name, "Haus G_02.cube")
            p2.write_text("dummy")
            p3 = cube_path(tmpdir, ref)
            self.assertEqual(p3.name, "Haus G_03.cube")

    def test_invalid_chars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = 'A<>:"/\\|?*B.jpg'
            p = cube_path(tmpdir, ref)
            self.assertNotIn("<", p.name)
            self.assertNotIn(">", p.name)
            self.assertNotIn(":", p.name)
            self.assertNotIn('"', p.name)
            self.assertNotIn("|", p.name)
            self.assertNotIn("?", p.name)
            self.assertNotIn("*", p.name)


class TestWriteCubeRoundtrip(unittest.TestCase):
    """write_cube → read_cube: dieselbe Tabelle (< 1e-6)."""

    def test_roundtrip(self):
        rng = np.random.default_rng(3)
        arr = rng.integers(0, 256, size=(50, 60, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(source, str(ref_file))
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                vals = values_with_defaults({"light_strength": 1, "color_strength": 1})
                path = write_cube(s, [str(ref_file)], "reinhard", vals, tmpdir)
                self.assertTrue(path.exists())
                written = read_cube(str(path))
                expected = build_lut(s, [str(ref_file)], "reinhard", vals)
                diff = (written["table"] - expected["table"]).abs().max().item()
                self.assertLess(diff, 1e-6)
            finally:
                os.unlink(ref_file)


class TestRank(unittest.TestCase):
    """rank mit bench\\pool (10 Bilder)."""

    def test_rank(self):
        if not os.path.isdir(_POOL_DIR):
            self.skipTest("bench/pool nicht vorhanden")
        rng = np.random.default_rng(4)
        arr = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        results = rank(s, _POOL_DIR, k=9)
        self.assertLessEqual(len(results), 9)
        # Absteigend
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i][1], results[i + 1][1])
        # Alle Pfade existieren
        for path, _ in results:
            self.assertTrue(os.path.isfile(path))

    def test_rank_exclude(self):
        if not os.path.isdir(_POOL_DIR):
            self.skipTest("bench/pool nicht vorhanden")
        rng = np.random.default_rng(5)
        arr = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        all_files = sorted(
            f for f in os.listdir(_POOL_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not all_files:
            self.skipTest("Keine Bilder in bench/pool")
        excluded = os.path.join(_POOL_DIR, all_files[0])
        results = rank(s, _POOL_DIR, k=9, exclude=(excluded,))
        for path, _ in results:
            self.assertNotEqual(path, excluded)


class TestErrors(unittest.TestCase):
    """Unbekannte Methode → ValueError; unbekannter Regler → KeyError."""

    def test_unknown_method(self):
        rng = np.random.default_rng(6)
        arr = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
        source = _tensor_from_array(arr)
        s = _make_session(source, device=torch.device("cpu"))
        ref_file = Path(tempfile.mktemp(suffix=".png"))
        rawio.save_png(source, str(ref_file))
        try:
            with self.assertRaises(ValueError):
                build_lut(s, [str(ref_file)], "unknown_method", {"light_strength": 1})
        finally:
            os.unlink(ref_file)

    def test_unknown_slider(self):
        with self.assertRaises(KeyError):
            values_with_defaults({"bogus_slider": 0.5})


class TestThumbnail(unittest.TestCase):
    """thumbnail: Vorschaubild als PNG."""

    def test_basic(self):
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        src_file = Path(tempfile.mktemp(suffix=".png"))
        out_file = Path(tempfile.mktemp(suffix=".png"))
        try:
            rawio.save_png(_tensor_from_array(arr), str(src_file))
            result = thumbnail(str(src_file), str(out_file), side=128)
            self.assertEqual(result, str(out_file))
            self.assertTrue(os.path.isfile(out_file))
            from PIL import Image
            img = Image.open(out_file)
            self.assertLessEqual(max(img.size), 128)
            img.close()
        finally:
            for p in (src_file, out_file):
                if p.exists():
                    os.unlink(p)


if __name__ == "__main__":
    unittest.main()
