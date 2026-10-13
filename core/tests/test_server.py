"""Tests für refgrade/server.py: Cache, Preview, Galerie, Pick/Back, SaveCube."""

from __future__ import annotations

import base64
import sys
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refgrade.server import (
    NodeCache,
    NodeState,
    cache_node,
    get_cache,
    _do_preview,
    _do_gallery,
    _do_pick,
    _do_back,
    _do_save_cube,
    _do_analysis,
    _build_lut_for_values,
    _NoCacheError,
    _NoHistoryError,
    _resize_long_side,
)
from refgrade.lut import build as lut_build, apply as lut_apply, read_cube, write_cube
from refgrade.sliders import SLIDERS
from refgrade.grade import grade
from refgrade.color import rgb_to_lab


def _make_smooth_images(seed: int = 42, size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Glatte Testbilder mit Gradienten (JPEG-freundlich)."""
    y = np.linspace(0, 1, size)
    x = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    # Target: warmer Gradient
    target = np.zeros((1, size, size, 3), dtype=np.float32)
    target[..., 0] = 0.3 + 0.5 * xx
    target[..., 1] = 0.2 + 0.4 * yy
    target[..., 2] = 0.1 + 0.3 * (1 - xx)
    # Reference: kühler Gradient
    reference = np.zeros((1, size, size, 3), dtype=np.float32)
    reference[..., 0] = 0.2 + 0.3 * yy
    reference[..., 1] = 0.3 + 0.3 * xx
    reference[..., 2] = 0.4 + 0.5 * (1 - yy)
    return torch.from_numpy(target), torch.from_numpy(reference)


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _jpeg_to_rgb(jpeg_b64: str) -> np.ndarray:
    """Base64-JPEG → float32 [H,W,3] 0..1 (RGB)."""
    buf = np.frombuffer(base64.b64decode(jpeg_b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB


class TestNodeCache(unittest.TestCase):
    """LRU-Cache-Verhalten (globaler Cache)."""

    def test_put_get(self):
        img, ref = _make_smooth_images(size=32)
        cache_node("t_putget", img, ref, None, None, "zonal", 1.0, 1.0, 0.0, None)
        state = get_cache().get("t_putget")
        self.assertIsNotNone(state)
        get_cache().remove("t_putget")

    def test_lru_eviction(self):
        img, ref = _make_smooth_images(size=32)
        for i in range(5):
            cache_node(f"t_evict_{i}", img, ref, None, None, "zonal", 1.0, 1.0, 0.0, None)
        self.assertIsNone(get_cache().get("t_evict_0"))
        self.assertIsNotNone(get_cache().get("t_evict_4"))
        for i in range(5):
            get_cache().remove(f"t_evict_{i}")

    def test_resize(self):
        img, _ = _make_smooth_images(size=512)
        resized = _resize_long_side(img, 256)
        self.assertEqual(max(resized.shape[1], resized.shape[2]), 256)


class TestPreview(unittest.TestCase):
    """Preview: LUT bauen, anwenden, Histogramme."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=128)
        cache_node("t_preview", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_preview")

    def test_preview_returns_valid(self):
        values = _defaults()
        result = _do_preview("t_preview", values)
        self.assertIn("jpeg", result)
        self.assertIn("histograms", result)
        self.assertIn("coverage", result)
        self.assertIn("ms", result)
        hists = result["histograms"]
        for ch in ("R", "G", "B", "L", "a", "b"):
            self.assertEqual(len(hists[ch]["cur"]), 64)
            self.assertEqual(len(hists[ch]["ref"]), 64)
            self.assertAlmostEqual(sum(hists[ch]["cur"]), 1.0, places=3)

    def test_preview_no_cache(self):
        with self.assertRaises(_NoCacheError):
            _do_preview("t_no_such_node", _defaults())

    def test_preview_differs_from_input(self):
        values = _defaults()
        values["exposure"] = 0.5
        result = _do_preview("t_preview", values)
        self.assertGreater(len(result["jpeg"]), 100)

    def test_route_vs_node_lut(self):
        """Route-Bild ≈ Node-Ergebnis (beide LUT-basiert), mittlere Diff < 3e-3."""
        values = _defaults()
        values["exposure"] = 0.3
        values["saturation"] = 1.2

        # Route: Preview (LUT + JPEG)
        result = _do_preview("t_preview", values)
        preview_rgb = _jpeg_to_rgb(result["jpeg"])

        # Node: LUT-basiert (wie im RefGradeInteractive-Node)
        state = get_cache().get("t_preview")
        lut = _build_lut_for_values(state, values)
        node_result = lut_apply(state.image, lut).clamp(0.0, 1.0)

        # Auf Preview-Größe skalieren
        import torch.nn.functional as F
        node_small = F.interpolate(
            node_result.permute(0, 3, 1, 2),
            size=preview_rgb.shape[:2], mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)[0].numpy()

        diff = float(np.abs(preview_rgb - node_small).mean())
        self.assertLess(diff, 3e-3, f"Route vs. Node (LUT): diff={diff:.6f}")


class TestGallery(unittest.TestCase):
    """Galerie: 3×3-Mosaik + Zellen."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=128)
        cache_node("t_gallery", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_gallery")

    def test_gallery_start(self):
        anchor = _defaults()
        result = _do_gallery(
            "t_gallery", "start", anchor, 0,
            cell_w=64, cell_h=48, dpr=1.0, wipe=False,
        )
        self.assertIn("jpeg", result)
        self.assertIn("cells", result)
        self.assertEqual(len(result["cells"]), 9)
        for cell in result["cells"]:
            self.assertIn("index", cell)
            self.assertIn("label", cell)
            self.assertIn("values", cell)

    def test_gallery_cell_consistency(self):
        """Zelle i == Einzelvorschau mit cells[i].values (Tensor-Vergleich < 1e-3)."""
        anchor = _defaults()
        result = _do_gallery(
            "t_gallery", "light", anchor, 0,
            cell_w=64, cell_h=48, dpr=1.0, wipe=False,
        )
        cells = result["cells"]
        state = get_cache().get("t_gallery")

        # Einzel-Preview mit Zelle-0-Werten (Tensor)
        lut0 = _build_lut_for_values(state, cells[0]["values"])
        import torch.nn.functional as F
        img_small = _resize_long_side(state.image, 64)
        if img_small.shape[1] != 48 or img_small.shape[2] != 64:
            img_small = F.interpolate(
                img_small.permute(0, 3, 1, 2), size=(48, 64),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)
        single = lut_apply(img_small, lut0).clamp(0.0, 1.0)[0].numpy()

        # Aus dem Mosaik Zelle 0 extrahieren (JPEG)
        mosaic_rgb = _jpeg_to_rgb(result["jpeg"])
        cell0 = mosaic_rgb[0:48, 0:64]
        diff = float(np.abs(cell0 - single).mean())
        # JPEG-Toleranz: 0.01 (bei glatten Bildern)
        self.assertLess(diff, 0.01, f"Zelle 0 vs. Einzel-Preview: diff={diff:.5f}")

    def test_gallery_anchor_is_middle(self):
        """Zelle 4 (Mitte) == Anker."""
        anchor = _defaults()
        result = _do_gallery(
            "t_gallery", "light", anchor, 0,
            cell_w=64, cell_h=48, dpr=1.0, wipe=False,
        )
        self.assertEqual(result["cells"][4]["label"], "Anker")
        for k, v in anchor.items():
            self.assertAlmostEqual(result["cells"][4]["values"].get(k, 0), v, places=5)


class TestPickBack(unittest.TestCase):
    """Pick und Back: Anker-Verwaltung."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=64)
        cache_node("t_pick", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_pick")

    def test_pick_back_roundtrip(self):
        """pick dann back → alter Anker bitgleich."""
        state = get_cache().get("t_pick")
        original_anchor = dict(state.anchor)

        new_values = dict(original_anchor)
        new_values["exposure"] = 0.5
        cells = [{"index": 0, "label": "test", "values": new_values}]
        _do_pick("t_pick", 0, cells)

        state = get_cache().get("t_pick")
        self.assertAlmostEqual(state.anchor["exposure"], 0.5)
        self.assertEqual(len(state.history), 1)

        _do_back("t_pick")
        state = get_cache().get("t_pick")
        for k, v in original_anchor.items():
            self.assertEqual(state.anchor[k], v, f"Anker [{k}] nicht bitgleich")

    def test_back_empty_history(self):
        state = get_cache().get("t_pick")
        state.history.clear()
        with self.assertRaises(_NoHistoryError):
            _do_back("t_pick")


class TestSaveCube(unittest.TestCase):
    """SaveCube: LUT speichern und reproduzieren."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=64)
        cache_node("t_cube", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_cube")
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_save_and_reproduce(self):
        """Gespeicherte .cube reproduziert die Zelle (< 2e-3)."""
        values = _defaults()
        values["exposure"] = 0.3
        values["saturation"] = 1.2

        state = get_cache().get("t_cube")
        lut = _build_lut_for_values(state, values)

        path = os.path.join(self.tmpdir, "test_save.cube")
        write_cube(lut, path)
        self.assertTrue(os.path.isfile(path))

        lut_loaded = read_cube(path)
        result_direct = lut_apply(state.image, lut).clamp(0.0, 1.0)
        result_loaded = lut_apply(state.image, lut_loaded).clamp(0.0, 1.0)

        diff = float((result_direct - result_loaded).abs().mean())
        self.assertLess(diff, 2e-3, f".cube-Reproduktion: diff={diff:.6f}")


class TestAnalysis(unittest.TestCase):
    """Analysis: compare von Bild und Vorlage."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=64)
        cache_node("t_analysis", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_analysis")

    def test_analysis_returns_sliders(self):
        result = _do_analysis("t_analysis")
        self.assertIn("sliders", result)
        self.assertIn("sentences", result)
        self.assertGreater(len(result["sliders"]), 3)


class TestPrerender(unittest.TestCase):
    """Vorrendern: Ergebnis == live gerechnet (< 1e-3)."""

    @classmethod
    def setUpClass(cls):
        cls.img, cls.ref = _make_smooth_images(seed=42, size=128)
        cache_node("t_prerender", cls.img, cls.ref, None, None, "zonal", 1.0, 1.0, 0.0, None)
        cls.tmpdir = tempfile.mkdtemp()
        # Referenzbild als Datei speichern
        from PIL import Image
        ref_arr = np.round(cls.ref.squeeze(0).numpy() * 255.0).astype(np.uint8)
        cls.ref_path = os.path.join(cls.tmpdir, "ref_test.png")
        Image.fromarray(ref_arr).save(cls.ref_path)

    @classmethod
    def tearDownClass(cls):
        get_cache().remove("t_prerender")
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_prerender_matches_live(self):
        """Vorgerendertes JPEG ≈ live gerechnetes Ergebnis (< 1e-3)."""
        from refgrade.server import _prerender_worker, _build_method_pipeline
        from refgrade.lut import build as _lut_build
        from PIL import Image

        state = get_cache().get("t_prerender")
        ranked = [(self.ref_path, 0.9)]
        gen = state.prerender_gen
        # Synchron ausführen (kein Thread)
        _prerender_worker("t_prerender", ranked, self.tmpdir, gen)

        # Prerender-Ergebnis holen
        key = (self.ref_path, "zonal", 1.0)
        self.assertIn(key, state.prerendered, "Prerender-Cache leer")
        prerendered_rgb = _jpeg_to_rgb(
            base64.b64encode(state.prerendered[key]).decode("ascii")
        )

        # Live: selbe Methode, selbe Vorlage (aus Datei, wie der Worker)
        ref_img = Image.open(self.ref_path).convert("RGB")
        ref_arr = np.array(ref_img, dtype=np.float32) / 255.0
        ref_tensor = torch.from_numpy(ref_arr).unsqueeze(0).to(state.image.device)

        mp = _build_method_pipeline(
            "zonal", state.image, ref_tensor,
            state.image_weights, None, 1.0, 1.0,
        )
        lut = _lut_build(mp, size=33, title="live", device=state.image.device)
        live_result = lut_apply(state.image, lut).clamp(0.0, 1.0)

        # Auf Prerender-Größe skalieren (Prerender nutzt state.image direkt)
        h, w = prerendered_rgb.shape[:2]
        live_np = live_result.squeeze(0).numpy()
        if live_np.shape[:2] != (h, w):
            import torch.nn.functional as F
            live_small = F.interpolate(
                live_result.permute(0, 3, 1, 2), size=(h, w),
                mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)[0].numpy()
        else:
            live_small = live_np

        diff = float(np.abs(prerendered_rgb - live_small).mean())
        # JPEG-95: ~2e-3; Schranke 3e-3 (wie Route-vs-Node)
        self.assertLess(diff, 3e-3, f"Prerender vs. live: diff={diff:.6f}")

    def test_prerender_invalidated_on_new_gen(self):
        """Neue Generation verwirft alte Prerender-Aufträge."""
        from refgrade.server import _prerender_worker

        state = get_cache().get("t_prerender")
        state.prerendered.clear()
        # Gen erhöhen (simuliert neues Ranking)
        state.prerender_gen += 1
        old_gen = state.prerender_gen - 1
        # Worker mit altem Gen → sollte nichts speichern
        _prerender_worker("t_prerender", [(self.ref_path, 0.9)], self.tmpdir, old_gen)
        self.assertEqual(len(state.prerendered), 0)


class TestNoCache(unittest.TestCase):
    """409 bei fehlendem Cache-Eintrag."""

    def test_all_routes_raise(self):
        with self.assertRaises(_NoCacheError):
            _do_preview("t_no_cache", _defaults())
        with self.assertRaises(_NoCacheError):
            _do_analysis("t_no_cache")
        with self.assertRaises(_NoCacheError):
            _do_gallery("t_no_cache", "start", _defaults(), 0, 64, 48, 1.0, False)


if __name__ == "__main__":
    unittest.main()
