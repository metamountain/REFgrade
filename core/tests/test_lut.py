"""Tests für refgrade.lut: build, apply, mix, read/write_cube."""

import sys
import os
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import colour

from refgrade.lut import build, apply, mix, read_cube, write_cube
from refgrade.color import rgb_to_lab, lab_to_rgb
from refgrade.transfer import reinhard, mkl, histogram
from refgrade.zonal import analyze, apply_params
from refgrade.grade import grade
from refgrade.sliders import SLIDERS
from slow import slow  # noqa: E402


def _get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


class TestLUTIdentity(unittest.TestCase):
    """Identitäts-LUT: Fehler < 1e-5."""

    def test_identity_lut(self):
        dev = _get_device()
        torch.manual_seed(1)
        img = torch.rand(1, 32, 32, 3, device=dev)

        def identity_fn(x):
            return x

        lut = build(identity_fn, size=33)
        out = apply(img, lut)
        err = (out - img).abs().max().item()
        self.assertLess(err, 1e-5, f"Identitäts-LUT Fehler: {err}")


class TestLUTAccuracy(unittest.TestCase):
    """Direkt vs. über LUT: mittlere < 2e-3, max < 2e-2 (33³); mittlere < 1e-3 (65³)."""

    def _test_method(self, fn_factory, name: str):
        dev = _get_device()
        torch.manual_seed(10)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)

        fn = fn_factory(img, ref)
        direct = fn(img).clamp(0.0, 1.0)

        for size, tol_mean, tol_max in ((33, 2e-3, 2e-2), (65, 1e-3, 2e-2)):
            lut = build(fn, size=size, device=dev)
            lut_out = apply(img, lut)
            mean_err = (lut_out - direct).abs().mean().item()
            max_err = (lut_out - direct).abs().max().item()
            with self.subTest(method=name, size=size):
                self.assertLess(mean_err, tol_mean,
                                f"{name} {size}³: mittlere {mean_err:.2e} > {tol_mean}")
                self.assertLess(max_err, tol_max,
                                f"{name} {size}³: max {max_err:.2e} > {tol_max}")

    def test_reinhard(self):
        self._test_method(reinhard, "reinhard")

    def test_mkl(self):
        self._test_method(mkl, "mkl")

    @slow

    def test_histogram(self):
        self._test_method(histogram, "histogram")

    def test_zonal(self):
        dev = _get_device()
        torch.manual_seed(10)
        img = torch.rand(1, 32, 32, 3, device=dev)
        ref = torch.rand(1, 32, 32, 3, device=dev)
        params = analyze(img, ref)
        lab_fn = apply_params(params)

        def zonal_fn(x):
            lab = rgb_to_lab(x)
            lab_out = lab_fn(lab)
            return lab_to_rgb(lab_out)

        self._test_method(lambda i, r: zonal_fn, "zonal")

    def test_grade(self):
        dev = _get_device()
        torch.manual_seed(10)
        img = torch.rand(1, 32, 32, 3, device=dev)
        vals = _defaults()
        vals["exposure"] = 0.3
        vals["temperature"] = 0.2
        vals["contrast"] = 1.15
        vals["saturation"] = 1.2
        vals["shadow_amount"] = 0.2
        vals["highlight_amount"] = 0.15

        def grade_fn(x):
            return grade(x, vals)

        self._test_method(lambda i, r: grade_fn, "grade")


class TestLUTChannelIndependence(unittest.TestCase):
    """Funktion r→r² verändert nach Export/Import nur R."""

    def test_r_squared(self):
        dev = _get_device()

        def r_sq_fn(x):
            out = x.clone()
            out[..., 0] = out[..., 0] ** 2
            return out

        lut = build(r_sq_fn, size=33)
        # R-Kanal: r²
        # G-Kanal: g (Identität)
        # B-Kanal: b (Identität)

        # Test: Pixel (0.5, 0.3, 0.7)
        pixel = torch.tensor([[[[0.5, 0.3, 0.7]]]], device=dev)
        out = apply(pixel, lut)

        # R sollte ≈ 0.25 sein (0.5²)
        r_out = out[0, 0, 0, 0].item()
        self.assertAlmostEqual(r_out, 0.25, delta=0.02,
                               msg=f"R-Kanal: {r_out} (erwartet ≈0.25)")
        # G und B sollten ≈ unverändert sein
        g_out = out[0, 0, 0, 1].item()
        b_out = out[0, 0, 0, 2].item()
        self.assertAlmostEqual(g_out, 0.3, delta=0.02,
                               msg=f"G-Kanal: {g_out} (erwartet ≈0.3)")
        self.assertAlmostEqual(b_out, 0.7, delta=0.02,
                               msg=f"B-Kanal: {b_out} (erwartet ≈0.7)")


class TestLUTMix(unittest.TestCase):
    """mix t=0 == A, t=1 == B."""

    def test_mix_endpoints(self):
        dev = _get_device()
        torch.manual_seed(20)

        def fn_a(x):
            return x * 0.8

        def fn_b(x):
            return x * 1.2

        lut_a = build(fn_a, size=33, title="A")
        lut_b = build(fn_b, size=33, title="B")

        img = torch.rand(1, 16, 16, 3, device=dev)

        # t=0 → A
        mixed_0 = mix(lut_a, lut_b, 0.0)
        out_0 = apply(img, mixed_0)
        out_a = apply(img, lut_a)
        err_0 = (out_0 - out_a).abs().max().item()
        self.assertLess(err_0, 1e-6, f"mix t=0 ≠ A: {err_0}")

        # t=1 → B
        mixed_1 = mix(lut_a, lut_b, 1.0)
        out_1 = apply(img, mixed_1)
        out_b = apply(img, lut_b)
        err_1 = (out_1 - out_b).abs().max().item()
        self.assertLess(err_1, 1e-6, f"mix t=1 ≠ B: {err_1}")


class TestCubeReadWrite(unittest.TestCase):
    """.cube schreiben→lesen < 1e-6."""

    def test_roundtrip(self):
        dev = _get_device()
        torch.manual_seed(30)

        def fn(x):
            return x * 1.1

        lut = build(fn, size=17, title="Test")

        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as f:
            path = f.name

        try:
            write_cube(lut, path)
            lut2 = read_cube(path)
            self.assertEqual(lut2["size"], 17)
            self.assertEqual(lut2["title"], "Test")
            err = (lut2["table"].to(dev) - lut["table"]).abs().max().item()
            self.assertLess(err, 1e-6, f"Cube Roundtrip Fehler: {err}")
        finally:
            os.unlink(path)

    def test_read_rejects_1d(self):
        """LUT_1D_SIZE → klarer Fehler."""
        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False, mode="w") as f:
            f.write('TITLE "1D"\nLUT_1D_SIZE 33\n0.0 0.0 0.0\n1.0 1.0 1.0\n')
            path = f.name
        try:
            with self.assertRaises(ValueError):
                read_cube(path)
        finally:
            os.unlink(path)

    def test_read_rejects_bad_extension(self):
        """Falsche Endung → Fehler."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("LUT_3D_SIZE 2\n0 0 0\n0 0 1\n1 0 0\n1 0 1\n0 1 0\n0 1 1\n1 1 0\n1 1 1\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                read_cube(path)
        finally:
            os.unlink(path)


class TestCubeColourLib(unittest.TestCase):
    """Unsere Datei mit colour.read_LUT gelesen und angewandt ≈ eigene Anwendung."""

    def test_colour_read(self):
        dev = _get_device()
        torch.manual_seed(40)

        def fn(x):
            return x * 1.05

        lut = build(fn, size=17, title="ColourTest")

        with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as f:
            path = f.name

        try:
            write_cube(lut, path)

            # Mit colour lesen (Achsen: [r,g,b] → transponieren zu [b,g,r])
            lut_colour = colour.read_LUT(path)
            tc = np.asarray(lut_colour.table)  # [N,N,N,3] Index [r][g][b]
            tc_bgr = np.transpose(tc, (2, 1, 0, 3))  # → [b][g][r][3]
            data = torch.tensor(tc_bgr, dtype=torch.float32, device=dev)
            err_table = (data - lut["table"]).abs().max().item()
            self.assertLess(err_table, 1e-5,
                            f"Table-Diskrepanz: {err_table}")
        finally:
            os.unlink(path)


class TestLutLibrary(unittest.TestCase):
    """Jede Datei in der LUT-Bibliothek (RefGrade/luts/) wird gelesen und stimmt mit colour überein."""

    LUT_DIR = str(Path(__file__).resolve().parents[2] / "luts")

    @slow

    def test_all_luts(self):
        if not os.path.isdir(self.LUT_DIR):
            self.skipTest(f"LUT-Verzeichnis nicht gefunden: {self.LUT_DIR}")

        dev = _get_device()
        skipped: list[str] = []
        tested = 0

        for fname in sorted(os.listdir(self.LUT_DIR)):
            if not fname.lower().endswith(".cube"):
                continue
            path = os.path.join(self.LUT_DIR, fname)

            # Mit colour lesen (können fehlschlagen)
            try:
                lut_colour = colour.read_LUT(path)
            except Exception:
                skipped.append(fname)
                continue

            # Mit unsrem Reader lesen
            lut_ours = read_cube(path)

            # Vergleich (colour: [r,g,b] → transponieren zu [b,g,r])
            tc = np.asarray(lut_colour.table)
            tc_bgr = np.transpose(tc, (2, 1, 0, 3))
            data = torch.tensor(tc_bgr, dtype=torch.float32)
            err = (data - lut_ours["table"]).abs().max().item()
            self.assertLess(err, 1e-6,
                            f"{fname}: Diskrepanz {err:.2e}")
            tested += 1

        self.assertGreater(tested, 0, "Keine LUTs getestet")
        if skipped:
            self.log = f"Übersprungen: {skipped}"


if __name__ == "__main__":
    unittest.main()
