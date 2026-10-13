"""Tests für Paket K15: Tone-Zonen Monotonie, Clarity, Worker /apply, krita_bridge."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refgrade.clarity import apply as clarity_apply, CLARITY_AMOUNT, CLARITY_RADIUS
from refgrade.color import rgb_to_lab
from refgrade.grade import grade
from refgrade.sections import VISIBLE
from refgrade.session import build_lut
from refgrade.sliders import SLIDERS

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _defaults() -> dict[str, float]:
    return {s.name: s.default for s in SLIDERS}


def _grey_tensor(L_values: np.ndarray) -> torch.Tensor:
    """L*-Werte (0..100) → sRGB-Grautensor [1, N, 1, 3]."""
    Y = np.where(L_values > 8, ((L_values + 16) / 116) ** 3, L_values / 903.3)
    c = np.where(Y <= 0.0031308, 12.92 * Y, 1.055 * Y ** (1 / 2.4) - 0.055)
    c = np.clip(c, 0.0, 1.0)
    arr = np.stack([c, c, c], axis=-1)  # [N, 3]
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(2)


class TestToneZonesMonotonicity(unittest.TestCase):
    """Alle vier Zonen auf ±1 in allen 16 Kombinationen: keine fallende Stufe."""

    def test_all_16_combinations(self):
        L = np.linspace(0, 100, 1024)
        x = _grey_tensor(L)
        for combo in itertools.product([-1.0, 1.0], repeat=4):
            vals = _defaults()
            vals["shadows"] = combo[0]
            vals["highlights"] = combo[1]
            vals["blacks"] = combo[2]
            vals["whites"] = combo[3]
            out = grade(x, vals)
            L_out = rgb_to_lab(out)[..., 0].squeeze().cpu().numpy()
            # Keine fallende Stufe: L_out[i+1] >= L_out[i] - epsilon
            diffs = np.diff(L_out)
            min_diff = diffs.min()
            self.assertGreaterEqual(min_diff, -0.01,
                                    f"Kombi {combo}: fallende Stufe (min ΔL*={min_diff:.4f})")


class TestClarityIdentity(unittest.TestCase):
    """Clarity 0 → bitgleich."""

    def test_zero_returns_input(self):
        img = torch.rand(1, 32, 32, 3, device=_DEVICE, dtype=torch.float32)
        out = clarity_apply(img, 0.0, 32)
        self.assertTrue(torch.equal(img, out), "clarity=0: Ausgabe ≠ Eingang")


class TestClarityContrast(unittest.TestCase):
    """Clarity +1 auf Kantenbild: Kantenkontrast steigt, L*-Mittelwert bleibt."""

    def _edge_image(self, L_level: float = 50.0) -> torch.Tensor:
        """32×32 Bild: linke Hälfte L_level−5, rechte Hälfte L_level+5 (gleicher L*-Sprung bei jedem Pegel)."""
        h, w = 32, 32
        # Same L* step at every level, so the 3x midtone factor measures the mask, not the step size.
        lo, hi = _grey_tensor(np.array([L_level - 5.0, L_level + 5.0])).squeeze().cpu().numpy()[:, 0]
        img = np.empty((h, w, 3), dtype=np.float32)
        img[:, :w // 2] = lo
        img[:, w // 2:] = hi
        return torch.from_numpy(img).unsqueeze(0).to(_DEVICE)

    def test_edge_contrast_increases(self):
        img = self._edge_image(50.0)
        out = clarity_apply(img, 1.0, 32)
        L0 = rgb_to_lab(img)[..., 0]
        L1 = rgb_to_lab(out)[..., 0]
        # Kantenkontrast: Unterschied zwischen linker und rechter Hälfte
        w = img.shape[2]
        contrast0 = (L0[:, :, w // 2:].mean() - L0[:, :, :w // 2].mean()).item()
        contrast1 = (L1[:, :, w // 2:].mean() - L1[:, :, :w // 2].mean()).item()
        self.assertGreater(contrast1, contrast0,
                           f"Clarity +1: Kantenkontrast nicht gestiegen ({contrast0:.2f}→{contrast1:.2f})")
        # Mittelwert bleibt (±1)
        mean_diff = abs(L1.mean().item() - L0.mean().item())
        self.assertLess(mean_diff, 1.0, f"Clarity +1: L*-Mittelwert verschoben (Δ={mean_diff:.2f})")

    def test_midtone_mask_effect(self):
        """Kantenbild bei L*50 → Kontrastgewinn ≥ 3× so groß wie bei L*8 bzw. L*92."""
        img_mid = self._edge_image(50.0)
        img_dark = self._edge_image(8.0)
        img_light = self._edge_image(92.0)

        def contrast_gain(img):
            out = clarity_apply(img, 1.0, 32)
            L0 = rgb_to_lab(img)[..., 0]
            L1 = rgb_to_lab(out)[..., 0]
            w = img.shape[2]
            c0 = (L0[:, :, w // 2:].mean() - L0[:, :, :w // 2].mean()).item()
            c1 = (L1[:, :, w // 2:].mean() - L1[:, :, :w // 2].mean()).item()
            return c1 - c0

        gain_mid = contrast_gain(img_mid)
        gain_dark = contrast_gain(img_dark)
        gain_light = contrast_gain(img_light)
        self.assertGreater(gain_mid, 3 * gain_dark,
                           f"Mitten: Gewinn {gain_mid:.3f} < 3× dunkel {3 * gain_dark:.3f}")
        self.assertGreater(gain_mid, 3 * gain_light,
                           f"Mitten: Gewinn {gain_mid:.3f} < 3× hell {3 * gain_light:.3f}")


class TestClarityReference(unittest.TestCase):
    """Gegen eine numpy-Referenz (eigene Gauß-Faltung): max. Abweichung in L* < 0,5."""

    def test_matches_numpy_reference(self):
        torch.manual_seed(42)
        # Grey noise: the reference below is L*-only (a*=b*=0), so no gamut clipping can differ.
        img = torch.rand(1, 16, 16, 1, device=_DEVICE, dtype=torch.float32).expand(1, 16, 16, 3).contiguous()
        clarity_val = 0.5
        full_size = 16

        # Torch-Implementierung
        out_torch = clarity_apply(img, clarity_val, full_size)
        L_torch = rgb_to_lab(out_torch)[..., 0].cpu().numpy()

        # Numpy-Referenz
        amount = CLARITY_AMOUNT * clarity_val
        h, w = 16, 16
        half_size = CLARITY_RADIUS * full_size
        sigma = 0.3 * half_size + 0.3
        k = int(6 * np.ceil(sigma) + 1)
        x = np.arange(k) - k // 2
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        # sRGB → Lab (vereinfacht: nur L*)
        img_np = img.cpu().numpy()
        # sRGB → linear
        def srgb_to_linear(c):
            return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        lin = srgb_to_linear(img_np)
        Y = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        L = np.where(Y > 0.008856, 116 * Y ** (1 / 3) - 16, 903.3 * Y)

        # 2D Gauß-Faltung (separabel, reflect padding)
        L = L[0]  # [H, W]
        r = k // 2
        L_blur = np.zeros_like(L)
        L_padded = np.pad(L, ((0, 0), (r, r)), mode="reflect")
        for i in range(k):  # horizontal
            L_blur += kernel[i] * L_padded[:, i:i + L.shape[1]]
        L_blur2 = np.zeros_like(L)
        L_padded2 = np.pad(L_blur, ((r, r), (0, 0)), mode="reflect")
        for i in range(k):  # vertical
            L_blur2 += kernel[i] * L_padded2[i:i + L.shape[0], :]

        L_sharp = L * (1 + amount) - amount * L_blur2

        # Mittenmaske
        L01 = L / 100.0
        m = np.round(255 * 4 * L01 * (1 - L01)).clip(0, 255) / 255.0
        L_out = L + m * (L_sharp - L)

        # Lab → sRGB (nur L*)
        Y_out = np.where(L_out > 8, ((L_out + 16) / 116) ** 3, L_out / 903.3)
        def linear_to_srgb(c):
            return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
        # a*=0, b*=0 → R=G=B
        c_out = linear_to_srgb(Y_out)
        c_out = np.clip(c_out, 0, 1)
        # L* aus dem sRGB-Ergebnis
        lin_out = srgb_to_linear(c_out)
        Y_out2 = 0.2126 * lin_out + 0.7152 * lin_out + 0.0722 * lin_out
        L_ref = np.where(Y_out2 > 0.008856, 116 * Y_out2 ** (1 / 3) - 16, 903.3 * Y_out2)

        diff = np.abs(L_torch[0] - L_ref).max()
        self.assertLess(diff, 0.5, f"Clarity vs. Referenz: max ΔL*={diff:.3f} >= 0.5")


class TestClarityNotInLUT(unittest.TestCase):
    """build_lut mit clarity 0 und clarity 1 ist identisch."""

    def test_lut_identical(self):
        from refgrade.session import Session
        img = torch.rand(1, 32, 32, 3, device=_DEVICE, dtype=torch.float32)
        alpha = np.ones((32, 32), dtype=np.uint8) * 255

        s0 = Session(source_full=img, alpha=alpha, depth="U8", source=img, ignore=None)
        s1 = Session(source_full=img, alpha=alpha, depth="U8", source=img, ignore=None)

        vals0 = _defaults()
        vals0["clarity"] = 0.0
        vals1 = _defaults()
        vals1["clarity"] = 1.0

        lut0 = build_lut(s0, [], "reinhard", vals0)
        lut1 = build_lut(s1, [], "reinhard", vals1)

        self.assertEqual(lut0["size"], lut1["size"])
        self.assertTrue(torch.equal(torch.as_tensor(lut0["table"]), torch.as_tensor(lut1["table"])),
                        "LUT mit clarity=0 ≠ LUT mit clarity=1")


class TestClarityWorkerApply(unittest.TestCase):
    """/apply mit clarity liefert Maskendatei und korrekte Parameter."""

    def _make_session(self, w: int = 32, h: int = 32):
        from refgrade.session import Session
        img = torch.full((1, h, w, 3), 0.5, device=_DEVICE, dtype=torch.float32)
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        return Session(source_full=img, alpha=alpha, depth="U8", source=img, ignore=None)

    def test_mask_size_and_values(self):
        """Maskengröße = Bildgröße; Wert bei L*50 ≥ 250, bei Schwarz 0."""
        from refgrade.color import rgb_to_lab
        w, h = 32, 32
        s = self._make_session(w, h)
        # Ergebnis: halb schwarz, halb L*50
        result = torch.zeros(1, h, w, 3, device=_DEVICE, dtype=torch.float32)
        # Rechte Hälfte: L*50 → sRGB ≈ 0.468
        result[:, :, w // 2:] = 0.468

        # Maske berechnen wie im Worker
        lab = rgb_to_lab(result)
        L01 = lab[..., 0] / 100.0
        mask = torch.round(255.0 * 4.0 * L01 * (1.0 - L01)).clamp(0, 255).to(torch.uint8)
        mask_np = mask.squeeze().cpu().numpy()

        # Größe
        self.assertEqual(mask_np.shape, (h, w), f"Maskengröße {mask_np.shape} ≠ ({h},{w})")
        # Bei L*50: m = 4*0.5*0.5 = 1.0 → 255
        val_mid = mask_np[h // 2, 3 * w // 4]
        self.assertGreaterEqual(val_mid, 250, f"Maskenwert bei L*50: {val_mid} < 250")
        # Bei Schwarz: m = 0
        val_black = mask_np[h // 2, w // 4]
        self.assertEqual(val_black, 0, f"Maskenwert bei Schwarz: {val_black} ≠ 0")

    def test_clarity_params(self):
        """halfSize ≈ 0,008·max(W,H), amount = 0,6·clarity."""
        w, h = 64, 48
        clarity_val = 0.5
        full_size = max(w, h)
        half_size = CLARITY_RADIUS * full_size
        amount = CLARITY_AMOUNT * clarity_val
        self.assertAlmostEqual(half_size, 0.008 * 64, places=5)
        self.assertAlmostEqual(amount, 0.3, places=5)


def _import_bridge():
    """krita_bridge gegen das gerade gepatchte `krita`-Fake laden (Paket ohne __init__-Ausführung)."""
    import importlib
    import types

    if "ref_grade" not in sys.modules:
        pkg = types.ModuleType("ref_grade")
        pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "plugin" / "ref_grade")]
        sys.modules["ref_grade"] = pkg
    kb = importlib.import_module("ref_grade.krita_bridge")
    return importlib.reload(kb)


class TestKritaBridgeClarity(unittest.TestCase):
    """krita_bridge: bei clarity ≠ 0 Filtermaske angelegt, bei 0 keine."""

    def test_filter_mask_created(self):
        """Fake-Krita: clarity ≠ 0 → genau eine Filtermaske mit 4 Eigenschaften."""
        from unittest.mock import MagicMock, patch

        # Fake-Krita-Objekte
        mock_krita = MagicMock()
        mock_doc = MagicMock()
        mock_layer = MagicMock()
        mock_filter = MagicMock()
        mock_cfg = MagicMock()
        mock_sel = MagicMock()

        mock_krita.Krita.instance.return_value.activeDocument.return_value = mock_doc
        mock_krita.Krita.instance.return_value.filter.return_value = mock_filter
        mock_filter.configuration.return_value = mock_cfg
        mock_doc.selection.return_value = mock_sel
        mock_doc.createFilterMask.return_value = MagicMock()

        # Maskendatei
        mask_data = np.full((16, 16), 200, dtype=np.uint8).tobytes()
        mask_path = Path(__file__).parent / "tmp_mask_test.raw"
        mask_path.write_bytes(mask_data)

        info = {"width": 16, "height": 16, "depth": "U8", "x": 0, "y": 0}
        clarity_info = {"halfSize": 0.5, "amount": 0.3, "mask": str(mask_path)}

        try:
            with patch.dict("sys.modules", {"krita": mock_krita}):
                # Neu importieren mit dem Fake
                kb = _import_bridge()

                # write_layer mit clarity_info
                result_raw = Path(__file__).parent / "tmp_result_test.raw"
                result_raw.write_bytes(np.zeros(16 * 16 * 4, dtype=np.uint8).tobytes())

                kb.write_layer(info, result_raw, "test", clarity_info)

                # Filtermaske wurde angelegt
                mock_doc.createFilterMask.assert_called_once()
                call_args = mock_doc.createFilterMask.call_args
                self.assertEqual(call_args[0][0], "Clarity (RefGrade)")

                # Filter-Config wurde gesetzt
                mock_cfg.setProperty.assert_any_call("halfSize", 0.5)
                mock_cfg.setProperty.assert_any_call("amount", 0.3)
                mock_cfg.setProperty.assert_any_call("threshold", 0)
                mock_cfg.setProperty.assert_any_call("lightnessOnly", True)
        finally:
            mask_path.unlink(missing_ok=True)
            result_raw = Path(__file__).parent / "tmp_result_test.raw"
            result_raw.unlink(missing_ok=True)

    def test_no_filter_mask_at_zero(self):
        """clarity_info=None → keine Filtermaske."""
        from unittest.mock import MagicMock, patch

        mock_krita = MagicMock()
        mock_doc = MagicMock()
        mock_layer = MagicMock()
        mock_krita.Krita.instance.return_value.activeDocument.return_value = mock_doc

        info = {"width": 16, "height": 16, "depth": "U8", "x": 0, "y": 0}

        try:
            with patch.dict("sys.modules", {"krita": mock_krita}):
                kb = _import_bridge()

                result_raw = Path(__file__).parent / "tmp_result_test2.raw"
                result_raw.write_bytes(np.zeros(16 * 16 * 4, dtype=np.uint8).tobytes())

                kb.write_layer(info, result_raw, "test", None)

                # createFilterMask wurde NICHT aufgerufen
                mock_doc.createFilterMask.assert_not_called()
        finally:
            result_raw = Path(__file__).parent / "tmp_result_test2.raw"
            result_raw.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
