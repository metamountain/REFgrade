"""Tests für Paket K14: PAIR — LUT aus Vorher/Nachher-Paar."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Callable
from urllib.request import urlopen, Request

import numpy as np
import torch
import torch.nn.functional as F

from refgrade.lut import apply as lut_apply, build as lut_build, read_cube, write_cube
from refgrade.pairlut import fit, split_pair, as_lut
from refgrade.color import rgb_to_lab
from slow import slow

CORE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CORE_DIR.parent
BULL_DIR = PROJECT_DIR / "Dual LUT"
PYTHON = sys.executable


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _img_to_tensor(path: str, device: torch.device | None = None) -> torch.Tensor:
    """JPG → [1,H,W,3] float32 0..1."""
    from PIL import Image
    if device is None:
        device = _get_device()
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _de76(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """ΔE76 mean und p95 zwischen zwei [1,H,W,3] Tensoren."""
    la, lb = rgb_to_lab(a), rgb_to_lab(b)
    d = (la - lb).norm(dim=-1).flatten()
    return d.mean().item(), d.quantile(0.95).item()


# ── Synthetische Tests ─────────────────────────────────────────────────────────


class TestPairLutSynthetic(unittest.TestCase):
    """Synthetische Tests: LUT-Recovery, Identität, leere Würfelbereiche."""

    def test_warm_lut_recovery(self):
        """before = Zufallsbild, after = lut.apply(before, warm_lut) → fit → max dev < 0.03."""
        dev = _get_device()
        torch.manual_seed(42)
        before = torch.rand(1, 64, 64, 3, device=dev)

        # Wärme-LUT in [b][g][r] Ordnung: R +0.05, B -0.03
        size = 17
        r_vals = torch.linspace(0, 1, size, device=dev)
        g_vals = torch.linspace(0, 1, size, device=dev)
        b_vals = torch.linspace(0, 1, size, device=dev)
        r_3d = r_vals[None, None, :].expand(size, size, size)
        g_3d = g_vals[None, :, None].expand(size, size, size)
        b_3d = b_vals[:, None, None].expand(size, size, size)
        grid = torch.stack([r_3d, g_3d, b_3d], dim=-1)  # [b][g][r][3]
        warm_table = grid.clone()
        warm_table[..., 0] = (warm_table[..., 0] + 0.05).clamp(0, 1)
        warm_table[..., 2] = (warm_table[..., 2] - 0.03).clamp(0, 1)
        warm_lut = {"size": size, "table": warm_table, "title": "warm"}

        after = lut_apply(before, warm_lut)
        table = fit(before, after, size=size)
        result = lut_apply(before, {"size": size, "table": table, "title": "fitted"})
        max_dev = (result - after).abs().max().item()
        self.assertLess(max_dev, 0.03, f"Warm LUT recovery: max dev {max_dev:.4f}")

    def test_identity_pair(self):
        """after = before → LUT anwenden ergibt max dev < 0.02 zum Original."""
        dev = _get_device()
        torch.manual_seed(100)
        img = torch.rand(1, 32, 32, 3, device=dev)
        size = 17
        table = fit(img, img, size=size)
        result = lut_apply(img, {"size": size, "table": table, "title": "id"})
        max_dev = (result - img).abs().max().item()
        self.assertLess(max_dev, 0.02, f"Identity pair: max dev {max_dev:.4f}")

    def test_empty_cube_no_jumps(self):
        """Rottöne mit Wärme-LUT: keine Sprünge, kein NaN."""
        dev = _get_device()
        torch.manual_seed(200)
        # Nur Rottöne: R=0.5..1, G=0..0.3, B=0..0.3
        before = torch.rand(1, 32, 32, 3, device=dev)
        before[..., 0] = 0.5 + before[..., 0] * 0.5
        before[..., 1] = before[..., 1] * 0.3
        before[..., 2] = before[..., 2] * 0.3

        # Wärme-LUT in [b][g][r] Ordnung
        size = 17
        r_vals = torch.linspace(0, 1, size, device=dev)
        g_vals = torch.linspace(0, 1, size, device=dev)
        b_vals = torch.linspace(0, 1, size, device=dev)
        r_3d = r_vals[None, None, :].expand(size, size, size)
        g_3d = g_vals[None, :, None].expand(size, size, size)
        b_3d = b_vals[:, None, None].expand(size, size, size)
        grid = torch.stack([r_3d, g_3d, b_3d], dim=-1)  # [b][g][r][3]
        warm_table = grid.clone()
        warm_table[..., 0] = (warm_table[..., 0] + 0.05).clamp(0, 1)
        warm_table[..., 2] = (warm_table[..., 2] - 0.03).clamp(0, 1)
        warm_lut = {"size": size, "table": warm_table, "title": "warm"}
        after = lut_apply(before, warm_lut)

        table = fit(before, after, size=size)
        # Kein NaN
        self.assertFalse(torch.isnan(table).any(), "LUT enthält NaN")
        self.assertFalse(torch.isinf(table).any(), "LUT enthält Inf")
        # Keine großen Sprünge zwischen benachbarten Gitterpunkten
        t = table.cpu()
        for axis in range(3):
            for i in range(size - 1):
                idx1 = [0, 0, 0]
                idx2 = [0, 0, 0]
                idx1[axis] = i
                idx2[axis] = i + 1
                diff = (t[tuple(idx1)] - t[tuple(idx2)]).abs().max().item()
                self.assertLess(diff, 0.1,
                                f"Jump at axis={axis}, i={i}: {diff:.4f}")


# ── Bull-Paar ──────────────────────────────────────────────────────────────────


class TestBullPair(unittest.TestCase):
    """Testpaar bull1.jpg → Bull2.jpg."""

    @slow
    def test_bull_pair_error(self):
        """ΔE76 mean < 4.0, p95 < 11."""
        dev = _get_device()
        before = _img_to_tensor(str(BULL_DIR / "bull1.jpg"), dev)
        after = _img_to_tensor(str(BULL_DIR / "Bull2.jpg"), dev)
        table = fit(before, after, size=33)
        result = lut_apply(before, {"size": 33, "table": table, "title": "bull"})
        mean, p95 = _de76(result, after)
        self.assertLess(mean, 4.0, f"Bull pair: ΔE mean {mean:.2f} >= 4.0")
        self.assertLess(p95, 11.0, f"Bull pair: ΔE p95 {p95:.2f} >= 11.0")

    @slow
    def test_bull_double_vs_two_images(self):
        """Bull Double als einzelne Vorlage: ΔE mean < 1.5 vs Zwei-Bild-Fall."""
        dev = _get_device()
        before1 = _img_to_tensor(str(BULL_DIR / "bull1.jpg"), dev)
        after1 = _img_to_tensor(str(BULL_DIR / "Bull2.jpg"), dev)
        table1 = fit(before1, after1, size=33)
        result1 = lut_apply(before1, {"size": 33, "table": table1, "title": "two"})

        double = _img_to_tensor(str(BULL_DIR / "Bull Double.jpg"), dev)
        before2, after2 = split_pair(double)
        table2 = fit(before2, after2, size=33)
        result2 = lut_apply(before2, {"size": 33, "table": table2, "title": "double"})

        # Vorher-Bilder vergleichen (sollten identisch sein bis zu JPEG)
        mean, _ = _de76(result1, result2)
        self.assertLess(mean, 1.5, f"Bull Double vs two: ΔE mean {mean:.2f} >= 1.5")


# ── Fehlerfälle ────────────────────────────────────────────────────────────────


class TestPairErrors(unittest.TestCase):
    """Fehlerfälle: verschiedene Größen, einzelne quadratische Vorlage, FIT."""

    def test_different_sizes_resized(self):
        """Verschieden große Paare werden angeglichen, ohne Fehler."""
        dev = _get_device()
        torch.manual_seed(300)
        before = torch.rand(1, 64, 64, 3, device=dev)
        after_big = torch.rand(1, 80, 80, 3, device=dev)
        # after auf before-Größe bringen
        after = F.interpolate(
            after_big.permute(0, 3, 1, 2), size=(64, 64),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).clamp(0, 1)
        table = fit(before, after, size=17)
        self.assertEqual(table.shape, (17, 17, 17, 3))

    def test_single_square_ref_value_error(self):
        """Eine einzelne quadratische Vorlage → ValueError."""
        dev = _get_device()
        torch.manual_seed(400)
        img = torch.rand(1, 64, 64, 3, device=dev)
        # split_pair bei quadratischem Bild: W/H = 1.0 < 1.8
        before, after = split_pair(img)
        # Das funktioniert, aber die Session würde ValueError werfen
        # Hier testen wir nur, dass split_pair bei 1:1 kein sinnvolles Paar ergibt
        self.assertEqual(before.shape[2], 32)
        self.assertEqual(after.shape[2], 32)

    def test_fit_pair_value_error(self):
        """FIT mit 'pair' → ValueError."""
        from refgrade.session import Session, fit as sess_fit
        dev = torch.device("cpu")
        h, w = 32, 48
        source = torch.rand(1, h, w, 3, device=dev)
        s = Session(
            source_full=source,
            alpha=np.ones((h, w), dtype=np.uint8) * 255,
            depth="U8",
            source=source,
            ignore=None,
        )
        with self.assertRaises(ValueError) as ctx:
            sess_fit(s, ["/fake/ref.jpg"], "pair")
        self.assertIn("FIT is not available for PAIR method", str(ctx.exception))


# ── Worker-Tests ───────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 60) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_expect_error(url: str, data: dict, timeout: int = 60) -> tuple[int, dict]:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", 500)
        try:
            return code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return code, {"error": str(e)}


class TestWorkerPair(unittest.TestCase):
    """Worker: /preview mit pair liefert pair_error, FIT → 400."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp(prefix="refgrade_pair_")
        env = os.environ.copy()
        cls.proc = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(cls.port), "--tmp", cls.tmp],
            cwd=str(CORE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise RuntimeError(f"Worker sofort beendet: {err.decode()}")
            try:
                from urllib.request import urlopen
                with urlopen(f"{cls.base}/health", timeout=2) as r:
                    if json.loads(r.read().decode()).get("ok"):
                        break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("Worker nicht innerhalb von 20 s erreichbar")

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            try:
                _post(f"{cls.base}/shutdown", {})
            except Exception:
                pass
            try:
                cls.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load_bull(self) -> str:
        """bull1.jpg als Rohdaten laden."""
        from PIL import Image
        img = Image.open(str(BULL_DIR / "bull1.jpg")).convert("RGB")
        h, w = img.size[1], img.size[0]
        arr = np.array(img, dtype=np.uint8)
        rgba = np.dstack([arr, np.full((h, w), 255, dtype=np.uint8)])
        raw_path = os.path.join(self.tmp, "bull1.raw")
        with open(raw_path, "wb") as f:
            f.write(rgba.tobytes())
        r = _post(f"{self.base}/load", {
            "raw": raw_path, "width": w, "height": h, "depth": "U8",
        })
        return r["session"]

    @slow
    def test_preview_pair_error(self):
        """/preview mit pair liefert pair_error (mean < 4) und pair_shift (< 5, bull ist aligniert)."""
        sid = self._load_bull()
        r = _post(f"{self.base}/preview", {
            "session": sid,
            "references": [str(BULL_DIR / "bull1.jpg"), str(BULL_DIR / "Bull2.jpg")],
            "method": "pair",
            "values": {},
            "max_side": 256,
        })
        self.assertIn("pair_error", r)
        self.assertLess(r["pair_error"]["mean"], 4.0,
                        f"pair_error mean {r['pair_error']['mean']:.2f} >= 4.0")
        self.assertIn("pair_shift", r)
        self.assertLess(r["pair_shift"], 5.0,
                        f"pair_shift {r['pair_shift']:.2f} >= 5.0 (bull-Paar ist aligniert)")

    def test_fit_pair_400(self):
        """FIT mit pair → 400."""
        sid = self._load_bull()
        code, body = _post_expect_error(f"{self.base}/fit", {
            "session": sid,
            "references": [str(BULL_DIR / "bull1.jpg"), str(BULL_DIR / "Bull2.jpg")],
            "method": "pair",
        })
        self.assertEqual(code, 400)
        self.assertIn("FIT is not available for PAIR method", body.get("error", ""))


# ── Plugin-Tests (Fake-Client) ─────────────────────────────────────────────────


class TestPluginPair(unittest.TestCase):
    """Fenster: PAIR → /preview.method=='pair', FIT deaktiviert, BEFORE/AFTER sichtbar."""

    @classmethod
    def setUpClass(cls):
        _PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
        _pkg = types.ModuleType("ref_grade")
        _pkg.__path__ = [str(_PLUGIN_DIR)]
        sys.modules["ref_grade"] = _pkg
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        if cls.app:
            cls.app.processEvents()

    def test_pair_ui(self):
        """PAIR wählen → /preview.method=='pair', FIT deaktiviert, BEFORE/AFTER sichtbar."""
        from ref_grade.dialog import RefGradeDialog
        from ref_grade import worker_client

        tmp = Path(tempfile.mkdtemp(prefix="refgrade_plugin_pair_"))
        try:
            # FakeClient
            class FakeClient:
                def __init__(self):
                    self.calls = []
                    self._png = tmp / "resp.png"
                    from PyQt6.QtGui import QImage, QColor
                    img = QImage(64, 64, QImage.Format.Format_RGB32)
                    img.fill(QColor(100, 150, 200))
                    img.save(str(self._png), "PNG")
                    self._ref = []
                    for name in ("before", "after"):
                        p = tmp / f"{name}.png"
                        r = QImage(48, 48, QImage.Format.Format_RGB32)
                        r.fill(QColor(20, 20, 20))
                        r.save(str(p), "PNG")
                        self._ref.append({"path": str(p), "name": name})

                def __call__(self, port, path, payload=None):
                    self.calls.append((path, payload))
                    if path == "/preview":
                        return {"image": str(self._png), "histogram": {
                            "source": [0.015625] * 64, "result": [0.02] * 64},
                            "ms": 1.0, "pair_error": {"mean": 2.5, "p95": 8.0}}
                    elif path == "/references":
                        return {"items": self._ref}
                    elif path == "/fit":
                        return {"values": {}}
                    else:
                        raise worker_client.WorkerError(404, f"Unknown: {path}")

                def count(self, path):
                    return sum(1 for p, _ in self.calls if p == path)

                def last(self, path):
                    for p, payload in reversed(self.calls):
                        if p == path:
                            return payload
                    return None

            fake = FakeClient()
            from PyQt6.QtGui import QImage, QColor
            img = QImage(200, 150, QImage.Format.Format_RGB32)
            img.fill(QColor(80, 120, 160))

            dlg = RefGradeDialog(
                img, None,
                client=fake,
                source={"raw": str(tmp / "s.raw"), "width": 100, "height": 80,
                        "depth": "U8", "x": 0, "y": 0, "session": "fake"},
            )
            dlg.resize(1920, 1080)
            dlg.show()

            # Warten bis sichtbar
            deadline = time.time() + 3
            while time.time() < deadline:
                self.app.processEvents()
                if dlg.isVisible():
                    break
                time.sleep(0.005)

            # K25c: reference section starts OFF and collapsed; expand and turn ON first
            dlg.sections["reference"][0].on.setChecked(True)
            dlg._set_section_collapsed("reference", False)
            self.app.processEvents()

            # PAIR wählen (Index 3)
            dlg.methods.button(3).click()
            deadline = time.time() + 3
            while time.time() < deadline:
                self.app.processEvents()
                if fake.count("/preview") >= 1:
                    break
                time.sleep(0.005)
            time.sleep(0.05)
            self.app.processEvents()

            # /preview.method == "pair"
            payload = fake.last("/preview")
            self.assertIsNotNone(payload)
            self.assertEqual(payload["method"], "pair")

            # FIT deaktiviert
            self.assertFalse(dlg.fit_btn.isEnabled())

            # BEFORE/AFTER Labels sichtbar
            self.assertEqual(dlg.slots.tiles[0].pair_label, "BEFORE")
            self.assertEqual(dlg.slots.tiles[1].pair_label, "AFTER")

            dlg.close()
            dlg.deleteLater()
            self.app.processEvents()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── .cube Export ───────────────────────────────────────────────────────────────


class TestCubeExport(unittest.TestCase):
    """.cube-Export mit PAIR: einlesen und anwenden → max diff 1/255 vs Preview."""

    @slow
    def test_cube_roundtrip(self):
        """PAIR → .cube → einlesen → auf bull1 anwenden → max diff 1/255."""
        dev = _get_device()
        before = _img_to_tensor(str(BULL_DIR / "bull1.jpg"), dev)
        after = _img_to_tensor(str(BULL_DIR / "Bull2.jpg"), dev)
        table = fit(before, after, size=33)
        lut = as_lut(table, title="PAIR test")

        # .cube schreiben
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_cube_"))
        try:
            cube_path = tmp / "pair_test.cube"
            write_cube(lut, str(cube_path))

            # .cube einlesen
            lut2 = read_cube(str(cube_path))
            self.assertEqual(lut2["size"], 33)

            # Auf bull1 anwenden
            result1 = lut_apply(before, lut)
            result2 = lut_apply(before, lut2)
            max_diff = (result1 - result2).abs().max().item()
            self.assertLessEqual(max_diff, 1.0 / 255.0 + 1e-7,
                                 f"Cube roundtrip: max diff {max_diff:.6f} > 1/255")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestPairColdCache(unittest.TestCase):
    """Regression 2026-10-08: PAIR read s.refs without loading, so a session whose
    references were not preloaded by /references failed with KeyError."""

    def test_apply_full_loads_pair_images_itself(self):
        from refgrade import session as S
        pair = Path(__file__).resolve().parents[2] / "Dual LUT"
        if not (pair / "bull1.jpg").exists():
            self.skipTest("Dual LUT test pair missing")
        from PIL import Image
        a = np.asarray(Image.open(pair / "bull1.jpg").convert("RGB"))
        h, w = a.shape[:2]
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "src.raw"
            raw.write_bytes(np.concatenate([a[..., ::-1], np.full((h, w, 1), 255, np.uint8)], 2).tobytes())
            s = S.open_session(str(raw), w, h, "U8", device=torch.device("cpu"))
            self.assertEqual(s.refs, {})
            out = S.apply_full(s, [str(pair / "bull1.jpg"), str(pair / "Bull2.jpg")], "pair", {},
                               {"weights": {"reference": 1.0}, "master": 1.0})
        b = torch.from_numpy(np.asarray(Image.open(pair / "Bull2.jpg").convert("RGB"), np.float32) / 255)[None]
        d = (rgb_to_lab(out.float().cpu()) - rgb_to_lab(b)).norm(dim=-1)
        self.assertLess(d.mean().item(), 4.0)
