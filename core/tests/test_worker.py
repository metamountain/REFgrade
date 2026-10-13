"""Tests für core/worker.py: echter Worker auf freiem Port via subprocess."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
from PIL import Image

CORE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CORE_DIR.parent
POOL_DIR = PROJECT_DIR / "bench" / "pool"
PYTHON = sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(url: str, data: dict, timeout: int = 60) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: int = 60) -> dict:
    with urlopen(url, timeout=timeout) as resp:
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


def _get_expect_error(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        code = getattr(e, "code", 500)
        try:
            return code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return code, {"error": str(e)}


class WorkerTest(unittest.TestCase):
    """Echter Worker als Subprozess."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.tmp = tempfile.mkdtemp(prefix="refgrade_test_")
        env = os.environ.copy()
        cls.proc = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(cls.port), "--tmp", cls.tmp],
            cwd=str(CORE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Warten bis /health antwortet (max 20 s)
        cls.base = f"http://127.0.0.1:{cls.port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out, err = cls.proc.communicate()
                raise RuntimeError(f"Worker sofort beendet: {err.decode()}")
            try:
                r = _get(f"{cls.base}/health")
                if r.get("ok"):
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

    def _make_raw(self, width: int, height: int, depth: str = "U16") -> str:
        """Rohdatendatei (B,G,R,A) erzeugen."""
        if depth == "U16":
            arr = np.random.randint(0, 65536, (height, width, 4), dtype=np.uint16)
        else:
            arr = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
        path = os.path.join(self.tmp, f"raw_{width}x{height}_{depth}.raw")
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        return path

    def _load(self, width: int, height: int, depth: str = "U16") -> str:
        raw = self._make_raw(width, height, depth)
        r = _post(f"{self.base}/load", {"raw": raw, "width": width, "height": height, "depth": depth})
        return r["session"]

    def test_01_health(self):
        r = _get(f"{self.base}/health")
        self.assertTrue(r["ok"])
        self.assertIn("version", r)
        self.assertIn(r["device"], ("cuda", "cpu"))

    def test_02_sliders(self):
        r = _get(f"{self.base}/sliders")
        self.assertIn("sliders", r)
        self.assertIn("help", r)
        self.assertGreater(len(r["sliders"]), 10)
        s = r["sliders"][0]
        for key in ("name", "group", "min", "max", "default", "step", "core"):
            self.assertIn(key, s)

    def test_03_full_flow_preview(self):
        """64×48 U16 → /load → /preview (Reinhard, Vorlage aus pool, max_side 64)."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        r = _post(f"{self.base}/preview", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {},
            "max_side": 64,
        })
        self.assertIn("image", r)
        self.assertIn("histogram", r)
        self.assertIn("ms", r)
        # PNG existiert und hat die richtige Größe
        img = Image.open(r["image"])
        self.assertEqual(img.size, (64, 48))
        # Histogramme: 3 × 64 Werte
        for key in ("source", "reference", "result"):
            self.assertEqual(len(r["histogram"][key]), 64)
        # Vectorscope: je 192 Zeilen à 192 Werte
        self.assertIn("vectorscope", r)
        self.assertIn("vectorscope_source", r)
        self.assertEqual(len(r["vectorscope"]), 192)
        self.assertEqual(len(r["vectorscope_source"]), 192)
        self.assertEqual(len(r["vectorscope"][0]), 192)
        self.assertEqual(len(r["vectorscope_source"][0]), 192)
        # K29d: RGB Parade as PNG paths
        self.assertIn("parade", r)
        self.assertIn("r", r["parade"])
        self.assertIn("g", r["parade"])
        self.assertIn("b", r["parade"])
        self.assertTrue(os.path.isfile(r["parade"]["r"]))   # neon BMP since 2026-10-11
        self.assertTrue(os.path.isfile(r["parade"]["g"]))   # neon BMP since 2026-10-11
        self.assertTrue(os.path.isfile(r["parade"]["b"]))   # neon BMP since 2026-10-11

    def test_04_apply_identity(self):
        """/apply mit light_strength=0, color_strength=0 → bytegleich zur Eingabe."""
        width, height = 64, 48
        raw = self._make_raw(width, height, "U16")
        r = _post(f"{self.base}/load", {"raw": raw, "width": width, "height": height, "depth": "U16"})
        sid = r["session"]
        ref_path = str(POOL_DIR / "01.jpg")
        out_path = os.path.join(self.tmp, "apply_identity.raw")
        _post(f"{self.base}/apply", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {"light_strength": 0, "color_strength": 0},
            "out": out_path,
        })
        with open(raw, "rb") as f:
            data_in = f.read()
        with open(out_path, "rb") as f:
            data_out = f.read()
        self.assertEqual(data_in, data_out, "Apply-Identität: Ausgabe nicht bytegleich")

    def test_05_crop(self):
        """/crop: 1200×900, rect=[100,50,64,48], max_side=4096 → PNG 64×48, scale 1.0."""
        width, height = 1200, 900
        # Deterministische Rohdaten
        arr = np.zeros((height, width, 4), dtype=np.uint16)
        arr[:, :, 0] = 1000
        arr[:, :, 1] = 2000
        arr[:, :, 2] = 3000
        arr[:, :, 3] = 65535
        # Einen Bereich mit anderem Wert
        arr[50:98, 100:164, 0] = 5000
        raw = os.path.join(self.tmp, "crop_src.raw")
        with open(raw, "wb") as f:
            f.write(arr.tobytes())

        r = _post(f"{self.base}/load", {"raw": raw, "width": width, "height": height, "depth": "U16"})
        sid = r["session"]
        ref_path = str(POOL_DIR / "01.jpg")

        # Crop mit Identität (light=0, color=0) → pixelgenau
        r = _post(f"{self.base}/crop", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {"light_strength": 0, "color_strength": 0},
            "rect": [100, 50, 64, 48],
            "max_side": 4096,
        })
        self.assertIn("image", r)
        self.assertAlmostEqual(r["scale"], 1.0)
        img = Image.open(r["image"])
        self.assertEqual(img.size, (64, 48))

        # Pixel stimmen mit dem Ausschnitt aus /apply überein
        out_path = os.path.join(self.tmp, "crop_apply.raw")
        _post(f"{self.base}/apply", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {"light_strength": 0, "color_strength": 0},
            "out": out_path,
        })
        apply_arr = np.fromfile(out_path, dtype=np.uint16).reshape(height, width, 4)
        crop_arr = np.array(img)  # uint8
        # Apply-Ausschnitt: B,G,R,A → R,G,B (uint8)
        region = apply_arr[50:98, 100:164, :]
        expected_rgb = region[:, :, [2, 1, 0]].astype(np.float32) / 65535.0 * 255.0
        diff = np.abs(crop_arr.astype(np.float32) - expected_rgb).max()
        self.assertLessEqual(diff, 1.0 / 255.0 * 255.0 + 1.0, f"Max-Abweichung: {diff}")

    def test_05b_crop_outside(self):
        """Rechteck außerhalb → 400."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        code, body = _post_expect_error(f"{self.base}/crop", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {},
            "rect": [200, 200, 10, 10],
            "max_side": 4096,
        })
        self.assertEqual(code, 400)

    def test_06_thumb(self):
        """/thumb: Querformat und Hochformat → beide 256×256."""
        # Querformat
        q = str(POOL_DIR / "01.jpg")
        r = _post(f"{self.base}/thumb", {"path": q})
        self.assertIn("thumb", r)
        img = Image.open(r["thumb"])
        self.assertEqual(img.size, (256, 256))

        # Hochformat (erzeugen)
        tall = os.path.join(self.tmp, "tall.png")
        Image.fromarray(np.random.randint(0, 256, (800, 400, 3), dtype=np.uint8)).save(tall)
        r = _post(f"{self.base}/thumb", {"path": tall})
        img = Image.open(r["thumb"])
        self.assertEqual(img.size, (256, 256))

    def test_07_cube(self):
        """/cube zweimal → zweite Datei endet auf _02.cube."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        cube_dir = os.path.join(self.tmp, "cubes")
        os.makedirs(cube_dir, exist_ok=True)

        r1 = _post(f"{self.base}/cube", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": {}, "dir": cube_dir,
        })
        self.assertTrue(os.path.isfile(r1["path"]))

        r2 = _post(f"{self.base}/cube", {
            "session": sid, "references": [ref_path],
            "method": "reinhard", "values": {}, "dir": cube_dir,
        })
        self.assertTrue(os.path.isfile(r2["path"]))
        self.assertTrue(r2["path"].endswith("_02.cube"), f"Erwartet _02.cube, erhalten: {r2['path']}")

    def test_08_errors(self):
        """409, 400, 404."""
        # 409: erfundene Sitzung
        ref_path = str(POOL_DIR / "01.jpg")
        code, _ = _post_expect_error(f"{self.base}/preview", {
            "session": "nonexistent",
            "references": [ref_path],
            "method": "reinhard",
            "values": {},
            "max_side": 64,
        })
        self.assertEqual(code, 409)

        # 400: /preview ohne method
        sid = self._load(64, 48, "U16")
        code, _ = _post_expect_error(f"{self.base}/preview", {
            "session": sid,
            "references": [ref_path],
            "values": {},
            "max_side": 64,
        })
        self.assertEqual(code, 400)

        # 404: unbekannter Pfad
        code, _ = _get_expect_error(f"{self.base}/nope")
        self.assertEqual(code, 404)

    def test_09_concurrency(self):
        """REFGRADE_TEST_SLOW_RANK=2 → /preview wartet nicht auf ein laufendes /references (relativ zu einer Einzelmessung)."""
        import threading

        port2 = _free_port()
        env = os.environ.copy()
        env["REFGRADE_TEST_SLOW_RANK"] = "2"
        proc2 = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(port2), "--tmp", self.tmp],
            cwd=str(CORE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        base2 = f"http://127.0.0.1:{port2}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if proc2.poll() is not None:
                out, err = proc2.communicate()
                raise RuntimeError(f"Worker2 sofort beendet: {err.decode()}")
            try:
                if _get(f"{base2}/health").get("ok"):
                    break
            except Exception:
                time.sleep(0.2)
        else:
            proc2.kill()
            raise RuntimeError("Worker2 nicht erreichbar")

        try:
            raw = self._make_raw(64, 48, "U16")
            r = _post(f"{base2}/load", {"raw": raw, "width": 64, "height": 48, "depth": "U16"})
            sid2 = r["session"]
            ref_path = str(POOL_DIR / "01.jpg")

            # Basis-Vorschau messen (ohne /references im Hintergrund)
            t0 = time.perf_counter()
            _post(f"{base2}/preview", {
                "session": sid2, "references": [ref_path],
                "method": "reinhard", "values": {"exposure": 0.2}, "max_side": 64,
            }, timeout=60)
            base_ms = (time.perf_counter() - t0) * 1000.0

            results: dict = {}
            errors: list = []

            def do_refs():
                try:
                    results["refs"] = _post(f"{base2}/references", {
                        "session": sid2, "pool": str(POOL_DIR), "k": 3,
                    }, timeout=60)
                except Exception as e:
                    errors.append(f"refs: {e}")

            def do_preview():
                try:
                    t0 = time.perf_counter()
                    results["preview"] = _post(f"{base2}/preview", {
                        "session": sid2, "references": [ref_path],
                        "method": "reinhard", "values": {"exposure": 0.1}, "max_side": 64,
                    }, timeout=60)
                    results["preview_ms"] = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    errors.append(f"preview: {e}")

            t_refs = threading.Thread(target=do_refs)
            t_prev = threading.Thread(target=do_preview)
            t_refs.start()
            time.sleep(0.5)  # /references läuft schon (2 s Sleep)
            t_prev.start()
            t_prev.join(timeout=30)
            t_refs.join(timeout=30)

            if "preview" not in results:
                raise AssertionError(f"Preview fehlgeschlagen: {errors}")
            # Blockiert würde die Vorschau ≈ 1,5 s länger dauern
            # (references schläft 2 s, startet 0,5 s vorher).
            self.assertLess(results["preview_ms"], base_ms + 1000.0,
                            f"Preview {results['preview_ms']:.0f} ms, Basis {base_ms:.0f} ms — wartet auf /references?")
        finally:
            try:
                _post(f"{base2}/shutdown", {}, timeout=5)
            except Exception:
                pass
            try:
                proc2.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc2.kill()

    def test_10_shutdown(self):
        """/shutdown → Prozess endet innerhalb von 5 s mit Exit-Code 0."""
        # Eigener Worker
        port3 = _free_port()
        proc3 = subprocess.Popen(
            [PYTHON, "-m", "worker", "--port", str(port3), "--tmp", self.tmp],
            cwd=str(CORE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        base3 = f"http://127.0.0.1:{port3}"
        deadline = time.time() + 20
        while time.time() < deadline:
            if proc3.poll() is not None:
                raise RuntimeError("Worker3 sofort beendet")
            try:
                if _get(f"{base3}/health").get("ok"):
                    break
            except Exception:
                time.sleep(0.2)
        else:
            proc3.kill()
            raise RuntimeError("Worker3 nicht erreichbar")

        _post(f"{base3}/shutdown", {})
        try:
            rc = proc3.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc3.kill()
            raise AssertionError("Worker3 beendet sich nicht innerhalb von 5 s")
        self.assertEqual(rc, 0)

    def test_11_lut_not_found(self):
        """mix.lut zeigt auf eine nicht vorhandene Datei → 400."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        fake_lut = os.path.join(self.tmp, "nonexistent.cube")
        code, body = _post_expect_error(f"{self.base}/preview", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {},
            "max_side": 64,
            "mix": {"weights": {"lut": 1.0}, "master": 1.0, "solo": None,
                    "lut": fake_lut},
        })
        self.assertEqual(code, 400)
        self.assertIn("LUT not readable", body.get("error", ""))

    def test_12_curves_invalid(self):
        """mix.curves mit ungültigem Format (x nicht steigend) → 400."""
        sid = self._load(64, 48, "U16")
        ref_path = str(POOL_DIR / "01.jpg")
        code, body = _post_expect_error(f"{self.base}/preview", {
            "session": sid,
            "references": [ref_path],
            "method": "reinhard",
            "values": {},
            "max_side": 64,
            "mix": {"weights": {"curves": 1.0}, "master": 1.0, "solo": None,
                    "curves": {"rgb": [[0.5, 0], [0.2, 1]]}},
        })
        self.assertEqual(code, 400)
        self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
