"""Tests für den ONNX-LUT-Prototyp (refgrade.onnx_lut).

Zwei Ebenen:
- Smoke-Test: Identity-Modell (reiner ONNX-Runtime-Overhead).
- Echt: trilineare LUT-Interpolation, Korrektheit vs. Torch-Referenz
  (``refgrade.lut.apply``) + Benchmark.

Runner: ``python -m pytest core/tests/test_onnx_lut.py -v -s``
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def lut_model(tmp_path_factory):
    """Einmaliges, validiertes ONNX-LUT-Modell (N=33) im Temp-Verzeichnis."""
    import onnx
    from refgrade.onnx_lut import create_lut_apply_model

    model = create_lut_apply_model(33)
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("onnx") / "lut_apply.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture(scope="module")
def identity_model(tmp_path_factory):
    """Einmaliges Identity-Smoke-Test-Modell im Temp-Verzeichnis."""
    import onnx
    from refgrade.onnx_lut import create_identity_model

    model = create_identity_model()
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("onnx") / "lut_identity.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.mark.skipif(not _HAS_ONNX, reason="onnx nicht installiert")
class TestONNXLutModel:

    def test_model_creation(self):
        from refgrade.onnx_lut import create_lut_apply_model

        model = create_lut_apply_model(33)
        assert model is not None
        assert len(model.graph.node) > 0

    def test_identity_model_creation(self):
        from refgrade.onnx_lut import create_identity_model

        model = create_identity_model()
        assert model is not None

    def test_model_save_load(self, lut_model):
        import onnx

        loaded = onnx.load(lut_model)
        onnx.checker.check_model(loaded)
        # Zwei Inputs (image, lut), ein Output (result)
        assert len(loaded.graph.input) == 2
        assert len(loaded.graph.output) == 1


@pytest.mark.skipif(not _HAS_ONNX or not _HAS_ORT, reason="onnx/onnxruntime fehlt")
class TestONNXLutInference:

    def test_identity_smoke(self, identity_model):
        """Identity-Modell: result == image (nur Overhead)."""
        from refgrade.onnx_lut import run_identity_inference

        img = np.random.default_rng(0).random((1, 3, 64, 64)).astype(np.float32)
        result = run_identity_inference(img, model_path=identity_model)
        assert result.shape == img.shape
        np.testing.assert_allclose(result, img, atol=1e-6)

    def test_real_lut_shape(self, lut_model):
        from refgrade.onnx_lut import run_onnx_inference, make_test_lut

        img = np.random.default_rng(1).random((1, 3, 48, 64)).astype(np.float32)
        lut = make_test_lut(33)
        result = run_onnx_inference(img, lut, model_path=lut_model)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        # Output bleibt im 0..1-Bereich (LUT ist geclamped)
        assert result.min() >= 0.0 and result.max() <= 1.0

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch fehlt")
    def test_correctness_vs_torch(self, lut_model):
        """ONNX-Trilinear vs. Torch grid_sample: max < 1e-5."""
        from refgrade.onnx_lut import run_onnx_inference, make_test_lut, torch_lut_apply

        img = np.random.default_rng(2).random((1, 3, 32, 40)).astype(np.float32)
        lut = make_test_lut(33)
        out_onnx = run_onnx_inference(img, lut, model_path=lut_model)
        out_torch = torch_lut_apply(img, lut)
        max_err = np.abs(out_onnx - out_torch).max()
        assert max_err < 1e-5, f"ONNX vs Torch max err {max_err:.2e}"

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch fehlt")
    def test_benchmark(self, lut_model, capsys):
        """Benchmark: ONNX (CPU) vs. Torch (CPU) vs. Torch (CUDA), 1024x768."""
        import time
        import torch

        from refgrade.onnx_lut import run_onnx_inference, make_test_lut, torch_lut_apply

        h, w = 768, 1024
        img = np.random.default_rng(3).random((1, 3, h, w)).astype(np.float32)
        lut = make_test_lut(33)
        n = 5

        # Warmup
        run_onnx_inference(img, lut, model_path=lut_model)
        torch_lut_apply(img, lut)

        # ONNX (CPU)
        t0 = time.perf_counter()
        for _ in range(n):
            run_onnx_inference(img, lut, model_path=lut_model)
        onnx_ms = (time.perf_counter() - t0) / n * 1000.0

        # Torch (CPU)
        t0 = time.perf_counter()
        for _ in range(n):
            torch_lut_apply(img, lut)
        torch_cpu_ms = (time.perf_counter() - t0) / n * 1000.0

        # Torch (CUDA), falls verfügbar
        torch_cuda_ms = float("nan")
        if torch.cuda.is_available():
            img_t = torch.from_numpy(img).permute(0, 2, 3, 1).to("cuda:0")
            table = torch.from_numpy(lut).squeeze(0).permute(1, 2, 3, 0).to("cuda:0")
            from refgrade.lut import apply

            lut_d = {"size": 33, "table": table, "title": "bench"}
            apply(img_t, lut_d)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                apply(img_t, lut_d)
            torch.cuda.synchronize()
            torch_cuda_ms = (time.perf_counter() - t0) / n * 1000.0

        print(f"\n[ONNX Benchmark] {w}x{h}, {n} Runs, LUT 33^3:")
        print(f"  ONNX (CPU):        {onnx_ms:8.2f} ms")
        print(f"  Torch (CPU):       {torch_cpu_ms:8.2f} ms")
        print(f"  Torch (CUDA):      {torch_cuda_ms:8.2f} ms")
        print(f"  ONNX/Torch-CPU:    {onnx_ms / torch_cpu_ms:8.2f}x")

        # ONNX muss laufen und eine plausible Zeit liefern
        assert onnx_ms > 0
        assert torch_cpu_ms > 0
