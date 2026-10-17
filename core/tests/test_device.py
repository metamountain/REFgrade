"""Tests für refgrade.device: pick_device (CUDA/ROCm → XPU → CPU), synchronize."""

import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import refgrade.device as device_mod
from refgrade.device import pick_device, synchronize


@contextmanager
def _no_refgrade_device():
    """REFGRADE_DEVICE während des Blocks entfernen, danach wiederherstellen."""
    saved = os.environ.pop("REFGRADE_DEVICE", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["REFGRADE_DEVICE"] = saved


class TestPickDevice(unittest.TestCase):

    def setUp(self):
        # Cache leeren, damit jeder Test mit einer frischen Wahl startet.
        device_mod._cache.clear()

    def tearDown(self):
        # Cache leeren, damit ein unter Mock gewähltes Gerät (z. B. xpu:0) nicht
        # in nachfolgende Tests/Module (z. B. test_lut) hineinläuft.
        device_mod._cache.clear()

    def test_default(self):
        """Ohne Umgebungsvariable: cuda wenn verfügbar, sonst cpu."""
        with _no_refgrade_device():
            dev = pick_device()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(dev.type, expected)

    def test_force_cpu(self):
        """REFGRADE_DEVICE=cpu → cpu."""
        with mock.patch.dict(os.environ, {"REFGRADE_DEVICE": "cpu"}):
            dev = pick_device()
        self.assertEqual(dev.type, "cpu")

    def test_xpu_fallback(self):
        """REFGRADE_DEVICE=xpu ohne XPU → Rückfall (nicht xpu, kein Fehler)."""
        with mock.patch.dict(os.environ, {"REFGRADE_DEVICE": "xpu"}):
            dev = pick_device()
        self.assertNotEqual(dev.type, "xpu")

    def test_xpu_when_available(self):
        """Mock: cuda=False, xpu=True → xpu (Cache ist in setUp geleert)."""
        with _no_refgrade_device():
            fake_xpu = mock.Mock()
            fake_xpu.is_available.return_value = True
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with mock.patch.object(torch, "xpu", fake_xpu, create=True):
                    dev = pick_device()
        self.assertEqual(dev.type, "xpu")

    def test_synchronize_cpu(self):
        """synchronize(cpu) wirft nicht."""
        synchronize(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
