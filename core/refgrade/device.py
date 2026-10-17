"""Zentrale Geräte-Wahl: CUDA/ROCm → XPU → CPU.

CUDA- und ROCm-Builds melden sich beide über torch.cuda (HIP), brauchen also
nichts Eigenes. XPU (Intel) wird nur genutzt, wenn torch.xpu vorhanden und
verfügbar ist. Das Ergebnis wird pro Wert von REFGRADE_DEVICE gecacht.
"""

from __future__ import annotations

import os

import torch

# Cache: Schlüssel = Wert von REFGRADE_DEVICE (None, wenn nicht gesetzt).
_cache: dict[str | None, torch.device] = {}


def _available(dev: str) -> bool:
    """Ist das Gerät auf dieser Maschine verfügbar?"""
    if dev == "cpu":
        return True
    if dev == "cuda":
        return torch.cuda.is_available()
    if dev == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return False


def pick_device() -> torch.device:
    """cuda:0 (auch ROCm) → xpu:0 → cpu. Ergebnis wird gecacht (Modul-Variable).

    Umgebungsvariable REFGRADE_DEVICE (cpu/cuda/xpu) erzwingt das Gerät, wenn
    verfügbar; sonst gilt die normale Reihenfolge. Der Cache wird pro Wert gehalten.
    """
    env = os.environ.get("REFGRADE_DEVICE")
    if env in _cache:
        return _cache[env]

    if env in ("cpu", "cuda", "xpu") and _available(env):
        dev = torch.device("cpu" if env == "cpu" else f"{env}:0")
    else:
        if _available("cuda"):
            dev = torch.device("cuda:0")
        elif _available("xpu"):
            dev = torch.device("xpu:0")
        else:
            dev = torch.device("cpu")

    _cache[env] = dev
    return dev


def synchronize(device: torch.device) -> None:
    """torch.cuda.synchronize() bzw. torch.xpu.synchronize(); bei cpu nichts."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
