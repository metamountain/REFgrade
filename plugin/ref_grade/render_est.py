"""Render time estimation: linear model ms = a + b·MP, learned per operation.

No Qt import — pure arithmetic.  The plugin reads seed values from render_seed.json
and updates them via QSettings after each completed worker call.
"""

from __future__ import annotations

import json
from pathlib import Path

_SEED_PATH = Path(__file__).parent / "render_seed.json"

# Default seed values (ms) if the file is missing or incomplete.
_DEFAULTS: dict[str, dict[str, float]] = {
    "preview": {"a": 100.0, "b": 50.0},
    "apply": {"a": 150.0, "b": 80.0},
    "cube": {"a": 50.0, "b": 0.0},
}

_ALPHA = 0.3  # learning rate for the sliding average on b


def load_seed() -> dict[str, dict[str, float]]:
    """Read render_seed.json; fall back to defaults for missing keys."""
    data: dict[str, dict[str, float]] = {k: dict(v) for k, v in _DEFAULTS.items()}
    if _SEED_PATH.is_file():
        try:
            raw = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
            for op in data:
                if op in raw:
                    data[op]["a"] = float(raw[op].get("a", data[op]["a"]))
                    data[op]["b"] = float(raw[op].get("b", data[op]["b"]))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return data


def est(op: str, mp: float, a: float, b: float) -> float:
    """Estimated render time in ms for `op` at `mp` megapixels."""
    return a + b * mp


def update_b(b_old: float, measured_ms: float, mp: float) -> float:
    """Sliding average: b_new = 0.7·b_old + 0.3·(measured_ms / mp)."""
    if mp <= 0:
        return b_old
    b_measured = measured_ms / mp
    return (1.0 - _ALPHA) * b_old + _ALPHA * b_measured


def progress(elapsed_ms: float, estimated_ms: float) -> float:
    """Progress fraction 0..0.95 (capped until the result arrives)."""
    if estimated_ms <= 0:
        return 0.95
    return min(0.95, elapsed_ms / estimated_ms)
