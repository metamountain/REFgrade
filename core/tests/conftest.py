"""pytest: same isolation as run_tests.py — never touch the user's QSettings or config.json."""

import os

try:                      # torch before PyQt6, else WinError 1114 when a later module imports torch
    import torch  # noqa: F401
except ImportError:
    pass
from pathlib import Path

os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
_cfg = Path(__file__).resolve().parents[2] / "tmp" / f"test_config_{os.getpid()}.json"
_cfg.parent.mkdir(exist_ok=True)
_cfg.unlink(missing_ok=True)
os.environ.setdefault("REFGRADE_CONFIG", str(_cfg))
