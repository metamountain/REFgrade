"""Version string so the installed state is visible."""

from __future__ import annotations

import time
from pathlib import Path

VERSION = "0.4.0"   # bump on every user-visible change


def build_stamp() -> str:
    """Time of the newest source file — changes whenever any code changes, no manual bump needed."""
    root = Path(__file__).resolve().parents[1]
    newest = max(p.stat().st_mtime for ext in ("*.py", "*.js") for p in root.rglob(ext)
                 if "__pycache__" not in p.parts and "tests" not in p.parts)
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))


def label() -> str:
    return f"v{VERSION} ({build_stamp()})"
