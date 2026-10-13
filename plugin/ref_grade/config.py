"""Pfade und Einstellungen für REF Grade, ohne krita, ohne Qt."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_HERE: Path = Path(__file__).resolve().parent
# ZIP-Installation: der Kern liegt gebündelt neben dem Plugin (ref_grade/core).
# Entwicklung (Junction auf plugin/ref_grade): Repo-Layout, Kern in <Repo>/core.
_BUNDLED: bool = (_HERE / "core").is_dir()
ROOT: Path = _HERE if _BUNDLED else _HERE.parents[1]
CORE: Path = ROOT / "core"
POOL: Path = ROOT / "bench" / "pool"   # nur im Repo vorhanden; fehlend ist unkritisch
LUTS: Path = (Path(os.environ.get("APPDATA", "")) / "krita" / "luts") if _BUNDLED else ROOT / "luts"
PORT: int = 47123

_SKIP_PATTERNS = ("windowsapps", "pythonmanager")


def python_exe() -> str | None:
    """Interpreter-Programm für den Worker; None wenn nicht gefunden."""
    env = os.environ.get("REFGRADE_PYTHON")
    if env:
        return env
    from . import runtime
    if runtime.is_installed():
        return str(runtime.venv_python())
    for name in ("python", "python3"):
        p = shutil.which(name)
        if p and not any(pat in p.lower() for pat in _SKIP_PATTERNS):
            return p
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        base = Path(local) / "Programs" / "Python"
        for c in sorted(base.glob("Python3*"), reverse=True):
            exe = c / "python.exe"
            if exe.is_file():
                return str(exe)
    return None


def lut_dir(kritarc: Path) -> Path:
    """LUT-Verzeichnis aus kritarc lesen; Fallback %APPDATA%\\krita\\luts."""
    try:
        text = kritarc.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("ResourceDirectory="):
                d = Path(line.split("=", 1)[1].strip())
                return d / "luts"
    except OSError:
        pass
    return Path(os.environ.get("APPDATA", "")) / "krita" / "luts"


def temp_dir() -> Path:
    """Temp-Ordner für den Worker, angelegt."""
    d = Path(os.environ.get("TEMP", "/tmp")) / "ref_grade"
    d.mkdir(parents=True, exist_ok=True)
    return d
