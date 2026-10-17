"""Pfade und Einstellungen für REF Grade, ohne krita, ohne Qt."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

_HERE: Path = Path(__file__).resolve().parent
# ZIP-Installation: der Kern liegt gebündelt neben dem Plugin (ref_grade/core).
# Entwicklung (Junction auf plugin/ref_grade): Repo-Layout, Kern in <Repo>/core.
_BUNDLED: bool = (_HERE / "core").is_dir()
ROOT: Path = _HERE if _BUNDLED else _HERE.parents[1]
CORE: Path = ROOT / "core"

# Debug: Pfad-Auflösung beim Modul-Load protokollieren (nur bei Problemen relevant).
try:
    from pathlib import Path as _P
    _dbg = _P(os.environ.get("TEMP", "/tmp")) / "refgrade_debug.txt"
    _dbg.write_text(
        f"HERE={_HERE}\n"
        f"BUNDLED={_BUNDLED}\n"
        f"ROOT={ROOT}\n"
        f"CORE={CORE}\n"
        f"exists={CORE.exists()}\n"
        f"is_dir={CORE.is_dir()}\n"
        f"listdir={sorted(p.name for p in _HERE.iterdir()) if _HERE.is_dir() else 'N/A'}\n",
        encoding="utf-8",
    )
except Exception:
    pass
POOL: Path = ROOT / "bench" / "pool"   # nur im Repo vorhanden; fehlend ist unkritisch
LUTS: Path = (Path(os.environ.get("APPDATA", "")) / "krita" / "luts") if _BUNDLED else ROOT / "luts"
PORT: int = 47123

_SKIP_PATTERNS = ("windowsapps", "pythonmanager")


def settings_path() -> Path:
    """Einstellungsdatei; REFGRADE_CONFIG überschreibt den Pfad (Tests)."""
    env = os.environ.get("REFGRADE_CONFIG")
    if env:
        return Path(env)
    return Path(os.environ.get("APPDATA", "")) / "RefGrade" / "config.json"


def read_settings() -> dict:
    """config.json lesen; fehlt/ungültig → leeres dict."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_settings(data: dict) -> None:
    """config.json schreiben; Ordner wird angelegt."""
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def python_exe() -> str | None:
    """Interpreter-Programm für den Worker; None wenn nicht gefunden."""
    from . import runtime
    # K38c: gespeichertes python_exe zuerst, wenn es den Check besteht.
    stored = read_settings().get("python_exe")
    if stored and runtime.python_ok(stored):
        return stored
    if runtime.is_installed():
        return str(runtime.venv_python())
    env = os.environ.get("REFGRADE_PYTHON")
    if env:
        return env
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
