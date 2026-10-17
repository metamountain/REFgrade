"""Worker starten, finden, stoppen – ohne krita, ohne Qt."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen, Request

from . import config


def _health(port: int) -> bool:
    """True wenn GET /health antwortet."""
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def clean_env(env: dict | None = None) -> dict:
    """The environment without any PYTHON* variable (those belong to Krita's interpreter)."""
    return {k: v for k, v in (os.environ if env is None else env).items()
            if not k.upper().startswith("PYTHON")}


def ensure(port: int, python: str, core: Path, timeout: float = 30.0) -> str:
    """Worker sicherstellen; 'running' oder 'started' zurückgeben."""
    if _health(port):
        return "running"

    if not python:
        raise RuntimeError("No Python with torch found — run the setup first.")
    tmp = config.temp_dir()
    log_path = tmp / "worker.log"
    log_file = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            # -E + clean env: Krita sets PYTHONHOME/PYTHONPATH for its own Python 3.13; inherited,
            # they make the system Python load Krita's stdlib and die ("bad magic number").
            [python, "-E", "-m", "worker", "--port", str(port)],
            cwd=str(core),
            env=clean_env(),
            stdout=log_file,
            stderr=log_file,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as e:
        # missing python.exe / core folder: callers catch RuntimeError only (an OSError
        # escaping their Qt slot ends Krita) — debug round 2026-10-17
        log_file.close()
        raise RuntimeError(f"Worker konnte nicht starten ({python}, {core}): {e}") from e

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health(port):
            log_file.close()
            return "started"
        if proc.poll() is not None:
            log_file.close()
            tail = _tail(log_path, 20)
            raise RuntimeError(f"Worker beendet (exit {proc.returncode}):\n{tail}")
        time.sleep(0.25)

    log_file.close()
    tail = _tail(log_path, 20)
    raise RuntimeError(f"Worker nicht innerhalb von {timeout:.0f}s erreichbar:\n{tail}")


def stop(port: int) -> None:
    """POST /shutdown; Fehler ignorieren."""
    try:
        req = Request(
            f"http://127.0.0.1:{port}/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=5)
    except Exception:
        pass


def _tail(path: Path, n: int) -> str:
    """Letzte n Zeilen einer Datei."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(kein Log)"
