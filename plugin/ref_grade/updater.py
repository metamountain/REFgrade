"""GitHub updater: check for newer commit, download ZIP, replace via batch script.

Windows locks running .pyc/.py files, so we can't overwrite in-place.
Instead: download to temp, write a .bat that replaces the folder after
Krita exits, launch the .bat, then close Krita.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from . import config

_REPO = "metamountain/REFgrade"
_API_LATEST = f"https://api.github.com/repos/{_REPO}/commits/main"
_ZIP_URL = f"https://github.com/{_REPO}/archive/refs/heads/main.zip"
_TIMEOUT = 30


def _version_file() -> Path:
    return config._HERE / "installed_version.json"


def installed_commit() -> str | None:
    try:
        data = json.loads(_version_file().read_text(encoding="utf-8"))
        return data.get("commit")
    except (OSError, json.JSONDecodeError):
        return None


def _save_commit(sha: str) -> None:
    _version_file().write_text(
        json.dumps({"commit": sha, "repo": _REPO}, indent=2), encoding="utf-8"
    )


def _fetch_latest() -> tuple[str, str] | None:
    """Return (sha, message) of latest commit on main, or None on error."""
    try:
        req = urllib.request.Request(
            _API_LATEST,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "RefGrade"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
        return data["sha"], data["commit"]["message"].split("\n")[0][:80]
    except Exception:
        return None


def _plugin_install_dir() -> Path:
    """Where the plugin lives in Krita's pykrita directory."""
    # In a junction (dev) setup, _HERE IS the dev dir — don't overwrite that.
    # In a ZIP install, _HERE is under %APPDATA%\krita\pykrita\ref_grade.
    if os.path.islink(config._HERE) or os.path.isjunction(config._HERE):
        return config._HERE  # junction: just overwrite (dev mode)
    # ZIP install: the parent of ref_grade is pykrita/
    return config._HERE


def _download_zip(dest: Path) -> None:
    req = urllib.request.Request(_ZIP_URL, headers={"User-Agent": "RefGrade"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):      # urlretrieve had no timeout: a stalled
            out.write(chunk)                    # download froze Krita for good


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract plugin/ref_grade/ and core/ from the repo ZIP into dest."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = "REFgrade-main/"
        for info in zf.infolist():
            if not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            if "__pycache__" in rel:
                continue
            if rel.startswith("plugin/ref_grade/"):
                dst = dest / rel[len("plugin/ref_grade/"):]
            elif rel == "core/worker.py" or rel.startswith("core/refgrade/"):
                # the bundled core lives in ref_grade/core/ (config._BUNDLED); only what the
                # worker needs, not core/tests, bench, reports (debug round 2026-10-17)
                dst = dest / rel
            else:
                continue
            if info.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dst, "wb") as out:
                    out.write(src.read())


def _write_batch(tmp_dir: Path, new_dir: Path, target: Path, krita_exe: str | None) -> Path:
    """Write a .bat that waits, replaces the plugin dir, optionally restarts Krita."""
    bat = tmp_dir / "refgrade_update.bat"
    lines = [
        "@echo off",
        "title RefGrade Update",
        "echo Waiting for Krita to close...",
        ":waitloop",
        "tasklist /FI \"IMAGENAME eq krita.exe\" 2>NUL | find /I /N \"krita.exe\" >NUL",
        "if not errorlevel 1 (",
        "    timeout /t 2 /nobreak >NUL",
        "    goto waitloop",
        ")",
        "echo Stopping the RefGrade worker...",
        # wmic is gone from Windows 11 24H2+; CIM finds the worker by its command line
        "powershell -NoProfile -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\""
        " | Where-Object { $_.CommandLine -match ' -m worker' -and $_.CommandLine -match '--port' }"
        " | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }\"",
        "timeout /t 1 /nobreak >NUL",
        "echo Replacing plugin files...",
        f"rmdir /S /Q \"{target}\"",
        f"xcopy /E /I /Y \"{new_dir}\" \"{target}\"",
    ]
    if krita_exe:
        lines.append(f'start "" "{krita_exe}"')
    lines += [
        "echo Update complete.",
        "timeout /t 3 /nobreak >NUL",
    ]
    # CRLF: cmd.exe misreads labels/goto in LF batch files (the wait loop never waited)
    bat.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return bat


def _find_krita() -> str | None:
    """Try to locate krita.exe for restart."""
    for base in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Krita",
        Path(os.environ.get("PROGRAMFILES", "")) / "Krita",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Krita",
    ):
        if not base:
            continue
        for exe in base.glob("**/krita.exe"):
            return str(exe)
    return None


def check_and_update(progress_cb=None) -> tuple[bool, str]:
    """Check GitHub and update if a newer commit exists.

    Returns (success, message).
    """
    def _p(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if not config._BUNDLED or any((d / ".git").exists() for d in config._HERE.parents):
        # standalone / git checkout: _HERE is the repo's plugin folder — replacing it would
        # delete local work and leave core/ behind (debug round 2026-10-17)
        return False, ("Update only for the Krita plugin ZIP install. Here: git pull, or "
                       "download the new release ZIP from github.com/" + _REPO)

    _p("Prüfe GitHub…")
    latest = _fetch_latest()
    if latest is None:
        return False, "GitHub nicht erreichbar (Netzwerk oder Rate-Limit)."

    latest_sha, latest_msg = latest
    current = installed_commit()

    if current is None:
        # First run: just record, no download
        _save_commit(latest_sha)
        return True, f"Version notiert: {latest_sha[:8]} — {latest_msg}"

    if current == latest_sha:
        return True, f"Bereits die neueste Version ({latest_sha[:8]}) installiert."

    # Newer version available
    _p(f"Neuere Version gefunden: {latest_sha[:8]}")
    _p("Update wird heruntergeladen…")

    tmp_dir = Path(tempfile.mkdtemp(prefix="refgrade_upd_"))
    zip_path = tmp_dir / "repo.zip"
    new_dir = tmp_dir / "new"
    new_dir.mkdir()

    try:
        _download_zip(zip_path)

        _p("Entpacke…")
        _extract_zip(zip_path, new_dir)
        # the batch replaces the whole folder: the version marker must be in the new one
        (new_dir / "installed_version.json").write_text(
            json.dumps({"commit": latest_sha, "repo": _REPO}, indent=2), encoding="utf-8")

        target = _plugin_install_dir()
        krita_exe = _find_krita()
        bat = _write_batch(tmp_dir, new_dir, target, krita_exe)

        _p("Arbeit speichern und Krita schließen!")
        # Give the user a moment to read the message
        import time
        time.sleep(2)

        # Launch the batch in a new window (survives Krita exit)
        subprocess.Popen(
            [str(bat)],
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010),
        )
        return True, (f"Update {latest_sha[:8]} bereit. Bitte Krita jetzt schließen — "
                      "die Dateien werden danach ersetzt und Krita neu gestartet.")
    except Exception as e:
        return False, f"Update fehlgeschlagen: {e}"
