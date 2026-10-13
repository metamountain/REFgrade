"""Verwaiste RefGrade-Worker beenden (vor jeder Jobvergabe).

Ein Worker ist `python -m worker --port … --tmp …`. Er gilt als verwaist, wenn sein
Elternprozess nicht mehr läuft. Worker, deren Eltern noch leben (Krita, ein laufender Test oder
Bench), bleiben unangetastet — außer mit `--all-but-krita`: dann wird jeder Worker beendet, dessen
Elternprozess nicht krita.exe ist (vor einem neuen Pipeline-Paket läuft sonst nichts Eigenes).

Aufruf: python tools/kill_stale_workers.py [--all-but-krita] [--dry]
Exit 0 immer (ein Hook soll die Pipeline daran nicht scheitern lassen).
"""

from __future__ import annotations

import json
import subprocess
import sys

_PS = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-CimInstance Win32_Process | "
    "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
)


def _procs() -> list[dict]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _PS], capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    data = json.loads(out or "[]")
    return data if isinstance(data, list) else [data]


def is_worker(cmd: str) -> bool:
    c = cmd or ""
    return "python" in c.lower() and " -m worker" in c and "--port" in c


def stale(procs: list[dict], all_but_krita: bool) -> list[dict]:
    by_pid = {p["ProcessId"]: p for p in procs}
    result = []
    for p in procs:
        if not is_worker(p.get("CommandLine") or ""):
            continue
        parent = by_pid.get(p["ParentProcessId"])
        if parent is None:
            result.append(p)
        elif all_but_krita and (parent.get("Name") or "").lower() != "krita.exe":
            result.append(p)
    return result


def main() -> int:
    all_but_krita = "--all-but-krita" in sys.argv
    dry = "--dry" in sys.argv
    victims = stale(_procs(), all_but_krita)
    for p in victims:
        print(f"Worker {p['ProcessId']} beenden: {(p.get('CommandLine') or '')[:140]}")
        if not dry:
            subprocess.run(["taskkill", "/PID", str(p["ProcessId"]), "/T", "/F"], capture_output=True)
    if not victims:
        print("Keine verwaisten Worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
