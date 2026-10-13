"""Bench unter drei GPU-Zuständen laufen lassen: BeeLlama normal, komprimiert, pausiert.

Aufruf (aus core\\, NICHT aus einer Qwen-Sitzung — Pausieren beendet den Server):
    python tools/gpu_states.py "<bench-befehl>" [--states normal,compressed,paused]

Je Zustand: BeeLlama passend (neu) starten bzw. beenden, VRAM notieren, Bench-Befehl mit
Umgebungsvariable RG_GPU_STATE=<zustand> ausführen. Am Ende läuft BeeLlama wieder normal.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

BEE = r"D:\BeeLlama\start-beellama.cmd"
COMPRESSED_CTX = "65536"


def vram() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return out


def server_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3) as r:
            return b"ok" in r.read()
    except OSError:
        return False


def stop_server() -> None:
    subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)
    for _ in range(60):
        if not server_up():
            break
        time.sleep(1)
    time.sleep(5)  # VRAM freigeben lassen


def start_server(ctx: str | None) -> None:
    env = dict(os.environ)
    env.pop("BEE_CTX", None)
    if ctx:
        env["BEE_CTX"] = ctx
    # Menü fällt nach 10 s auf den Standard (S) zurück; "1" = IQ4 kvarn6 1 Slot direkt.
    # Titel in Anführungszeichen, sonst hält "start" ihn für das Programm (2026-10-11)
    subprocess.Popen('start "BeeLlama" /D ' + os.path.dirname(BEE) + f' "{BEE}" 1', shell=True, env=env)
    for _ in range(300):
        if server_up():
            return
        time.sleep(2)
    raise RuntimeError("BeeLlama kam nicht hoch")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    states = ["normal", "compressed", "paused"]
    if "--states" in sys.argv:
        states = sys.argv[sys.argv.index("--states") + 1].split(",")
    rc = 0
    try:
        for st in states:
            stop_server()
            if st == "normal":
                start_server(None)
            elif st == "compressed":
                start_server(COMPRESSED_CTX)
            time.sleep(5)
            print(f"== {st}: VRAM {vram()}", flush=True)
            env = dict(os.environ, RG_GPU_STATE=st, RG_GPU_VRAM=vram())
            r = subprocess.run(cmd, shell=True, env=env)
            rc = rc or r.returncode
    finally:
        if not server_up():
            start_server(None)
        print(f"== Ende, BeeLlama normal: VRAM {vram()}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
