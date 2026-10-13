"""Laufzeit-Installer: uv + Python 3.12 + torch, ohne Qt, ohne krita."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

from . import config
from .worker_process import clean_env

UV_VERSION = "0.13.0"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.14.1"
PACKAGES = ("numpy", "kornia", "pillow")
TORCH_INDEX = "https://download.pytorch.org/whl/{flavor}"
CHECK_CODE = "import torch, numpy, kornia, PIL; print(torch.__version__, torch.cuda.is_available())"


def runtime_dir() -> Path:
    env = os.environ.get("REFGRADE_RUNTIME")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", "")) / "RefGrade" / "runtime"


def uv_exe() -> Path:
    return runtime_dir() / "uv" / "uv.exe"


def venv_dir() -> Path:
    return runtime_dir() / "venv"


def venv_python() -> Path:
    return venv_dir() / "Scripts" / "python.exe"


def marker_path() -> Path:
    return runtime_dir() / "runtime.json"


def is_installed() -> bool:
    if not venv_python().is_file():
        return False
    try:
        data = json.loads(marker_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("torch") == TORCH_VERSION and data.get("packages") == list(PACKAGES)


def parse_nvidia_smi(text: str) -> dict:
    driver = None
    m = re.search(r"NVIDIA-SMI\s+([\d.]+)", text)
    if m:
        driver = m.group(1)
    cuda = None
    m = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", text)
    if m:
        cuda = (int(m.group(1)), int(m.group(2)))
    return {"driver": driver, "cuda": cuda}


def parse_caps(text: str) -> list[float]:
    caps: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        try:
            caps.append(float(line))
        except ValueError:
            continue
    return caps


def choose_flavor(gpu: bool, cuda, caps) -> tuple[str, str]:
    if not gpu:
        return ("cpu", "No NVIDIA GPU found - installing the CPU version (slower).")
    if cuda is None:
        return ("cu130", "")
    if cuda >= (13, 0):
        return ("cu130", "")
    if cuda >= (12, 6):
        if caps and any(c >= 12.0 for c in caps):
            return ("cpu",
                    "Your GPU needs NVIDIA driver 580 or newer (CUDA 13). "
                    "Update the driver and reinstall for GPU speed - installing the CPU version now.")
        return ("cu126",
                "Older NVIDIA driver (CUDA 12.x) - using the cu126 build. "
                "Updating the driver is recommended.")
    return ("cpu",
            "NVIDIA driver too old for the GPU build - installing the CPU version. "
            "Update the driver and reinstall for GPU speed.")


def detect(run=subprocess.run) -> dict:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    no_gpu = {"gpu_name": None, "flavor": "cpu",
              "message": "No NVIDIA GPU found - installing the CPU version (slower)."}
    try:
        r = run(["nvidia-smi"], timeout=10, capture_output=True, text=True, creationflags=flags)
        if r.returncode != 0:
            return no_gpu
        info = parse_nvidia_smi(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return no_gpu

    try:
        r2 = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                 timeout=10, capture_output=True, text=True, creationflags=flags)
        if r2.returncode != 0:
            return no_gpu
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return no_gpu

    lines = r2.stdout.strip().splitlines()
    gpu_name: str | None = None
    caps: list[float] = []
    if lines:
        parts = lines[0].split(",")
        if parts:
            gpu_name = parts[0].strip()
        cap_lines = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                cap_lines.append(parts[1].strip())
        caps = parse_caps("\n".join(cap_lines))

    flavor, message = choose_flavor(True, info["cuda"], caps)
    return {"gpu_name": gpu_name, "flavor": flavor, "message": message}


def install_env() -> dict:
    env = clean_env()
    env["UV_PYTHON_INSTALL_DIR"] = str(runtime_dir() / "python")
    env["UV_CACHE_DIR"] = str(runtime_dir() / "cache")
    env["UV_LINK_MODE"] = "copy"
    return env


def install_commands(flavor: str) -> list[list[str]]:
    uv = str(uv_exe())
    vp = str(venv_python())
    return [
        [uv, "venv", "--python", PYTHON_VERSION, "--managed-python", "--clear", str(venv_dir())],
        [uv, "pip", "install", "--python", vp, f"torch=={TORCH_VERSION}",
         "--index-url", TORCH_INDEX.format(flavor=flavor)],
        [uv, "pip", "install", "--python", vp, *PACKAGES],
    ]


class InstallError(RuntimeError):
    pass


class Cancelled(InstallError):
    pass


def install(flavor, log, cancelled=lambda: False, popen=subprocess.Popen, fetch=urlopen) -> str:
    rd = runtime_dir()
    rd.mkdir(parents=True, exist_ok=True)

    log_path = config.temp_dir() / "install.log"
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        return _install_inner(flavor, log, cancelled, popen, fetch, log_file, log_path)
    finally:
        log_file.close()


def _install_inner(flavor, log, cancelled, popen, fetch, log_file, log_path) -> str:
    rd = runtime_dir()

    def _log(line: str) -> None:
        log(line)
        log_file.write(line + "\n")
        log_file.flush()

    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # 1. Download uv if missing
    if not uv_exe().is_file():
        zip_path = rd / "uv.zip"
        _log(f"Downloading {UV_URL}")
        resp = fetch(UV_URL, timeout=60)
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -10
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct >= last_pct + 10:
                        last_pct = pct
                        _log(f"Downloading uv: {pct} %")
                if cancelled():
                    resp.close()
                    raise Cancelled("Installation cancelled during download.")
        resp.close()

        uv_exe().parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("uv.exe", str(uv_exe().parent))
        zip_path.unlink(missing_ok=True)
        _log("uv downloaded and extracted.")

    # 2. Delete marker
    marker_path().unlink(missing_ok=True)

    # 3. Run install commands
    env = install_env()
    for i, cmd in enumerate(install_commands(flavor), 1):
        _log(f"Step {i}: {' '.join(cmd)}")
        proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, env=env, creationflags=flags)
        for line in proc.stdout:
            _log(line.rstrip())
            if cancelled():
                proc.terminate()
                raise Cancelled("Installation cancelled.")
        proc.wait()
        if proc.returncode != 0:
            raise InstallError(f"Step {i} failed (exit {proc.returncode}). Log: {log_path}")

    # 4. Check
    _log("Running check...")
    check_cmd = [str(venv_python()), "-E", "-c", CHECK_CODE]
    proc = popen(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                 text=True, env=env, creationflags=flags)
    check_output = ""
    for line in proc.stdout:
        check_output = line.rstrip()
        _log(check_output)
    proc.wait()
    if proc.returncode != 0:
        raise InstallError(f"Check failed (exit {proc.returncode}). Log: {log_path}")

    # 5. Cleanup + write marker
    shutil.rmtree(rd / "cache", ignore_errors=True)
    marker = {
        "flavor": flavor,
        "torch": TORCH_VERSION,
        "packages": list(PACKAGES),
        "uv": UV_VERSION,
        "check": check_output,
    }
    marker_path().write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return check_output


_ok_cache: dict[str, bool] = {}


def python_ok(exe: str, popen=subprocess.run) -> bool:
    if exe in _ok_cache:
        return _ok_cache[exe]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = popen([exe, "-E", "-c", CHECK_CODE], timeout=120,
                  capture_output=True, text=True,
                  env=clean_env(), creationflags=flags)
        result = (r.returncode == 0)
    except (OSError, subprocess.TimeoutExpired):
        result = False
    _ok_cache[exe] = result
    return result


def reset_cache() -> None:
    _ok_cache.clear()
