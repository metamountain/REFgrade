"""Laufzeit-Installer: uv + Python 3.12 + torch, ohne Qt, ohne krita."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
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


def check_code(flavor: str) -> str:
    """Check-Code für einen Flavor; GPU-Flavors beenden sich mit Exit 3, wenn die GPU fehlt."""
    if flavor == "cpu":
        return CHECK_CODE
    if flavor == "xpu":
        return (CHECK_CODE
                + '; import sys; sys.exit(0 if hasattr(torch, "xpu")'
                + " and torch.xpu.is_available() else 3)")
    return CHECK_CODE + "; import sys; sys.exit(0 if torch.cuda.is_available() else 3)"

FLAVORS: dict[str, dict] = {
    "cu130": {"torch": "2.14.1", "label": "NVIDIA CUDA 13", "size_mb": 3000},
    "cu126": {"torch": "2.14.1", "label": "NVIDIA CUDA 12", "size_mb": 2800},
    "rocm": {"torch": "2.9.1", "label": "AMD ROCm (preview)", "size_mb": 2100},
    "xpu": {"torch": "2.14.1", "label": "Intel Arc (XPU)", "size_mb": 1500},
    "cpu": {"torch": "2.14.1", "label": "CPU only", "size_mb": 250},
}

_ROCM_BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
ROCM_SDK = [
    _ROCM_BASE + "rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    _ROCM_BASE + "rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    _ROCM_BASE + "rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    _ROCM_BASE + "rocm-7.2.1.tar.gz",
]
ROCM_TORCH = [
    _ROCM_BASE + "torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
]

_AMD_RE = re.compile(
    r"Radeon RX (7|9)\d{3}|Radeon (AI )?PRO (W7\d{3}|R9\d{3})|Radeon 8\d{2}0S|Radeon 8[89]0M",
    re.IGNORECASE)
_INTEL_RE = re.compile(r"Intel.*Arc", re.IGNORECASE)


def list_adapters(run=subprocess.run) -> list[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_VideoController | ForEach-Object Name"]
    try:
        r = run(cmd, timeout=10, capture_output=True, text=True, creationflags=flags)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def amd_supported(name: str) -> bool:
    return bool(_AMD_RE.search(name))


def intel_xpu(name: str) -> bool:
    return bool(_INTEL_RE.search(name))


def runtime_dir() -> Path:
    env = os.environ.get("REFGRADE_RUNTIME")
    if env:
        return Path(env)
    rd = config.read_settings().get("runtime_dir")
    if rd:
        return Path(rd)
    return Path(os.environ.get("LOCALAPPDATA", "")) / "RefGrade" / "runtime"


def set_runtime_dir(path: str | None) -> None:
    """runtime_dir in config.json speichern; None löscht den Eintrag."""
    data = config.read_settings()
    if path is None:
        data.pop("runtime_dir", None)
    else:
        data["runtime_dir"] = path
    config.write_settings(data)


def _first_existing(path) -> Path:
    """Erster existierender Ordner von path aufwärts (root, wenn keiner)."""
    p = Path(path)
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    return p


def free_mb(path) -> int:
    """Freier Speicher in MB auf dem ersten existierenden Elternordner."""
    return shutil.disk_usage(_first_existing(path)).free // (1024 * 1024)


def check_target(path, flavor) -> str:
    """Zielordner prüfen; "" wenn ok, sonst englische Meldung."""
    if any(ord(c) > 127 for c in str(path)):
        return ("The path contains non-ASCII characters - uv and the torch DLLs "
                "do not work reliably with such paths.")
    need = FLAVORS[flavor]["size_mb"] * 2
    free = free_mb(path)
    if free < need:
        return (f"Not enough free space - the install needs about {need} MB, "
                f"but only {free} MB are free.")
    try:
        probe = _first_existing(path) / ".refgrade_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return f"The path is not writable: {path}"
    return ""


def use_existing(exe: str) -> str:
    """Vorhandenes Python übernehmen; "" wenn ok, sonst Meldung."""
    if python_ok(exe):
        data = config.read_settings()
        data["python_exe"] = exe
        config.write_settings(data)
        return ""
    return "This Python cannot import torch, numpy, kornia and PIL."


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
    flavor = data.get("flavor")
    if flavor is None:
        # Legacy marker without flavor: compare against the default torch version.
        return data.get("torch") == TORCH_VERSION and data.get("packages") == list(PACKAGES)
    if flavor not in FLAVORS:
        return False
    return (data.get("torch") == FLAVORS[flavor]["torch"]
            and data.get("packages") == list(PACKAGES))


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
    adapters = list_adapters(run)
    try:
        r = run(["nvidia-smi"], timeout=10, capture_output=True, text=True, creationflags=flags)
        if r.returncode != 0:
            return _no_nvidia(adapters)
        info = parse_nvidia_smi(r.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return _no_nvidia(adapters)

    try:
        r2 = run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                 timeout=10, capture_output=True, text=True, creationflags=flags)
        if r2.returncode != 0:
            return _no_nvidia(adapters)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return _no_nvidia(adapters)

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
    return {"gpu_name": gpu_name, "flavor": flavor, "message": message, "adapters": adapters}


def _no_nvidia(adapters: list[str]) -> dict:
    for name in adapters:
        if amd_supported(name):
            return {"gpu_name": name, "flavor": "rocm",
                    "message": "AMD GPU found - ROCm on Windows is a preview; "
                               "falls back to CPU if the check fails.",
                    "adapters": adapters}
    for name in adapters:
        if intel_xpu(name):
            return {"gpu_name": name, "flavor": "xpu", "message": "", "adapters": adapters}
    return {"gpu_name": None, "flavor": "cpu",
            "message": "No supported GPU found - installing the CPU version (slower).",
            "adapters": adapters}


def install_env() -> dict:
    env = clean_env()
    env["UV_PYTHON_INSTALL_DIR"] = str(runtime_dir() / "python")
    env["UV_CACHE_DIR"] = str(runtime_dir() / "cache")
    env["UV_LINK_MODE"] = "copy"
    return env


def install_commands(flavor: str, extra: tuple[str, ...] = ()) -> list[list[str]]:
    uv = str(uv_exe())
    vp = str(venv_python())
    venv_cmd = [uv, "venv", "--python", PYTHON_VERSION, "--managed-python", "--clear", str(venv_dir())]
    pkgs_cmd = [uv, "pip", "install", "--python", vp, *PACKAGES, *extra]
    if flavor == "rocm":
        return [
            venv_cmd,
            [uv, "pip", "install", "--python", vp, *ROCM_SDK],
            [uv, "pip", "install", "--python", vp, *ROCM_TORCH],
            pkgs_cmd,
        ]
    torch_version = FLAVORS[flavor]["torch"]
    return [
        venv_cmd,
        [uv, "pip", "install", "--python", vp, f"torch=={torch_version}",
         "--index-url", TORCH_INDEX.format(flavor=flavor)],
        pkgs_cmd,
    ]


class InstallError(RuntimeError):
    pass


class Cancelled(InstallError):
    pass


def install(flavor, log, cancelled=lambda: False, popen=subprocess.Popen, fetch=urlopen,
            extra: tuple[str, ...] = ()) -> str:
    rd = runtime_dir()
    rd.mkdir(parents=True, exist_ok=True)

    log_path = config.temp_dir() / "install.log"
    log_file = open(log_path, "a", encoding="utf-8")
    try:
        return _install_inner(flavor, log, cancelled, popen, fetch, log_file, log_path, extra)
    finally:
        log_file.close()


def _install_inner(flavor, log, cancelled, popen, fetch, log_file, log_path,
                   extra: tuple[str, ...] = ()) -> str:
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
    for i, cmd in enumerate(install_commands(flavor, extra), 1):
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
    def _run_check(flavor: str) -> tuple[int, str]:
        _log("Running check...")
        check_cmd = [str(venv_python()), "-E", "-c", check_code(flavor)]
        proc = popen(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, env=env, creationflags=flags)
        check_output = ""
        for line in proc.stdout:
            check_output = line.rstrip()
            _log(check_output)
        proc.wait()
        return proc.returncode, check_output

    rc, check_output = _run_check(flavor)
    if rc != 0:
        if flavor != "cpu":
            _log("GPU build failed the check - installing the CPU version.")
            for i, cmd in enumerate(install_commands("cpu", extra), 1):
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
            flavor = "cpu"
            rc, check_output = _run_check("cpu")
            if rc != 0:
                raise InstallError(f"Check failed (exit {rc}). Log: {log_path}")
        else:
            raise InstallError(f"Check failed (exit {rc}). Log: {log_path}")

    # 5. Cleanup + write marker
    shutil.rmtree(rd / "cache", ignore_errors=True)
    marker = {
        "flavor": flavor,
        "torch": FLAVORS[flavor]["torch"],
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


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ref_grade.runtime",
        description="RefGrade runtime: detect hardware or install (no Qt).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--detect", action="store_true",
                       help="print hardware detection as JSON (installs nothing)")
    group.add_argument("--install", metavar="FLAVOR",
                       help=f"install the runtime for FLAVOR ({', '.join(FLAVORS)})")
    parser.add_argument("--dir", metavar="ORT",
                        help="install directory (default: current runtime dir)")
    args = parser.parse_args(argv)

    if args.detect:
        print(json.dumps(detect()))
        return 0

    flavor = args.install
    if flavor not in FLAVORS:
        print(f"unknown flavor: {flavor} (choose from {', '.join(FLAVORS)})",
              file=sys.stderr)
        return 1
    if args.dir:
        set_runtime_dir(args.dir)
    try:
        install(flavor, log=print)
    except InstallError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
