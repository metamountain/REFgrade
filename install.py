"""Install RefGrade into Krita's Python plugin directory.

Usage:
    python install.py              # install
    python install.py --uninstall  # remove
    python install.py --dry-run    # print targets, write nothing
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLUGIN_SRC = ROOT / "plugin" / "ref_grade"
DESKTOP_SRC = ROOT / "plugin" / "ref_grade.desktop"
ACTION_SRC = PLUGIN_SRC / "ref_grade.action"


def _appdata() -> Path:
    return Path(os.environ.get("APPDATA", ""))


def _targets() -> dict[str, Path]:
    appdata = _appdata()
    return {
        "junction": appdata / "krita" / "pykrita" / "ref_grade",
        "desktop": appdata / "krita" / "pykrita" / "ref_grade.desktop",
        "action": appdata / "krita" / "actions" / "ref_grade.action",
    }


def _is_junction(path: Path) -> bool:
    return path.exists() and os.path.islink(path)


def _do_install(dry: bool) -> None:
    t = _targets()

    # 1. Junction
    if _is_junction(t["junction"]):
        print(f"[ok]   junction already exists: {t['junction']}")
    elif t["junction"].exists():
        print(f"[skip] {t['junction']} exists but is not a junction")
    else:
        if dry:
            print(f"[dry]  would create junction: {t['junction']} -> {PLUGIN_SRC}")
        else:
            t["junction"].parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(t["junction"]), str(PLUGIN_SRC)],
                check=True,
            )
            print(f"[done] created junction: {t['junction']} -> {PLUGIN_SRC}")

    # 2. Desktop file
    if t["desktop"].exists():
        print(f"[ok]   desktop already exists: {t['desktop']}")
    else:
        if dry:
            print(f"[dry]  would copy: {DESKTOP_SRC} -> {t['desktop']}")
        else:
            t["desktop"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DESKTOP_SRC, t["desktop"])
            print(f"[done] copied: {DESKTOP_SRC} -> {t['desktop']}")

    # 3. Action file
    if t["action"].exists():
        print(f"[ok]   action already exists: {t['action']}")
    else:
        if dry:
            print(f"[dry]  would copy: {ACTION_SRC} -> {t['action']}")
        else:
            t["action"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ACTION_SRC, t["action"])
            print(f"[done] copied: {ACTION_SRC} -> {t['action']}")


def _do_uninstall(dry: bool) -> None:
    t = _targets()

    # 1. Junction
    if _is_junction(t["junction"]):
        if dry:
            print(f"[dry]  would remove junction: {t['junction']}")
        else:
            os.rmdir(t["junction"])
            print(f"[done] removed junction: {t['junction']}")
    else:
        print(f"[ok]   junction not present: {t['junction']}")

    # 2. Desktop file
    if t["desktop"].exists():
        if dry:
            print(f"[dry]  would delete: {t['desktop']}")
        else:
            t["desktop"].unlink()
            print(f"[done] deleted: {t['desktop']}")
    else:
        print(f"[ok]   desktop not present: {t['desktop']}")

    # 3. Action file
    if t["action"].exists():
        if dry:
            print(f"[dry]  would delete: {t['action']}")
        else:
            t["action"].unlink()
            print(f"[done] deleted: {t['action']}")
    else:
        print(f"[ok]   action not present: {t['action']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install RefGrade into Krita.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uninstall", action="store_true", help="Remove the installation.")
    group.add_argument("--dry-run", action="store_true", help="Print targets, write nothing.")
    args = parser.parse_args()

    if args.uninstall:
        _do_uninstall(dry=args.dry_run)
    else:
        _do_install(dry=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
