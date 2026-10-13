"""Build the Krita-importable plugin ZIP: dist/ref_grade-<version>.zip.

Usage:  python tools/build_zip.py

Layout (what Krita's "Import Python Plugin from File..." expects):
    ref_grade.desktop
    ref_grade/            plugin package (incl. ref_grade.action)
    ref_grade/core/       worker.py + refgrade/ (+ LICENSE, NOTICE) for the external Python worker
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin" / "ref_grade"
DESKTOP = ROOT / "plugin" / "ref_grade.desktop"
CORE = ROOT / "core"
DIST = ROOT / "dist"

_SKIP_DIRS = {"__pycache__", ".claude", "tests", "bench", "reports", "logs", ".pytest_cache"}
_SKIP_FILES = {"crash.log", "debug_log.md"}


def version() -> str:
    text = (CORE / "refgrade" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^VERSION\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        sys.exit("VERSION not found in core/refgrade/version.py")
    return m.group(1)


def _files(base: Path):
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        if not p.is_file() or _SKIP_DIRS & set(rel.parts[:-1]):
            continue
        if p.suffix == ".pyc" or p.name in _SKIP_FILES:
            continue
        yield p, rel


def _parents(arc: str):
    parts = arc.split("/")[:-1]
    for i in range(1, len(parts) + 1):
        yield "/".join(parts[:i])


def main() -> int:
    DIST.mkdir(exist_ok=True)
    out = DIST / f"ref_grade-{version()}.zip"
    entries: list[tuple[Path, str]] = [(DESKTOP, "ref_grade.desktop")]
    entries += [(p, f"ref_grade/{r.as_posix()}") for p, r in _files(PLUGIN)]
    entries += [(CORE / "worker.py", "ref_grade/core/worker.py")]
    entries += [(p, f"ref_grade/core/refgrade/{r.as_posix()}") for p, r in _files(CORE / "refgrade")]
    for name in ("LICENSE", "NOTICE"):
        if (CORE / name).is_file():
            entries.append((CORE / name, f"ref_grade/core/{name}"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        # Krita's importer looks for an explicit "ref_grade/" directory entry.
        dirs = sorted({d for _, arc in entries for d in _parents(arc)})
        for d in dirs:
            z.writestr(d + "/", "")
        for src, arc in entries:
            z.write(src, arc)
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KiB, {len(entries)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
