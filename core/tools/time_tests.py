"""Per-Test-Zeiten messen.

Jedes Testmodul läuft in einem eigenen Subprozess (vermeidet die
torch-DLL/PyQt-Crash-Kaskade des Single-Process-Runners, s. nacht_plan 2026-10-12).
Die Dauer jedes Tests wird nach tmp/test_times.jsonl geloggt (eine JSON-Zeile pro
Test) und am Ende die langsamsten Tests gemeldet.

Aufruf (aus core/):
    python tools/time_tests.py            # schnelle Suite
    python tools/time_tests.py --full     # volle Suite (REFGRADE_SLOW=1)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
ROOT = CORE.parent
TESTS = CORE / "tests"
OUT = ROOT / "tmp" / "test_times.jsonl"

SKIP = {"test_server.py"}   # verwaist (wie in tests/run_tests.py)


def _worker(module_name: str) -> None:
    """Subprozess: genau ein Modul laden, Dauer je Test als JSON-Zeile auf stdout."""
    import importlib.util
    import unittest

    if os.environ.get("TIME_TESTS_FULL") == "1":
        os.environ["REFGRADE_SLOW"] = "1"
    os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
    _cfg = ROOT / "tmp" / f"test_config_{os.getpid()}.json"
    _cfg.parent.mkdir(exist_ok=True)
    _cfg.unlink(missing_ok=True)
    os.environ.setdefault("REFGRADE_CONFIG", str(_cfg))
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

    sys.path.insert(0, str(CORE))
    sys.path.insert(0, str(TESTS))
    os.environ["REFGRADE_SETTINGS"] = f"RefGradeTests_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, str(TESTS / (module_name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    class TimingResult(unittest.TestResult):
        def __init__(self) -> None:
            super().__init__()
            self._t0 = 0.0

        def startTest(self, test) -> None:
            super().startTest(test)
            self._t0 = time.perf_counter()

        def stopTest(self, test) -> None:
            dt = time.perf_counter() - self._t0
            print(json.dumps({"test": str(test), "seconds": round(dt, 3)}), flush=True)
            super().stopTest(test)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(mod)
    result = TimingResult()
    suite.run(result)
    if "PyQt6.QtCore" in sys.modules:
        from PyQt6.QtCore import QSettings
        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()


def main() -> int:
    full = "--full" in sys.argv
    names = sorted(
        f[:-3] for f in os.listdir(TESTS)
        if f.startswith("test_") and f.endswith(".py") and f[:-3] not in SKIP
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("", encoding="utf-8")   # dieser Lauf ist der Log (kein Append-Altbestand)

    rows: list[dict] = []
    module_times: list[tuple[str, float, bool]] = []
    t_start = time.perf_counter()
    for name in names:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        if full:
            env["TIME_TESTS_FULL"] = "1"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--module", name],
            capture_output=True, text=True, encoding="utf-8", env=env,
            cwd=str(CORE),
        )
        wall = time.perf_counter() - t0
        ok = proc.returncode == 0
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                    row["module"] = name
                    rows.append(row)
                except json.JSONDecodeError:
                    pass
        module_times.append((name, wall, ok))
        status = "ok" if ok else f"CRASH rc={proc.returncode}"
        print(f"{name:38s} {wall:8.1f}s  {status}", flush=True)
        if not ok:
            tail = (proc.stderr or "").strip().splitlines()[-5:]
            for l in tail:
                print(f"    {l}")

    total = time.perf_counter() - t_start
    with OUT.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    rows.sort(key=lambda r: r["seconds"], reverse=True)
    print(f"\nGesamt: {total:.1f}s  ({len(rows)} Tests, Log: {OUT})")
    print("\nTop 20 langsamste:")
    for r in rows[:20]:
        print(f"  {r['seconds']:8.3f}s  {r['test']}")
    crashed = [m for m, _w, ok in module_times if not ok]
    return 1 if crashed else 0


if __name__ == "__main__":
    if "--module" in sys.argv:
        _worker(sys.argv[sys.argv.index("--module") + 1])
    else:
        sys.exit(main())
