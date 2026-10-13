"""Per-test wall time of the fast suite (same discovery as tests/run_tests.py).

    python tools/test_times.py [N]      (from core\)  → the N slowest tests (default 25)

Use it when the fast suite nears 60 s: tests > 2 s get @slow (tests/slow.py).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
TESTS = CORE / "tests"
sys.path[:0] = [str(CORE), str(TESTS)]
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
SKIP = {"test_server.py"}


class Timed(unittest.TextTestResult):
    times: list[tuple[float, str]] = []

    def startTest(self, test):  # noqa: N802
        self._t = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):  # noqa: N802
        Timed.times.append((time.perf_counter() - self._t, test.id()))
        super().stopTest(test)


suite = unittest.TestSuite()
for f in sorted(TESTS.glob("test_*.py")):
    if f.name in SKIP:
        continue
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f.stem] = mod
    spec.loader.exec_module(mod)
    suite.addTests(unittest.TestLoader().loadTestsFromModule(mod))

t0 = time.perf_counter()
with open(os.devnull, "w", encoding="utf-8") as null:
    result = unittest.TextTestRunner(resultclass=Timed, stream=null, verbosity=0).run(suite)
total = time.perf_counter() - t0
print(f"total {total:.1f} s, {result.testsRun} tests, failures {len(result.failures)}, "
      f"errors {len(result.errors)}, skipped {len(result.skipped)}")
for sec, name in sorted(Timed.times, reverse=True)[: int(sys.argv[1]) if len(sys.argv) > 1 else 25]:
    print(f"{sec:6.2f}  {name}")
