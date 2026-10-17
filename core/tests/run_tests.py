"""Testrunner: sammelt tests/test_*.py. Standard = schnelle Suite (< 60 s); Tests mit @slow
(> 2 s) laufen nur mit REFGRADE_SLOW=1 bzw. `--full` (volle Suite)."""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for `from slow import slow`
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")   # tests never touch the user's settings
# ... nor the user's config.json (python_exe / runtime_dir there changed config tests, 2026-10-17)
_cfg = Path(__file__).resolve().parents[2] / "tmp" / f"test_config_{os.getpid()}.json"
_cfg.parent.mkdir(exist_ok=True)
_cfg.unlink(missing_ok=True)
os.environ.setdefault("REFGRADE_CONFIG", str(_cfg))
if "--full" in sys.argv:
    os.environ["REFGRADE_SLOW"] = "1"

# Windows-Konsole (cp1252) kann Δ/≥/→ nicht encodieren → UTF-8 erzwingen.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def main() -> int:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    SKIP = {"test_server.py"}  # verwaist (Server entfernt); Datei löscht der Nutzer
    for fname in sorted(os.listdir(test_dir)):
        if fname.startswith("test_") and fname.endswith(".py") and fname not in SKIP:
            module_name = fname[:-3]
            loader.pushd = test_dir
            module_path = os.path.join(test_dir, fname)
            # PID-spezifischer Settings-Name: verhindert, dass parallele
            # Testprozesse dieselben QSettings teilen.
            os.environ["REFGRADE_SETTINGS"] = f"RefGradeTests_{os.getpid()}"
            spec = __import__("importlib").util.spec_from_file_location(
                module_name, module_path
            )
            mod = __import__("importlib").util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            suite.addTests(loader.loadTestsFromModule(mod))
            # Nach dem Import: Settings aufräumen, damit das nächste Modul
            # mit einem sauberen Zustand startet.
            if "PyQt6.QtCore" in sys.modules:
                from PyQt6.QtCore import QSettings
                QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).clear()

    print("Suite:", "voll" if os.environ.get("REFGRADE_SLOW") == "1" else "schnell (@slow übersprungen)")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
