"""Testrunner: sammelt tests/test_*.py. Standard = schnelle Suite (Ziel <= 45 s); Tests mit @slow
und ganze Module in SLOW_MODULES laufen nur mit REFGRADE_SLOW=1 bzw. `--full` (volle Suite)."""

import sys
import os
import subprocess
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

# torch EINMAL vor allen Testmodulen laden: wird torch erst NACH PyQt6 importiert,
# schlägt c10.dll mit WinError 1114 fehl (DLL-Init-Konflikt, s. nacht_plan 2026-10-12).
# Vorher geladen → PyQt6 danach ist unkritisch. (2026-10-18)
import torch  # noqa: F401

# Windows-Konsole (cp1252) kann Δ/≥/→ nicht encodieren → UTF-8 erzwingen.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


class _SubprocessModuleTest(unittest.TestCase):
    """Ein ganzes Testmodul in einem eigenen Subprozess ausführen.

    2026-10-18: Das Shared-Prozess-Setup (torch + PyQt6 in einem Prozess) crasht
    deterministisch mit 0xC0000409 — laut WinDbg-Semantik Subcode 0x7 =
    FAST_FAIL_FATAL_APP_EXIT, d.h. eine UNHANDLITE C++-Exception (std::terminate)
    aus der torch/PyQt-DLL-Interaktion, kein echter Stack-Overrun. Der Ort ist
    nicht deterministisch (je nach Modul-Menge an test_mcp_server, test_echtzeit,
    test_plugin_k13 …). Isolierte Module laufen einzeln grün → Subprozess pro Modul.
    """

    def __init__(self, module_name: str, test_dir: str):
        super().__init__()
        self.module_name = module_name
        self.test_dir = test_dir

    def runTest(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [self.test_dir, os.path.dirname(self.test_dir)]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", self.module_name],
            cwd=os.path.dirname(self.test_dir),   # core/
            env=env, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0,
                         f"Subprozess {self.module_name} fehlgeschlagen (rc={proc.returncode}):\n"
                         f"{(proc.stderr or proc.stdout)[-2000:]}")


def main() -> int:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    SKIP = {"test_server.py"}  # verwaist (Server entfernt); Datei löscht der Nutzer
    # Ganze Module, die in der schnellen Suite zu langsam sind (Summe > ~1.5 s):
    # Qt-Dialoge in mehreren Auflösungen, Worker-/MCP-Subprozesse, torch-E2E-Benchmarks.
    # Sie laufen nur mit REFGRADE_SLOW=1 / --full. (2026-10-18: Suite auf <= 45 s,
    # gemessen mit tools/time_tests.py -> tmp/test_times.jsonl.)
    SLOW_MODULES = {
        "test_analysis", "test_auto", "test_clip_warn", "test_combo_regressions",
        "test_fit_pair", "test_fit_sliders", "test_help_toggle", "test_mcp_server",
        "test_pair_bench", "test_plugin_k10b", "test_plugin_k17", "test_plugin_k18",
        "test_plugin_k22", "test_plugin_k25b", "test_plugin_k25c", "test_plugin_k25d",
        "test_plugin_k25i", "test_plugin_k27a", "test_plugin_k27d", "test_plugin_k27e",
        "test_plugin_k27f", "test_plugin_k28f", "test_plugin_k29_wheels_ui",
        "test_plugin_k30_auto", "test_plugin_k6", "test_plugin_k9b", "test_presets",
        "test_preview_no_loop", "test_preview_stable", "test_sections", "test_session",
        "test_sort_luts", "test_userprompts_1016", "test_wave", "test_worker",
    }
    # 2026-10-18: Module, die im Shared-Prozess 0xC0000409 auslösen (unhandled C++
    # exception, torch+PyQt-DLL-Interaktion) → eigener Subprozess. Empirisch ermittelt:
    # Suite laufen lassen, crashtes Modul hier ergänzen, wiederholen.
    SUBPROCESS_MODULES = {
        "test_echtzeit",
        # Cursor-Test: im Shared-Prozess unzuverlässig (Maus-Zustand aus früheren
        # Qt-Modulen), frischer Prozess = grün. (2026-10-18)
        "test_split_line",
    }
    full = os.environ.get("REFGRADE_SLOW") == "1"
    for fname in sorted(os.listdir(test_dir)):
        if fname.startswith("test_") and fname.endswith(".py") and fname not in SKIP:
            module_name = fname[:-3]
            if module_name in SLOW_MODULES and not full:
                continue
            if module_name in SUBPROCESS_MODULES:
                suite.addTest(_SubprocessModuleTest(module_name, test_dir))
                continue
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
