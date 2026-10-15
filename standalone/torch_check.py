"""Torch-Erkennung und Setup für die Standalone-GUI.

`ensure_torch()` ist reine Python-Logik (kein Qt) und wird VOR dem PyQt6-Import
gerufen, damit torch vor Qt geladen wird – auf Windows führt ein Qt-Import vor
torch zu WinError 1114 (DLL-Konflikt). Der Setup-Dialog braucht Qt und wird
deshalb erst importiert, wenn torch tatsächlich fehlt (dann ist kein Konflikt
möglich). Der einmal gewählte torch-Pfad liegt in der JSON-Config
(``%APPDATA%/RefGrade/config.json``, Modul ``config``) und wird vor dem
Qt-Import gelesen.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:  # Paket-Kontext (standalone.torch_check) vs. Top-Level (Test/EXE)
    from . import config
except ImportError:
    import config

_PIP_INDEX = "https://download.pytorch.org/whl/cu118"


def _no_window() -> int:
    """CREATE_NO_WINDOW auf Windows, sonst 0 (kein Fenster für Subprozesse)."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _candidate_paths() -> list[Path]:
    """Externe Standorte, die torch enthalten könnten (ohne Qt)."""
    paths: list[Path] = []
    for part in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if part:
            paths.append(Path(part))
    paths.append(Path.home() / "torch_libs")
    paths.append(Path("C:/torch_libs"))
    paths.append(Path(sys.prefix) / "Lib" / "site-packages")
    return paths


def _try_import_torch() -> bool:
    """True, wenn torch (jetzt, in diesem Prozess) importierbar ist."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def ensure_torch() -> tuple[bool, str]:
    """Check for torch. Returns (available: bool, message: str).

    Reine Python-Logik ohne Qt – darf vor dem PyQt6-Import laufen.
    """
    print(f"[INFO] Config: {config.CONFIG_FILE}")
    # 1. Gespeicherter Pfad (config.json) zuerst in den sys.path legen.
    saved = config.get_torch_path()
    if saved is not None and saved.is_dir() and str(saved) not in sys.path:
        sys.path.insert(0, str(saved))
    # 2. Normaler Import
    if _try_import_torch():
        import torch
        return True, f"torch {torch.__version__}"
    # 3. Externe Standorte
    for p in _candidate_paths():
        if p.is_dir() and (p / "torch").is_dir():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            if _try_import_torch():
                import torch
                return True, f"torch {torch.__version__} (extern: {p})"
    return False, "torch nicht gefunden"


def _torch_importable(extra_path: Path | None = None) -> bool:
    """Prüft in einem frischen Python-Prozess (ohne Qt), ob torch importierbar ist.

    Der GUI-Prozess hat Qt geladen und könnte torch nicht mehr importieren
    (WinError 1114); der Worker ist ein separater Prozess, darum genügt hier die
    Frage, ob torch auf der Maschine verfügbar ist.
    """
    code = "import torch"
    if extra_path is not None:
        code = f"import sys; sys.path.insert(0, {str(extra_path)!r}); import torch"
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=_no_window(),
            timeout=180,
        )
        return r.returncode == 0
    except Exception:
        return False


class TorchSetupDialog:
    """Modaler Setup-Dialog, wenn torch fehlt: installieren, Pfad wählen oder ohne GPU."""

    def __init__(self) -> None:
        self._has_torch = False
        self._proc: subprocess.Popen | None = None

    def run(self) -> bool:
        """Zeigt den Dialog und liefert den finalen torch-Zustand zurück."""
        from PyQt6.QtCore import QTimer
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QDialog, QFileDialog, QLabel, QPushButton, QVBoxLayout
        from ref_grade import style

        dlg = QDialog()
        self._dlg = dlg
        dlg.setWindowTitle("PyTorch nicht gefunden")
        dlg.setStyleSheet(style.STYLESHEET)
        dlg.setFont(style.font("base"))
        dlg.setModal(True)

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(28, 28, 28, 28)
        lay.setSpacing(14)

        msg = QLabel(
            "GPU-Features (Color Replace) brauchen PyTorch.\n"
            "Ohne PyTorch läuft RefGrade nur mit klassischen Features.")
        msg.setWordWrap(True)
        lay.addWidget(msg)

        status = QLabel("")
        status.setWordWrap(True)
        status.setProperty("muted", True)
        lay.addWidget(status)

        btn_install = QPushButton("Automatisch installieren")
        btn_path = QPushButton("Pfad manuell wählen…")
        btn_skip = QPushButton("Ohne GPU weiter")
        btn_install.setProperty("main", "ok")
        btn_skip.setProperty("main", "cancel")
        for b in (btn_install, btn_path, btn_skip):
            b.setFont(style.font("title", QFont.Weight.Normal))
            b.setMinimumHeight(38)
        lay.addWidget(btn_install)
        lay.addWidget(btn_path)
        lay.addWidget(btn_skip)

        def on_skip() -> None:
            self._has_torch = False
            dlg.accept()

        def on_pick_path() -> None:
            d = QFileDialog.getExistingDirectory(
                dlg, "Ordner wählen (enthält torch/)", str(Path.home()))
            if not d:
                return
            p = Path(d)
            if _torch_importable(p):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                config.set_torch_path(p)
                self._has_torch = True
                dlg.accept()
            else:
                status.setText(f"Kein torch in {p} gefunden.")

        def on_install() -> None:
            if not Path(sys.executable).name.lower().startswith("python"):
                status.setText(
                    "Python nicht verfügbar – Installation nur in Python-Umgebung möglich")
                return
            btn_install.setEnabled(False)
            btn_path.setEnabled(False)
            status.setText("Installiere PyTorch… bitte warten (ca. 2–5 min)")
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "torch",
                 "--index-url", _PIP_INDEX],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=_no_window(),
            )
            timer = QTimer(dlg)
            timer.setInterval(500)

            def poll() -> None:
                if self._proc is None or self._proc.poll() is not None:
                    timer.stop()
                    btn_install.setEnabled(True)
                    btn_path.setEnabled(True)
                    rc = self._proc.returncode if self._proc is not None else 1
                    if rc == 0 and _torch_importable():
                        self._has_torch = True
                        dlg.accept()
                    else:
                        status.setText(
                            "Installation fehlgeschlagen – PyTorch konnte nicht geladen werden.\n"
                            "Pfad manuell wählen oder ohne GPU weiter.")

            timer.timeout.connect(poll)
            timer.start()

        btn_install.clicked.connect(on_install)
        btn_path.clicked.connect(on_pick_path)
        btn_skip.clicked.connect(on_skip)

        dlg.exec()
        return self._has_torch
