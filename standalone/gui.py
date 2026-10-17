"""Standalone RefGrade GUI: volle Dialog-Oberfläche ohne Krita.

Start:
    python standalone/gui.py [bild.png] [--setup]

Ohne Argument: Datei-Dialog. Mit Argument: direkt laden.
`--setup` zeigt den Install-Dialog immer (Neu einrichten / Ort ändern).
Ohne torch im laufenden Python: Install-Dialog (runtime.py), danach Neustart
mit dem Runtime-Python (torch darf nicht nach PyQt6 im selben Prozess laden).
Worker wird automatisch gestartet und beim Beenden gestoppt.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "core"

for p in (str(_CORE), str(_ROOT / "plugin")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Kein Krita-Mock: ref_grade/__init__.py toleriert das Fehlen von krita
# (ImportError-Guard), und die hier importierten Module brauchen krita nicht.

import numpy as np
from PIL import Image

from ref_grade import config, worker_client, worker_process


def _has_torch() -> bool:
    """True, wenn das laufende Python torch importieren kann."""
    return importlib.util.find_spec("torch") is not None


def _restart(python: str) -> None:
    """App mit anderem Python neu starten und diesen Prozess beenden.

    torch darf nicht nach PyQt6 im selben Prozess geladen werden (DLL-Regel),
    darum Neustart statt Import.
    """
    subprocess.Popen([str(python), __file__, *sys.argv[1:]])
    sys.exit(0)


def _ensure_torch(setup: bool) -> None:
    """torch sicherstellen; bei Bedarf Install-Dialog zeigen und neu starten.

    Läuft vor jedem PyQt6-Import: hat das Python torch, wird torch zuerst
    geladen (DLL-Regel); sonst kommt PyQt6 für den Dialog und der Prozess
    endet über Neustart/Abbruch.
    """
    if _has_torch() and not setup:
        import torch  # noqa: F401  (vor dem PyQt6-Import – DLL-Regel)
        return
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox
    from ref_grade import runtime
    from ref_grade.dlg_install import InstallDialog

    app = QApplication(sys.argv)
    before = config.read_settings().get("python_exe")

    def _install(flavor, log, cancelled):
        return runtime.install(flavor, log, cancelled, extra=("PyQt6",))

    dlg = InstallDialog(parent=None, install_fn=_install)
    accepted = dlg.exec() == QDialog.DialogCode.Accepted
    if not accepted:
        QMessageBox.information(
            None, "RefGrade", "Setup cancelled - RefGrade is closing.")
        sys.exit(0)

    after = config.read_settings().get("python_exe")
    if after and after != before:
        target = after  # „Use existing Python“ gewählt
    elif runtime.is_installed():
        target = str(runtime.venv_python())
    elif after and runtime.python_ok(after):
        target = after
    else:
        QMessageBox.critical(
            None, "RefGrade",
            "No usable Python with torch was found after the setup.")
        sys.exit(1)
    _restart(target)


def _read_source(path: Path, tmp: Path) -> dict:
    """Bilddatei → source-Dict (wie krita_bridge.read_active, aber ohne Krita)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    a = np.array(img)
    bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)])
    raw_path = tmp / "source.raw"
    raw_path.write_bytes(bgra.tobytes())
    return {
        "raw": str(raw_path),
        "width": w,
        "height": h,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "path": str(path.resolve()),
    }


def _pick_file(parent=None) -> Path | None:
    from PyQt6.QtWidgets import QFileDialog
    path_str = QFileDialog.getOpenFileName(
        parent, "Bild öffnen", str(Path.home()),
        "Bilder (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")[0]
    return Path(path_str) if path_str else None


def _apply_window_scheme(win) -> None:
    """Theme main window (palette + stylesheet) and the native title bar with the active scheme."""
    from ref_grade import schemes, win_chrome
    schemes.apply_scheme(win)
    win_chrome.apply(win, schemes.current_scheme())


def build_window(title: str = "REFgrade"):
    """Build the (empty) main window with toolbar and container, themed — no worker needed.

    Returns (win, lbl_file, btn_open, dlg_container)."""
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QLabel, QMainWindow, QToolBar, QVBoxLayout, QWidget
    from PyQt6.QtCore import QSettings
    from ref_grade import schemes

    schemes.set_current(QSettings("RefGrade", "RefGrade").value("scheme", schemes.DEFAULT, type=str))
    win = QMainWindow()
    win.setWindowTitle(title)
    win.resize(1400, 900)
    central = QWidget()
    win.setCentralWidget(central)
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    tb = QToolBar()
    tb.setMovable(False)
    btn_open = QAction("Open…", win)
    btn_open.setShortcut("Ctrl+O")
    tb.addAction(btn_open)
    lbl_file = QLabel("")
    tb.addWidget(lbl_file)
    win.addToolBar(tb)

    dlg_container = QWidget()
    layout.addWidget(dlg_container)
    _apply_window_scheme(win)
    return win, lbl_file, btn_open, dlg_container


def main() -> None:
    _ensure_torch("--setup" in sys.argv[1:])
    args = [a for a in sys.argv[1:] if a != "--setup"]

    # Erst jetzt PyQt6 laden (dialog.py importiert es): torch ist bereits
    # geladen oder der Prozess endet über _ensure_torch (Neustart/Abbruch).
    from ref_grade.dialog import RefGradeDialog
    from PyQt6.QtWidgets import (
        QApplication, QFileDialog, QVBoxLayout,
    )
    from PyQt6.QtGui import QImage
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("REFgrade")

    # 1. Bild wählen (CLI-Argument oder Dialog)
    if args:
        img_path = Path(args[0])
        if not img_path.is_file():
            print(f"Fehler: Datei nicht gefunden: {img_path}")
            sys.exit(1)
    else:
        picked = _pick_file()
        if not picked:
            print("Abgebrochen.")
            return
        img_path = picked

    port = config.PORT
    python = config.python_exe()
    from ref_grade.version import __version__ as version

    win, lbl_file, btn_open, dlg_container = build_window(f"REFgrade {version} — {img_path.name}")

    def load_image(path: Path) -> None:
        nonlocal img_path
        print(f"Bild: {path} ({path.stat().st_size // 1024} KB)")
        worker_process.ensure(port, python, config.CORE, timeout=30.0)
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_gui_"))
        source = _read_source(path, tmp)
        print(f"Source: {source['width']}x{source['height']}")
        resp = worker_client.call(port, "/load", source)
        source["session"] = resp["session"]
        print(f"Session: {source['session']}")

        pil_img = Image.open(path).convert("RGB")
        pil_img.thumbnail((2048, 2048))
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888).copy()

        def on_ok(raw_path: str, layer_name: str, clarity_info: dict | None = None) -> None:
            default = str(path.with_name(f"{path.stem}_grad.png"))
            out, fmt = QFileDialog.getSaveFileName(
                win, "Save graded image", default,
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)")
            if not out:
                print("Nicht gespeichert")
                return
            out_p = Path(out)
            if out_p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                default_ext = {"PNG (*.png)": ".png",
                               "JPEG (*.jpg *.jpeg)": ".jpg",
                               "TIFF (*.tif *.tiff)": ".tif"}.get(fmt, ".png")
                out_p = out_p.with_suffix(default_ext)
            arr = np.fromfile(raw_path, dtype=np.uint8).reshape(
                source["height"], source["width"], 4)
            rgb = np.ascontiguousarray(arr[..., [2, 1, 0]])
            img = Image.fromarray(rgb, mode="RGB")
            if out_p.suffix.lower() in (".jpg", ".jpeg"):
                img.save(out_p, quality=95)
            elif out_p.suffix.lower() in (".tif", ".tiff"):
                img.save(out_p)
            else:
                img.save(out_p)
            print(f"Gespeichert: {out_p}")

        def on_update() -> bool:
            return False

        def save_lut(folder: str, filename: str) -> str | None:
            p, _ = QFileDialog.getSaveFileName(
                win, "LUT speichern", str(Path(folder) / filename), "LUT (*.cube)")
            return p or None

        # Altes Dialog-Widget entfernen
        old_layout = dlg_container.layout()
        if old_layout:
            item = old_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.close()
                w.deleteLater()

        inner = QVBoxLayout(dlg_container)
        inner.setContentsMargins(0, 0, 0, 0)
        dlg = RefGradeDialog(
            qimg, dlg_container,
            client=worker_client.call,
            source=source,
            save_lut=save_lut,
            on_ok=on_ok,
            on_update=on_update,
        )
        dlg.setWindowFlags(Qt.WindowType.Widget)   # embed: a QDialog is a top-level window otherwise
        inner.addWidget(dlg)
        dlg.show()
        dlg._scheme_combo.currentTextChanged.connect(lambda _n: _apply_window_scheme(win))
        img_path = path
        win.setWindowTitle(f"REFgrade {version} — {path.name}")
        lbl_file.setText(f"  {path}")

    def on_open() -> None:
        picked = _pick_file(win)
        if picked:
            load_image(picked)

    btn_open.triggered.connect(on_open)

    # Erstes Bild laden
    load_image(img_path)
    win.show()
    _apply_window_scheme(win)

    try:
        app.exec()
    finally:
        worker_process.stop(port)
        print("Worker gestoppt. Tschüss.")


if __name__ == "__main__":
    main()
