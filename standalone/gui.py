"""Standalone RefGrade GUI: volle Dialog-Oberfläche ohne Krita.

Start:
    refgrade.bat [bild.png] [--setup]          (oder: python standalone/gui.py ...)

Ohne Argument: leeres Fenster, Bild dort waehlen (Open.../Ctrl+O, Ctrl+V, Datei hineinziehen).
Mit Argument: direkt laden.
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
    # without --setup: the restarted process would show the installer again, for ever
    subprocess.Popen([str(python), __file__, *(a for a in sys.argv[1:] if a != "--setup")])
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


def open_rgb8(path: Path):
    """Open an image upright (EXIF orientation) as 8-bit RGB. 16-bit greyscale is scaled, not
    clipped (Pillow's convert("RGB") truncates I;16 → almost white). Returns (PIL image, icc)."""
    from PIL import ImageOps
    with Image.open(path) as src:
        icc = src.info.get("icc_profile")
        img = ImageOps.exif_transpose(src)
        if img.mode in ("I;16", "I;16L", "I;16B", "I"):
            a = np.asarray(img, dtype=np.float32)
            a = a / 257.0 if a.max() > 255 else a
            img = Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8)).convert("RGB")
        else:
            img = img.convert("RGB")
    return img, icc


def _read_source(path: Path, tmp: Path) -> tuple[dict, bytes | None]:
    """Bilddatei → (source-Dict wie krita_bridge.read_active, ICC-Profil). The profile stays out
    of the dict: the dict goes to the worker as JSON (bytes → "not JSON serializable", 2026-10-17)."""
    img, icc = open_rgb8(path)
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
    }, icc


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
SAVE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_FILTER_EXT = {"PNG (*.png)": ".png", "JPEG (*.jpg *.jpeg)": ".jpg", "TIFF (*.tif *.tiff)": ".tif"}


def _pick_file(parent=None, folder: str | None = None) -> Path | None:
    from PyQt6.QtWidgets import QFileDialog
    path_str = QFileDialog.getOpenFileName(
        parent, "Open image", folder or str(Path.home()),
        "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")[0]
    return Path(path_str) if path_str else None


def default_save_path(src: Path) -> Path:
    """Original name + "_rg", the original's format where we write it, else .png (2026-10-17)."""
    ext = src.suffix.lower() if src.suffix.lower() in SAVE_EXT else ".png"
    return src.with_name(f"{src.stem}_rg{ext}")


def is_pasted(src: Path) -> bool:
    """A pasted screenshot lives in a temp folder — never save next to it."""
    return any(part.startswith("refgrade_paste_") for part in src.parts)


def write_result(raw_path: str, width: int, height: int, out: Path, icc: bytes | None = None) -> None:
    """Worker result (BGRA 8-bit raw) → image file; the source's ICC profile goes along (the
    grading works on the pixel values, so the profile still describes them)."""
    arr = np.fromfile(raw_path, dtype=np.uint8).reshape(height, width, 4)
    img = Image.fromarray(np.ascontiguousarray(arr[..., [2, 1, 0]]))
    extra = {"icc_profile": icc} if icc else {}
    if out.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out, quality=95, **extra)
    else:
        img.save(out, **extra)


def _apply_window_scheme(win) -> None:
    """Theme main window (palette + stylesheet) and the native title bar with the active scheme."""
    from ref_grade import schemes, win_chrome
    schemes.apply_scheme(win)
    win_chrome.apply(win, schemes.current_scheme())


def build_window(title: str = "REFgrade"):
    """Build the (empty) main window with toolbar, start screen and container, themed — no
    worker needed. Files dropped on the window go to `win.file_dropped`.

    Returns (win, lbl_file, btn_open, dlg_container)."""
    from PyQt6.QtCore import QSettings, Qt, pyqtSignal
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (QLabel, QMainWindow, QPushButton, QToolBar, QVBoxLayout,
                                 QWidget)
    from ref_grade import schemes

    class _Window(QMainWindow):
        file_dropped = pyqtSignal(str)

        def dragEnterEvent(self, e):  # noqa: N802
            if any(Path(u.toLocalFile()).suffix.lower() in IMAGE_EXT for u in e.mimeData().urls()):
                e.acceptProposedAction()

        def dropEvent(self, e):  # noqa: N802
            for u in e.mimeData().urls():
                if Path(u.toLocalFile()).suffix.lower() in IMAGE_EXT:
                    e.acceptProposedAction()
                    self.file_dropped.emit(u.toLocalFile())
                    return

    schemes.set_current(QSettings("RefGrade", "RefGrade").value("scheme", schemes.DEFAULT, type=str))
    win = _Window()
    win.setAcceptDrops(True)
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

    # start screen while no image is loaded (user 2026-10-17: pick the image in the UI)
    start = QWidget()
    start.setObjectName("start_screen")
    sl = QVBoxLayout(start)
    sl.addStretch(1)
    big = QPushButton("Open image…")
    big.setObjectName("start_open")
    big.setProperty("main", "ok")
    big.setMinimumWidth(240)
    big.clicked.connect(btn_open.trigger)
    sl.addWidget(big, 0, Qt.AlignmentFlag.AlignHCenter)
    hint = QLabel("Ctrl+O  ·  paste a screenshot with Ctrl+V  ·  or drop an image here")
    hint.setProperty("muted", True)
    sl.addWidget(hint, 0, Qt.AlignmentFlag.AlignHCenter)
    sl.addStretch(1)
    layout.addWidget(start)

    dlg_container = QWidget()
    QVBoxLayout(dlg_container).setContentsMargins(0, 0, 0, 0)
    dlg_container.hide()
    layout.addWidget(dlg_container)
    win.start_screen = start
    _apply_window_scheme(win)
    return win, lbl_file, btn_open, dlg_container


def main() -> None:
    _ensure_torch("--setup" in sys.argv[1:])
    args = [a for a in sys.argv[1:] if a != "--setup"]

    # Erst jetzt PyQt6 laden (dialog.py importiert es): torch ist bereits
    # geladen oder der Prozess endet über _ensure_torch (Neustart/Abbruch).
    import shutil
    from ref_grade.dialog import RefGradeDialog
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt6.QtGui import QImage, QKeySequence, QShortcut
    from PyQt6.QtCore import QEvent, QObject, QSettings, QSizeF, Qt, QTimer

    app = QApplication(sys.argv)
    app.setApplicationName("REFgrade")

    port = config.PORT
    python = config.python_exe()
    from ref_grade.version import __version__ as version
    settings = QSettings("RefGrade", "RefGrade")

    win, lbl_file, btn_open, dlg_container = build_window(f"REFgrade {version}")
    # one grading at a time; a new image replaces the picture and keeps the grading (like
    # Krita's UPDATE) — Open, drop, paste and the dialog's LOAD menu all go through open_image
    state = {"dlg": None, "path": None, "source": None, "tmp": None, "written": set(),
             "temps": []}

    def drop_temp(d) -> None:
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def show_start() -> None:
        state["paste"].setEnabled(True)
        dlg_container.hide()
        win.start_screen.show()
        win.setWindowTitle(f"REFgrade {version}")
        lbl_file.setText("")

    def close_dialog() -> None:
        dlg = state["dlg"]
        if dlg is not None:
            try:
                dlg.finished.disconnect()
            except TypeError:
                pass
            dlg_container.layout().removeWidget(dlg)
            dlg.close()
            dlg.setParent(None)        # out of the widget tree now, not only at deleteLater
            dlg.deleteLater()
        state.update(dlg=None, path=None, source=None)
        drop_temp(state["tmp"])
        state["tmp"] = None
        show_start()

    def read_into_worker(path: Path):
        """New worker session for `path` → (source, tmp, qimg) or None after a message."""
        if not path.is_file():
            QMessageBox.warning(win, "REFgrade", f"File not found:\n{path}")
            return None
        print(f"Bild: {path} ({path.stat().st_size // 1024} KB)")
        lbl_file.setText("  starting worker …")
        app.processEvents()
        tmp = Path(tempfile.mkdtemp(prefix="refgrade_gui_"))
        try:
            worker_process.ensure(port, python, config.CORE, timeout=30.0)
            source, icc = _read_source(path, tmp)
            resp = worker_client.call(port, "/load", source)
        except Exception as ex:  # noqa: BLE001 - boundary: show instead of crashing
            drop_temp(tmp)
            lbl_file.setText(f"  {state['path']}" if state["path"] else "")
            QMessageBox.critical(win, "REFgrade", f"Could not load {path.name}:\n{ex}")
            return None
        source["session"] = resp["session"]
        print(f"Source: {source['width']}x{source['height']}  Session: {source['session']}")
        pil_img, _icc = open_rgb8(path)
        pil_img.thumbnail((2048, 2048))
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888).copy()
        return source, tmp, qimg, icc

    def show_path(path: Path) -> None:
        win.setWindowTitle(f"REFgrade {version} — {path.name}")
        lbl_file.setText(f"  {path}")
        dlg = state["dlg"]
        if dlg is not None:
            dlg._btn_ok.setToolTip("Save — asks where (pasted screenshot)" if is_pasted(path)
                                   else f"Save as {default_save_path(path).name} next to the original")
        if not is_pasted(path):
            settings.setValue("open_folder", str(path.parent))

    def on_ok(raw_path: str, layer_name: str, clarity_info: dict | None = None) -> bool:
        """SAVE: <name>_rg.<ext> next to the original, no dialog; the grading stays open
        (Krita's SAVE writes a layer). A pasted screenshot asks where to save."""
        dlg, path, source = state["dlg"], state["path"], state["source"]
        out = default_save_path(path)
        if is_pasted(path):
            last = settings.value("saveas_folder", str(Path.home()))
            name, flt = QFileDialog.getSaveFileName(
                win, "Save graded image", str(Path(last) / "screenshot_rg.png"),
                ";;".join(_FILTER_EXT))
            if not name:
                dlg._set_status("Not saved")
                return False
            out = Path(name)
            if out.suffix.lower() not in SAVE_EXT:
                out = out.with_suffix(_FILTER_EXT.get(flt, ".png"))
            settings.setValue("saveas_folder", str(out.parent))
        elif out.exists() and str(out).lower() not in state["written"]:
            ans = QMessageBox.question(win, "REFgrade", f"{out.name} exists. Overwrite?")
            if ans != QMessageBox.StandardButton.Yes:
                dlg._set_status("Not saved")
                return False
        try:
            write_result(raw_path, source["width"], source["height"], out, state["icc"])
        except Exception as ex:  # noqa: BLE001
            dlg._set_status(f"Could not save: {ex}")
            return False
        state["written"].add(str(out).lower())
        dlg._set_status(f"Saved: {out}")
        print(f"Gespeichert: {out}")
        return False

    def on_update() -> bool:
        """The dialog's LOAD menu put the chosen file into source["path"]: swap the picture
        after the dialog's slot has returned; a failed load restores the old path."""
        old = state["source"]
        new = Path(old["path"])
        old["path"] = str(state["path"].resolve())       # stays right if the load fails
        QTimer.singleShot(0, lambda: open_image(new))
        return True

    def save_lut(folder: str, filename: str) -> str | None:
        p, _ = QFileDialog.getSaveFileName(
            win, "Save LUT", str(Path(folder) / filename), "LUT (*.cube)")
        return p or None

    class _NoEsc(QObject):
        """Esc must not throw the grading away (CANCEL still closes the image)."""
        def eventFilter(self, obj, e):  # noqa: N802
            return (e.type() == QEvent.Type.KeyPress and e.key() == Qt.Key.Key_Escape
                    and obj is state["dlg"])

    no_esc = _NoEsc(win)

    def open_image(path: Path) -> None:
        got = read_into_worker(path)
        if got is None:
            return
        source, tmp, qimg, icc = got
        dlg = state["dlg"]
        if dlg is not None:
            # keep the grading: new session + picture, same sliders/references (Krita's UPDATE)
            drop_temp(state["tmp"])
            state.update(path=path, source=source, tmp=tmp, icc=icc)
            dlg._source = source
            dlg._session_id = source["session"]
            dlg._orig_w, dlg._orig_h = source["width"], source["height"]
            dlg._worker_original_requested = False
            dlg._original_image = qimg
            dlg._last_preview = None
            dlg.view.set_image(qimg, QSizeF(source["width"], source["height"]))
            dlg.view.set_original(qimg)
            dlg._trigger_preview()
            dlg._set_status(f"Loaded {path.name} — grading kept")
            show_path(path)
            return
        dlg = RefGradeDialog(
            qimg, dlg_container,
            client=worker_client.call,
            source=source,
            save_lut=save_lut,
            on_ok=on_ok,
            on_update=on_update,
        )
        dlg.setWindowFlags(Qt.WindowType.Widget)   # embed: a QDialog is a top-level window otherwise
        dlg._btn_cancel.setToolTip("Close this image")
        dlg.finished.connect(lambda _r: close_dialog())
        dlg.installEventFilter(no_esc)
        dlg_container.layout().addWidget(dlg)
        state.update(dlg=dlg, path=path, source=source, tmp=tmp, icc=icc)
        state["paste"].setEnabled(False)
        win.start_screen.hide()
        dlg_container.show()
        dlg.show()
        dlg._scheme_combo.currentTextChanged.connect(lambda _n: _apply_window_scheme(win))
        show_path(path)

    def on_open() -> None:
        picked = _pick_file(win, settings.value("open_folder", None))
        if picked:
            open_image(picked)

    def on_paste() -> None:
        img = QApplication.clipboard().image()
        if img is None or img.isNull():
            return
        d = Path(tempfile.mkdtemp(prefix="refgrade_paste_"))
        state["temps"].append(d)
        p = d / "pasted.png"
        img.save(str(p), "PNG")
        open_image(p)

    btn_open.triggered.connect(on_open)
    win.file_dropped.connect(lambda f: open_image(Path(f)))
    # only on the start screen: with an image open the dialog owns Ctrl+V (two shortcuts with
    # the same key in one window are ambiguous and neither fires)
    paste = QShortcut(QKeySequence("Ctrl+V"), win)
    paste.activated.connect(on_paste)
    state["paste"] = paste

    win.showMaximized()          # user 2026-10-17: open maximized
    _apply_window_scheme(win)
    if args:
        QTimer.singleShot(0, lambda: open_image(Path(args[0])))

    try:
        app.exec()
    finally:
        worker_process.stop(port)
        drop_temp(state["tmp"])
        for d in state["temps"]:
            drop_temp(d)
        print("Worker gestoppt. Tschüss.")


if __name__ == "__main__":
    main()
