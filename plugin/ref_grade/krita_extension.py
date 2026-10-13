"""Krita side of REF Grade: menu entry Filter → RefGrade, opens the wired window.

Starts the worker, reads the active layer, loads the session, and opens the dialog with
client / source / save_lut / on_ok. On Krita exit the worker is stopped.
"""

from __future__ import annotations

import os
from pathlib import Path

from krita import Extension, Krita
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from . import config, dlg_install, krita_bridge, runtime, worker_client, worker_process
from .dialog import RefGradeDialog

MAX_SIDE = 4096          # Document.thumbnail returns a null image if the size is too big


class RefGradeExtension(Extension):
    """Menu entry and window."""

    def __init__(self, parent):
        super().__init__(parent)
        self._port = config.PORT
        self._worker_started = False

    def setup(self) -> None:
        Krita.instance().notifier().applicationClosing.connect(self._on_app_closing)

    def createActions(self, window) -> None:  # noqa: N802 (Krita API)
        # "Filter" is the menu name in krita5.xmlgui (v6.0.4, line 359); Window.h:92
        action = window.createAction("ref_grade_open", "RefGrade", "Filter")
        action.triggered.connect(self._open)

    def _on_app_closing(self) -> None:
        if self._worker_started:
            worker_process.stop(self._port)
            self._worker_started = False

    def _open(self) -> None:
        app = Krita.instance()
        parent = app.activeWindow().qwindow() if app.activeWindow() else None   # Window.h:39
        doc = app.activeDocument()
        if doc is None:
            QMessageBox.information(parent, "RefGrade", "No image open.")
            return

        # 1. Ensure worker
        python = config.python_exe()
        if os.environ.get("REFGRADE_PYTHON"):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                ok = runtime.python_ok(python)
            finally:
                QApplication.restoreOverrideCursor()
            if not ok:
                QMessageBox.critical(parent, "RefGrade",
                    f"REFGRADE_PYTHON = {python}\ncannot import torch, numpy, kornia and PIL.\n"
                    "Fix that Python or remove the variable to use the built-in runtime.")
                return
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                need_install = python is None or not runtime.python_ok(python)
            finally:
                QApplication.restoreOverrideCursor()
            if need_install:
                if not dlg_install.run_installer(parent):
                    return
                runtime.reset_cache()
                python = config.python_exe()
        try:
            worker_process.ensure(self._port, python, config.CORE, timeout=30.0)
            self._worker_started = True
        except RuntimeError as e:
            QMessageBox.critical(parent, "RefGrade", f"Worker failed to start:\n{e}")
            return

        # 2. Read active layer
        tmp = config.temp_dir()
        source = krita_bridge.read_active(tmp)
        if isinstance(source, str):
            QMessageBox.warning(parent, "RefGrade", source)
            return

        # 3. Load session
        try:
            resp = worker_client.call(self._port, "/load", source)
        except worker_client.WorkerError as e:
            QMessageBox.critical(parent, "RefGrade", f"Load failed: {e.message}")
            return
        source["session"] = resp["session"]

        # 4. Get document thumbnail for the dialog
        w, h = doc.width(), doc.height()
        s = min(1.0, MAX_SIDE / max(w, h))
        image = doc.thumbnail(max(1, round(w * s)), max(1, round(h * s)))   # Document.h:717

        # 5. Create callbacks
        def save_lut(folder: str, filename: str) -> str | None:
            path, _ = QFileDialog.getSaveFileName(
                parent, "Save LUT", str(Path(folder) / filename),
                "LUT (*.cube)")
            return path or None

        def on_ok(raw_path: str, layer_name: str, clarity_info: dict | None = None) -> None:
            krita_bridge.write_layer(source, Path(raw_path), layer_name, clarity_info)

        def on_update() -> None:
            """UPDATE: re-read the active layer and load a new session."""
            nonlocal source
            tmp = config.temp_dir()
            new_source = krita_bridge.read_active(tmp)
            if isinstance(new_source, str):
                return
            try:
                resp = worker_client.call(self._port, "/load", new_source)
            except worker_client.WorkerError:
                return
            new_source["session"] = resp["session"]
            source = new_source
            dlg._session_id = resp["session"]
            dlg._source = new_source
            dlg._orig_w = new_source.get("width", 0)
            dlg._orig_h = new_source.get("height", 0)

        # 6. Open dialog
        dlg = RefGradeDialog(
            image, parent,
            client=worker_client.call,
            source=source,
            save_lut=save_lut,
            on_ok=on_ok,
            on_update=on_update,
        )
        dlg.show_screen()            # frameless over the free screen area; F11 = fullscreen
        dlg.exec()
