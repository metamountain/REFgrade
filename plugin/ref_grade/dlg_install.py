"""Install-Dialog für die RefGrade-Laufzeit (PyQt6, kein krita)."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from . import runtime


class _Worker(QThread):
    line = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, install_fn, flavor, cancelled):
        super().__init__()
        self._install_fn = install_fn
        self._flavor = flavor
        self._cancelled = cancelled

    def run(self) -> None:
        def _log(line: str) -> None:
            self.line.emit(line)
        try:
            result = self._install_fn(self._flavor, _log, self._cancelled)
            self.done.emit(result)
        except runtime.Cancelled:
            self.failed.emit("Cancelled")
        except Exception as e:
            self.failed.emit(str(e))


class InstallDialog(QDialog):
    def __init__(self, parent=None, detect_fn=runtime.detect, install_fn=runtime.install):
        super().__init__(parent)
        self.setWindowTitle("RefGrade - set up runtime")
        self.setModal(True)
        self._detect_fn = detect_fn
        self._install_fn = install_fn
        self._cancelled_flag = False
        self._worker: _Worker | None = None
        self.result_text: str | None = None
        self._detect_result: dict = {}
        self._gpu_was_enabled = False

        self._build_ui()
        self._run_detect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info = QLabel(
            "RefGrade runs its image engine in its own Python with PyTorch. "
            "This is a one-time download, installed to:\n"
            f"{runtime.runtime_dir()}\n"
            "Other Python installations are not touched."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.radio_gpu = QRadioButton()
        self.radio_cpu = QRadioButton("CPU only - smaller download, much slower")
        layout.addWidget(self.radio_gpu)
        layout.addWidget(self.radio_cpu)

        self.lbl_message = QLabel()
        self.lbl_message.setWordWrap(True)
        layout.addWidget(self.lbl_message)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setVisible(False)
        layout.addWidget(self.log_view)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.btn_install = QPushButton("Install")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_install.clicked.connect(self._on_install)
        self.btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self.btn_install)
        layout.addWidget(self.btn_cancel)

    def _run_detect(self) -> None:
        self._detect_result = self._detect_fn()
        flavor = self._detect_result.get("flavor", "cpu")
        gpu_name = self._detect_result.get("gpu_name")
        message = self._detect_result.get("message", "")

        if flavor != "cpu":
            self.radio_gpu.setText(
                f"GPU - {gpu_name} ({flavor}), large download (several GB)"
            )
            self.radio_gpu.setEnabled(True)
            self.radio_gpu.setChecked(True)
            self._gpu_was_enabled = True
        else:
            self.radio_gpu.setText("GPU - not available")
            self.radio_gpu.setEnabled(False)
            self.radio_cpu.setChecked(True)
            self._gpu_was_enabled = False

        if message:
            self.lbl_message.setText(message)
            self.lbl_message.setVisible(True)
        else:
            self.lbl_message.setVisible(False)

    def selected_flavor(self) -> str:
        if self.radio_gpu.isEnabled() and self.radio_gpu.isChecked():
            return self._detect_result.get("flavor", "cpu")
        return "cpu"

    def _on_install(self) -> None:
        self._cancelled_flag = False
        flavor = self.selected_flavor()
        self.radio_gpu.setEnabled(False)
        self.radio_cpu.setEnabled(False)
        self.btn_install.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_view.setVisible(True)
        self.progress.setVisible(True)

        self._worker = _Worker(self._install_fn, flavor, lambda: self._cancelled_flag)
        self._worker.line.connect(self._on_line)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_line(self, line: str) -> None:
        self.log_view.appendPlainText(line)

    def _on_done(self, result: str) -> None:
        self.result_text = result
        self.accept()

    def _on_failed(self, msg: str) -> None:
        self.progress.setVisible(False)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("red"))
        cursor.insertText("\n" + msg, fmt)
        if msg == "Cancelled":
            self.reject()
        else:
            self.btn_install.setText("Retry")
            self.btn_install.setEnabled(True)
            if self._gpu_was_enabled:
                self.radio_gpu.setEnabled(True)
            self.radio_cpu.setEnabled(True)

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._cancelled_flag = True
            self.btn_cancel.setEnabled(False)
        else:
            self.reject()

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._cancelled_flag = True
            self.btn_cancel.setEnabled(False)
            event.ignore()
        else:
            super().closeEvent(event)


def run_installer(parent=None, **kw) -> bool:
    dlg = InstallDialog(parent, **kw)
    return dlg.exec() == QDialog.DialogCode.Accepted
