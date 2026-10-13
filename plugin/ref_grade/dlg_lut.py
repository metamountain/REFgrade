"""LutMixin: LUT section building, folder, combo, apply (K26)."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QStyledItemDelegate,
                             QVBoxLayout, QWidget)

from . import config, scale, style
from .widgets import SectionHeader, SliderRow, small_button
from .dlg_common import DEFAULT_ON
from .dlg_sections import _separator

_LUT_HOVER_MS = 120


class _LutDelegate(QStyledItemDelegate):
    """Combo view delegate: 2 px frame on the hovered entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover_row: int = -1

    def set_hover(self, row: int) -> None:
        """Set the hovered row and repaint the view."""
        if row == self._hover_row:
            return
        self._hover_row = row
        view = self.parent()
        if view is not None and hasattr(view, "viewport"):
            view.viewport().update()

    def paint(self, painter, option, index) -> None:  # noqa: N802
        super().paint(painter, option, index)
        if index.row() == self._hover_row:
            rect = option.rect.adjusted(1, 1, -1, -1)
            painter.setPen(QPen(QColor(style.ACCENT_HI), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

class LutMixin:
    """Mixin for RefGradeDialog: LUT section, folder, combo, apply."""

    def _build_lut_section(self) -> QWidget:
        """Section "LUT": header + content (Amount, FOLDER…, SAVE, combo, folder label)."""
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 2)
        box.setSpacing(1)
        header = SectionHeader("lut", "LUT", on="lut" in DEFAULT_ON)
        box.addWidget(header)
        content = QWidget()
        cbox = QVBoxLayout(content)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(2)
        amount = SliderRow("amount_lut", 0.0, 1.0, 1.0, 0.01, "Amount")
        amount.setObjectName("amount_lut")
        cbox.addWidget(amount)
        header.set_amount_slider(amount)
        cbox.addWidget(_separator())
        row = QHBoxLayout()
        row.setSpacing(4)
        self._lut_folder_btn = QPushButton("FOLDER…")
        self._lut_folder_btn.setObjectName("lut_folder")
        self._lut_folder_btn.setProperty("small", True)
        self._lut_folder_btn.setFont(style.font("base"))
        self._lut_folder_btn.setProperty("font_role", "base")
        self._lut_folder_btn.setFixedHeight(scale.button_h(style.current_s()))
        self._lut_folder_btn.setToolTip("Choose the folder containing .cube LUT files")
        self._lut_folder_btn.clicked.connect(self._lut_folder_clicked)
        row.addWidget(self._lut_folder_btn)
        self._lut_save_btn = QPushButton("SAVE")
        self._lut_save_btn.setObjectName("lut_save")
        self._lut_save_btn.setProperty("small", True)
        self._lut_save_btn.setFont(style.font("base"))
        self._lut_save_btn.setProperty("font_role", "base")
        self._lut_save_btn.setFixedHeight(scale.button_h(style.current_s()))
        self._lut_save_btn.setToolTip("Save the current grading as a .cube LUT")
        self._lut_save_btn.clicked.connect(self._do_lut)
        row.addWidget(self._lut_save_btn)
        self._lut_combo = QComboBox()
        self._lut_combo.setObjectName("lut_combo")
        self._lut_combo.setFont(style.font("base"))
        self._lut_combo.setProperty("font_role", "base")
        self._lut_combo.setFixedHeight(scale.button_h(style.current_s()))
        self._lut_combo.currentIndexChanged.connect(self._lut_combo_changed)
        row.addWidget(self._lut_combo, 1)
        # K25i: folder name as italic text in the same row after the dropdown
        self._lut_folder_label = QLabel("")
        _f = style.font("small")
        _f.setItalic(True)
        self._lut_folder_label.setFont(_f)
        self._lut_folder_label.setProperty("font_role", "small")
        self._lut_folder_label.setProperty("muted", True)
        self._lut_folder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._lut_folder_label)
        cbox.addLayout(row)
        box.addWidget(content)
        self._lut_header = header
        self._lut_content = content
        self._lut_hover_path: str | None = None
        self._lut_hover_active = False
        self._lut_view_ready = False
        return w


    def _load_lut_folder(self) -> None:
        """Load the LUT folder from QSettings (or default) and fill the combo."""
        folder = self._settings.value("lut_folder", "")
        if not folder or not Path(folder).is_dir():
            # default (user 2026-10-08): Krita's resource folder \luts, else the project's
            krita_luts = config.lut_dir(Path(os.environ.get("LOCALAPPDATA", "")) / "kritarc")
            folder = str(krita_luts if krita_luts.is_dir() else config.LUTS)
        self._refresh_lut_list(folder)


    def _refresh_lut_list(self, folder: str) -> None:
        """Fill the combo with — none —, .cube files in `folder`, then each direct
        subfolder as a non-selectable uppercase header followed by its .cube files."""
        self._lut_combo.blockSignals(True)
        self._lut_combo.clear()
        self._lut_combo.addItem("— none —", "")
        try:
            root = Path(folder)
            files = sorted(
                (p for p in root.iterdir()
                 if p.is_file() and p.suffix.lower() == ".cube"),
                key=lambda p: p.name.lower(),
            )
            for p in files:
                self._lut_combo.addItem(p.stem, str(p))
            # Subfolders as categories
            subdirs = sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
            for d in subdirs:
                header_text = f"— {d.name.upper()} —"
                self._lut_combo.addItem(header_text, "__HEADER__")
                sub_files = sorted(
                    (p for p in d.iterdir()
                     if p.is_file() and p.suffix.lower() == ".cube"),
                    key=lambda p: p.name.lower(),
                )
                for p in sub_files:
                    self._lut_combo.addItem(p.stem, str(p))
        except OSError:
            pass
        self._lut_combo.setCurrentIndex(0)
        self._lut_combo.blockSignals(False)
        self._lut_folder_label.setText(f"in: {Path(folder).name}")
        self._settings.setValue("lut_folder", folder)
        self._lut_dim_state()
        if self._lut_view_ready:            # K27e: view is set up in showEvent (offscreen-safe)
            self._setup_lut_combo_view()


    def _lut_dim_state(self) -> None:
        """Dim combo and button when the LUT section is OFF."""
        on = self._lut_header.is_on()
        self._lut_combo.setProperty("dim", not on)
        self._lut_folder_btn.setProperty("dim", not on)
        self._lut_save_btn.setProperty("dim", not on)
        self._lut_combo.style().unpolish(self._lut_combo)
        self._lut_combo.style().polish(self._lut_combo)
        self._lut_folder_btn.style().unpolish(self._lut_folder_btn)
        self._lut_folder_btn.style().polish(self._lut_folder_btn)
        self._lut_save_btn.style().unpolish(self._lut_save_btn)
        self._lut_save_btn.style().polish(self._lut_save_btn)


    def _setup_lut_combo_view(self) -> None:
        """Install mouse-tracking, hover-preview, and delegate on the combo's popup view.
        Deferred to showEvent: combo.view() blocks during __init__ in offscreen mode."""
        if self._lut_view_ready:
            return
        self._lut_view_ready = True
        view = self._lut_combo.view()
        if view is None:
            return
        view.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        view.installEventFilter(self)
        if not hasattr(self, "_lut_delegate"):
            self._lut_delegate = _LutDelegate(view)
        view.setItemDelegate(self._lut_delegate)


    def _lut_view_hover(self, index: int) -> None:
        """Hover over a LUT entry: start 120 ms debounce; on timeout send a preview with
        that LUT. Headers and — none — are ignored."""
        # Update the delegate frame immediately
        if hasattr(self, "_lut_delegate"):
            self._lut_delegate.set_hover(index)
        if self._client is None or self._session_id is None:
            return
        if index < 0 or index >= self._lut_combo.count():
            return
        data = self._lut_combo.itemData(index)
        if data is None or data == "" or data == "__HEADER__":
            self._lut_hover_pending = -1
            return
        if data == self._lut_hover_path:
            return
        # Debounce: store the pending index, restart the timer
        self._lut_hover_pending = index
        self._lut_hover_timer.start(_LUT_HOVER_MS)

    def _lut_hover_timeout(self) -> None:
        """Timer fired: send the preview for the pending LUT entry."""
        idx = self._lut_hover_pending
        self._lut_hover_pending = -1
        if idx < 0 or idx >= self._lut_combo.count():
            return
        data = self._lut_combo.itemData(idx)
        if data is None or data == "" or data == "__HEADER__":
            return
        if data == self._lut_hover_path:
            return
        self._lut_hover_path = data
        self._lut_hover_active = True
        m = self.mode()
        m["weights"]["lut"] = m["weights"].get("lut", 1.0)
        m["lut"] = data
        payload = {
            "session": self._session_id,
            "references": self._references(),
            "method": self._method(),
            "values": self._values(),
            "max_side": self._max_side(),
            "mix": m,
        }
        self._run_task("/preview", payload, self._next_req())


    def _lut_view_leave(self) -> None:
        """Leave the combo popup without selecting: restore the previous grading."""
        self._lut_hover_timer.stop()
        self._lut_hover_pending = -1
        if hasattr(self, "_lut_delegate"):
            self._lut_delegate.set_hover(-1)
        if self._lut_hover_active:
            self._lut_hover_path = None
            self._lut_hover_active = False
            if self._client is not None:
                self._schedule_preview()


    def _lut_view_wheel(self, event) -> None:
        """Wheel on the combo popup: move selection, skipping headers."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1 if delta > 0 else -1
        cur = self._lut_combo.currentIndex()
        n = self._lut_combo.count()
        idx = cur
        for _ in range(n):
            idx += step
            if idx < 0 or idx >= n:
                break
            if self._lut_combo.itemData(idx) != "__HEADER__":
                self._lut_combo.setCurrentIndex(idx)
                break


    def _lut_folder_clicked(self) -> None:
        """FOLDER…: call the injected choose_folder, refresh the list."""
        start = self._settings.value("lut_folder", "") or str(config.LUTS)
        if self._choose_folder is not None:
            folder = self._choose_folder(start)
        else:
            folder = QFileDialog.getExistingDirectory(self, "LUT folder", start)
        if folder:
            self._refresh_lut_list(folder)
            self._lut_combo.setCurrentIndex(0)


    def _lut_combo_changed(self, index: int) -> None:
        """Combo selection changed: ON + preview for a LUT, OFF for — none —."""
        self._lut_hover_path = None
        self._lut_hover_active = False
        header = self._lut_header
        if index > 0:
            if not header.is_on():
                header.on.setChecked(True)
        else:
            if header.is_on():
                header.on.setChecked(False)
        self._lut_dim_state()
        if self._client is not None:
            self._schedule_preview()
        else:
            self._show_mode()


    def _do_lut(self) -> None:
        """SAVE: export the current grading as a .cube LUT in <LUT folder>/RefGrade."""
        if self._client is None or self._session_id is None or self._save_lut is None:
            return
        name = self._first_ref_name()
        excluded = [n for n in ("clarity", "texture", "dehaze")
                    if self._get_slider_value(n) != 0.0]
        suffix = f" ({', '.join(n.capitalize() for n in excluded)} not included)" if excluded else ""
        lut_folder = self._settings.value("lut_folder", "") or str(config.LUTS)
        save_dir = str(Path(lut_folder) / "RefGrade")
        path = self._save_lut(save_dir, f"{name}.cube{suffix}")
        if path is None:
            return
        self._lut_dir_last = str(Path(path).parent)
        vals = self._values()
        for n in ("clarity", "texture", "dehaze"):
            vals[n] = 0.0  # spatial effects are not in the LUT
        payload = {
            "session": self._session_id,
            "references": self._references(),
            "method": self._method(),
            "values": vals,
            "dir": self._lut_dir_last,
            "mix": self.mode(),
        }
        req_id = self._next_req()
        self._run_task("/cube", payload, req_id)

    # ── OK ────────────────────────────────────────────────────────────────────

    def _apply_cube(self, result: dict) -> None:
        """Handle /cube response: re-read the LUT list (new entry appears, not selected)."""
        fname = Path(result["path"]).name
        self._set_status(f"LUT · {fname}")
        lut_folder = self._settings.value("lut_folder", "") or str(config.LUTS)
        self._refresh_lut_list(lut_folder)

