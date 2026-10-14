"""LutMixin: LUT section building, folder, tree-popup, apply (K26)."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout,
                             QLabel, QPushButton, QSizePolicy, QStyle,
                             QStyledItemDelegate, QTreeView, QVBoxLayout,
                             QWidget)

from . import config, scale, style
from .widgets import SectionHeader, SliderRow, small_button
from .dlg_common import DEFAULT_ON
from .dlg_sections import _separator

_LUT_HOVER_MS = 120


class _LutDelegate(QStyledItemDelegate):
    """Tree view delegate: 2 px accent frame on hover, left bar on selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover_idx = None
        self._hover_row: int = -1  # kept for test compat

    def set_hover(self, index) -> None:
        if index == self._hover_idx:
            return
        self._hover_idx = index
        self._hover_row = index.row() if index and index.isValid() else -1
        view = self.parent()
        if view is not None and hasattr(view, "viewport"):
            view.viewport().update()

    def paint(self, painter, option, index) -> None:  # noqa: N802
        super().paint(painter, option, index)
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect.left(), option.rect.top(), 3,
                             option.rect.height(), QColor(style.ACCENT_HI))
        if index == self._hover_idx:
            rect = option.rect.adjusted(1, 1, -1, -1)
            painter.setPen(QPen(QColor(style.ACCENT_HI), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)


class _LutTreePopup(QWidget):
    """Popup with a QTreeView for navigating LUT folders."""

    lut_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._model = QStandardItemModel(self)
        self._model.setHorizontalHeaderLabels(["LUT"])
        self._tree = QTreeView(self)
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(14)
        self._tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.clicked.connect(self._on_clicked)
        self._delegate = _LutDelegate(self._tree)
        self._tree.setItemDelegate(self._delegate)
        self._tree.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._tree.installEventFilter(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._tree)
        self.setMinimumWidth(240)

    def _on_clicked(self, index) -> None:
        data = self._model.data(index, Qt.ItemDataRole.UserRole)
        if data and data != "__FOLDER__":
            self.lut_selected.emit(data)
            self.hide()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self._tree:
            if event.type() == event.Type.MouseMove:
                idx = self._tree.indexAt(event.position().toPoint())
                self._delegate.set_hover(idx if idx.isValid() else None)
            elif event.type() == event.Type.Leave:
                self._delegate.set_hover(None)
        return super().eventFilter(obj, event)

    def populate(self, folder: str) -> None:
        self._model.removeRows(0, self._model.rowCount())
        none_item = QStandardItem("— none —")
        none_item.setData("", Qt.ItemDataRole.UserRole)
        self._model.appendRow(none_item)
        try:
            root = Path(folder)
            files = sorted(
                (p for p in root.iterdir()
                 if p.is_file() and p.suffix.lower() == ".cube"),
                key=lambda p: p.name.lower(),
            )
            for p in files:
                item = QStandardItem(p.stem)
                item.setData(str(p), Qt.ItemDataRole.UserRole)
                self._model.appendRow(item)
            subdirs = sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: p.name.lower(),
            )
            for d in subdirs:
                folder_item = QStandardItem(d.name)
                folder_item.setEditable(False)
                folder_item.setData("__FOLDER__", Qt.ItemDataRole.UserRole)
                folder_item.setFlags(
                    folder_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self._model.appendRow(folder_item)
                sub_files = sorted(
                    (p for p in d.iterdir()
                     if p.is_file() and p.suffix.lower() == ".cube"),
                    key=lambda p: p.name.lower(),
                )
                for p in sub_files:
                    file_item = QStandardItem(p.stem)
                    file_item.setData(str(p), Qt.ItemDataRole.UserRole)
                    folder_item.appendRow(file_item)
        except OSError:
            pass
        self._tree.expandAll()

    def set_current(self, path: str) -> None:
        def find(idx):
            for row in range(self._model.rowCount(idx)):
                child = self._model.index(row, 0, idx)
                if self._model.data(child, Qt.ItemDataRole.UserRole) == path:
                    self._tree.setCurrentIndex(child)
                    self._tree.scrollTo(child)
                    return True
                if find(child):
                    return True
            return False
        find(self._model.index(0, 0, Qt.Index(-1, -1)))


class LutMixin:
    """Mixin for RefGradeDialog: LUT section, folder, tree-popup, apply."""

    def _build_lut_section(self) -> QWidget:
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
        # Tree-popup combo: button opens a QTreeView popup
        self._lut_combo = QPushButton("— none —")
        self._lut_combo.setObjectName("lut_combo")
        self._lut_combo.setFont(style.font("base"))
        self._lut_combo.setProperty("font_role", "base")
        self._lut_combo.setFixedHeight(scale.button_h(style.current_s()))
        self._lut_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lut_combo.clicked.connect(self._lut_toggle_popup)
        row.addWidget(self._lut_combo, 1)
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
        self._lut_popup: _LutTreePopup | None = None
        self._lut_current_path: str = ""
        return w

    def _lut_toggle_popup(self) -> None:
        """Open or close the LUT tree popup."""
        if self._lut_popup is not None and self._lut_popup.isVisible():
            self._lut_popup.hide()
            return
        if self._lut_popup is None:
            self._lut_popup = _LutTreePopup(self)
            self._lut_popup.lut_selected.connect(self._lut_on_selected)
        folder = self._settings.value("lut_folder", "") or str(config.LUTS)
        self._lut_popup.populate(folder)
        if self._lut_current_path:
            self._lut_popup.set_current(self._lut_current_path)
        btn_geo = self._lut_combo.geometry()
        popup_w = max(240, self._lut_combo.width())
        self._lut_popup.setFixedWidth(popup_w)
        self._lut_popup.adjustSize()
        pos = self._lut_combo.mapToGlobal(self._lut_combo.rect().bottomLeft())
        self._lut_popup.move(pos.x(), pos.y() + 2)
        self._lut_popup.show()
        self._lut_popup.raise_()

    def _lut_on_selected(self, path: str) -> None:
        """A LUT was selected in the tree popup."""
        self._lut_current_path = path
        name = Path(path).stem if path else "— none —"
        self._lut_combo.setText(name)
        self._lut_hover_path = None
        self._lut_hover_active = False
        header = self._lut_header
        if path:
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

    def _load_lut_folder(self) -> None:
        folder = self._settings.value("lut_folder", "")
        if not folder or not Path(folder).is_dir():
            krita_luts = config.lut_dir(Path(os.environ.get("LOCALAPPDATA", "")) / "kritarc")
            folder = str(krita_luts if krita_luts.is_dir() else config.LUTS)
        self._refresh_lut_list(folder)

    def _refresh_lut_list(self, folder: str) -> None:
        """Update the folder label and settings (popup is populated on open)."""
        self._lut_folder_label.setText(f"in: {Path(folder).name}")
        self._settings.setValue("lut_folder", folder)
        self._lut_dim_state()

    def _lut_data_at(self, row: int) -> str | None:
        """Compat: return the current path (flat model removed)."""
        return self._lut_current_path or None

    def _lut_dim_state(self) -> None:
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
        """No-op: popup is created lazily on first click."""
        self._lut_view_ready = True

    def _lut_view_hover(self, index) -> None:
        """Compat stub: hover is handled inside the popup."""
        pass

    def _lut_hover_timeout(self) -> None:
        """Compat stub: hover preview is handled inside the popup."""
        pass

    def _lut_view_leave(self) -> None:
        """Popup closed without selection: restore previous grading."""
        if self._lut_hover_active:
            self._lut_hover_path = None
            self._lut_hover_active = False
            if self._client is not None:
                self._schedule_preview()

    def _lut_folder_clicked(self) -> None:
        start = self._settings.value("lut_folder", "") or str(config.LUTS)
        if self._choose_folder is not None:
            folder = self._choose_folder(start)
        else:
            folder = QFileDialog.getExistingDirectory(self, "LUT folder", start)
        if folder:
            self._refresh_lut_list(folder)

    def _lut_combo_changed(self, index: int) -> None:
        """Compat stub: selection is handled by _lut_on_selected."""
        pass

    def _do_lut(self) -> None:
        if self._client is None or self._session_id is None or self._save_lut is None:
            return
        name = self._first_ref_name()
        excluded = [n for n in ("clarity", "texture", "dehaze")
                    if self._get_slider_value(n) != 0.0]
        suffix = f" ({', '.join(n.capitalize() for n in excluded)} not included)" if excluded else ""
        lut_folder = self._settings.value("lut_folder", "") or str(config.LUTS)
        save_dir = str(Path(lut_folder) / "RefGrade")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = self._save_lut(save_dir, f"{name}.cube{suffix}")
        if path is None:
            return
        self._lut_dir_last = str(Path(path).parent)
        vals = self._values()
        for n in ("clarity", "texture", "dehaze"):
            vals[n] = 0.0
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

    def _apply_cube(self, result: dict) -> None:
        fname = Path(result["path"]).name
        self._set_status(f"LUT · {fname}")
        lut_folder = self._settings.value("lut_folder", "") or str(config.LUTS)
        self._refresh_lut_list(lut_folder)
