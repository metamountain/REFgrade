"""Kettenreihenfolge (K35a): Popup-Liste der 8 Kettenglieder, per Drag & Drop umsortierbar.

Aus dialog.py ausgelagert (Modulgrenze 800 Zeilen, test_plugin_k27a).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QListWidget, QPushButton, QVBoxLayout, QWidget

from . import scale, style
from .dlg_common import CHAIN_ORDER


class ChainOrderPopup(QWidget):
    """Draggable list of the 8 chain-order sections; emits the new order on every move."""

    order_changed = pyqtSignal(list)

    def __init__(self, order: list[str], parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFixedWidth(round(scale.row_h(style.current_s()) * 9))   # aus scale, keine feste Pixelzahl
        self.setStyleSheet(
            f"background: {style.BLACK}; border: 1px solid {style.LINE};"
            f"QListWidget {{ background: transparent; border: none; color: {style.TEXT}; }}"
            f"QListWidget::item {{ padding: 4px 8px; color: {style.TEXT}; }}"
            f"QListWidget::item:selected {{ background: {style.LINE_STRONG}; color: {style.TEXT}; }}"
            f"QListWidget::item:hover {{ background: {style.LINE}; color: {style.TEXT}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._list = QListWidget()
        self._list.setObjectName("chain_order_list")
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setFont(style.font("base"))
        self._list.setFixedHeight(scale.row_h(style.current_s()) * len(order))
        for item in order:
            self._list.addItem(item.upper())
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("chain_order_reset")
        self._reset_btn.setFont(style.font("small"))
        self._reset_btn.setToolTip("Back to the standard order: REFERENCE, COLOR, TONE, RGB, CURVES, WHEELS, HSL, LUT.")
        self._reset_btn.clicked.connect(self._on_reset)
        layout.addWidget(self._reset_btn)

    def _on_rows_moved(self, *args) -> None:
        new_order = [self._list.item(i).text().lower() for i in range(self._list.count())]
        self.order_changed.emit(new_order)

    def _on_reset(self) -> None:
        self._list.clear()
        for item in CHAIN_ORDER:
            self._list.addItem(item.upper())
        self.order_changed.emit(list(CHAIN_ORDER))

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)
