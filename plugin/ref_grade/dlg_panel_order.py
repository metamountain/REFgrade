"""Panel order (K35a): chain order = panel order, drag & drop reorder in the scroll column.

Aus dialog.py ausgelagert (Modulgrenze 900 Zeilen, test_plugin_k27a).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from . import scale, style
from .dlg_common import CHAIN_ORDER


class PanelOrderMixin:
    """Mixin for RefGradeDialog: panel (= chain) order, drag & drop reorder, reset."""

    def set_chain_order(self, order: list[str]) -> None:
        """Set the chain order (permutation of CHAIN_ORDER); persist to QSettings and preview."""
        if set(order) != set(CHAIN_ORDER) or len(order) != len(CHAIN_ORDER):
            raise ValueError(f"chain order must be a permutation of {CHAIN_ORDER}, got {order!r}")
        # Rebuild panel order: chain sections in given order, color_replace at end
        self._panel_order = list(order) + ["color_replace"]
        self._chain_order = list(order)
        self._settings.setValue("chain_order", list(self._panel_order))
        self._apply_panel_order()
        self._schedule_preview()

    def _apply_panel_order(self) -> None:
        """Reorder the scroll layout widgets to match _panel_order."""
        self._scroll_keys = tuple(k for k in self._panel_order if k != "wheels")
        scroll_layout = self._scroll.layout()
        # Remove all items (widgets + stretches) from the layout
        while scroll_layout.count() > 0:
            item = scroll_layout.takeAt(0)
            if item.widget() is not None:
                pass  # widget stays parented, just removed from layout
        # Re-add in panel order (skip wheels — it lives in the wheel strip)
        for key in self._panel_order:
            if key == "wheels":
                continue
            w = self._scroll_widgets.get(key)
            if w is not None:
                scroll_layout.addWidget(w)
        scroll_layout.addStretch(1)

    def _on_section_drag_started(self, key: str) -> None:
        """A section header drag has started; show the insertion indicator."""
        self._drag_key = key
        self._drag_indicator = QWidget(self._scroll)
        # thin line like _separator: height from scale, no fixed pixel minimum
        self._drag_indicator.setFixedHeight(max(1, round(scale.row_h(style.current_s()) * 0.1)))
        self._drag_indicator.setStyleSheet(f"background-color: {style.TEXT};")
        self._drag_indicator.hide()

    def _on_section_drag_moved(self, key: str, pos) -> None:
        """Update the insertion indicator position while dragging."""
        if self._drag_indicator is None:
            return
        scroll_layout = self._scroll.layout()
        # Remove indicator from layout before recomputing position
        if scroll_layout.indexOf(self._drag_indicator) >= 0:
            scroll_layout.removeWidget(self._drag_indicator)
        # Find the insertion index based on mouse Y relative to scroll section widgets
        insert_idx = len(self._scroll_keys)
        for i, key2 in enumerate(self._scroll_keys):
            if key2 == key:
                continue
            w = self._scroll_widgets.get(key2)
            if w is None or not w.isVisible():
                continue
            w_top = w.mapToGlobal(w.rect().topLeft()).y()
            w_mid = w_top + w.height() // 2
            if pos.y() < w_mid:
                insert_idx = i
                break
        # Insert indicator into the scroll layout at the computed position
        scroll_layout.insertWidget(insert_idx, self._drag_indicator)
        self._drag_indicator.show()
        self._drag_indicator.adjustSize()

    def _on_section_drag_ended(self, key: str, was_drag: bool) -> None:
        """Drag ended: reorder the panel or do nothing (click)."""
        # Find the drop target BEFORE deleting the indicator: the first
        # scroll-section widget after the indicator that is not the dragged
        # section itself.
        drop_before_key = None
        if was_drag and self._drag_indicator is not None:
            scroll_layout = self._scroll.layout()
            ind_idx = scroll_layout.indexOf(self._drag_indicator)
            for j in range(ind_idx + 1, scroll_layout.count()):
                item = scroll_layout.itemAt(j)
                if item is None:
                    continue
                w = item.widget()
                if w is None:
                    continue
                for k2, w2 in self._scroll_widgets.items():
                    if w2 is w and k2 != key:
                        drop_before_key = k2
                        break
                if drop_before_key is not None:
                    break
        # Clean up the indicator
        if self._drag_indicator is not None:
            self._drag_indicator.deleteLater()
            self._drag_indicator = None
        if not was_drag:
            self._drag_key = None
            return
        # Build the new panel order: remove the dragged key, insert before the
        # drop target (or append if none was found).
        order = [k for k in self._panel_order if k != key]
        if drop_before_key is not None:
            insert_pos = order.index(drop_before_key)
        else:
            insert_pos = len(order)
        order.insert(insert_pos, key)
        self._panel_order = order
        self._chain_order = [k for k in order if k in CHAIN_ORDER]
        self._settings.setValue("chain_order", list(order))
        self._apply_panel_order()
        self._schedule_preview()
        self._drag_key = None

    def _on_reset_chain_order(self) -> None:
        """Context menu: reset to default panel/chain order."""
        self._panel_order = list(CHAIN_ORDER) + ["color_replace"]
        self._chain_order = list(CHAIN_ORDER)
        self._settings.setValue("chain_order", list(self._panel_order))
        self._apply_panel_order()
        self._schedule_preview()

    def _on_chain_order_clicked(self) -> None:
        """Toggle the chain-order popup next to the ⇅ button."""
        if self._chain_order_popup is not None and self._chain_order_popup.isVisible():
            self._chain_order_popup.close()
            self._chain_order_popup = None
            return
        from .dlg_chain import ChainOrderPopup
        popup = ChainOrderPopup(list(self._chain_order), self)
        popup.order_changed.connect(self.set_chain_order)
        pos = self._chain_order_btn.mapToGlobal(self._chain_order_btn.rect().bottomLeft())
        popup.move(pos.x(), pos.y())
        popup.show()
        popup.raise_()
        self._chain_order_popup = popup
