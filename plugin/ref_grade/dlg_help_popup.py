"""?-Hilfe: Popup mit Abschnittstext (aus dialog.py ausgelagert, Modulgrenze 800 Zeilen)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

from . import style
from .dlg_common import load_auto_help


def _help_html(section_key: str, text: str) -> str:
    """Format SECTION_HELP text as HTML: bold headings, monospace formulas, muted body."""
    import re, html as _html
    title = section_key.upper()
    lines = text.split("\n")
    parts = [f'<div style="color:{style.TEXT};font-weight:bold;font-size:1.2em;">{title}</div>']
    for line in lines:
        if not line.strip():
            parts.append("<br>")
            continue
        # Bold headings: **What it does**, **The math**, **Tips & recipes**
        if line.startswith("**") and line.endswith("**"):
            heading = _html.escape(line[2:-2])
            parts.append(f'<div style="color:{style.TEXT};font-weight:bold;margin-top:8px;">{heading}</div>')
            continue
        line = _html.escape(line)
        line = re.sub(
            r'`([^`]+)`',
            lambda m: f'<span style="font-family:monospace;color:{style.TEXT};">{m.group(1)}</span>',
            line,
        )
        # List items
        if line.startswith("- "):
            parts.append(f'<div style="color:{style.MUTED};padding-left:12px;">{line}</div>')
        else:
            parts.append(f'<div style="color:{style.MUTED};">{line}</div>')
    # Common footer
    parts.append(
        f'<div style="color:{style.MUTED};margin-top:6px;">'
        f'<span style="color:{style.TEXT};">Amount</span> — how strongly the section acts<br>'
        f'<span style="color:{style.TEXT};">ON/OFF</span> — section on/off<br>'
        f'<span style="color:{style.TEXT};">↺</span> — reset section<br>'
        f"Sections run as a chain, top to bottom (order: ⇅ next to MASTER); "
        f"<span style=\"color:{style.TEXT};\">MASTER</span> blends the whole chain."
        f"</div>"
    )
    return "".join(parts)


class HelpPopup(QWidget):
    """Frameless popup showing section help in a scroll area; closes on Esc or click outside."""

    closed = pyqtSignal(str)  # section_key

    def __init__(self, section_key: str, width: int, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup)
        self._section_key = section_key
        self.setFixedWidth(width)
        self.setStyleSheet(
            f"background: {style.BLACK}; border: 1px solid {style.LINE};"
        )
        # Max height: 70 % of the parent dialog
        max_h = int(parent.height() * 0.7) if parent else 600
        self.setFixedHeight(max_h)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{ width: 8px; background: {style.BLACK}; }}"
            f"QScrollBar::handle:vertical {{ background: {style.LINE_STRONG}; min-height: 30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setWordWrap(True)
        self._label.setFont(style.font("base"))
        from .ui_help import SECTION_HELP, controls_table
        text = SECTION_HELP.get(section_key, "")
        ctrl = controls_table(section_key)
        if ctrl:
            text = f"{text}\n{ctrl}" if text else ctrl
        auto_text = load_auto_help().get(section_key)   # K30c: append the "Auto" paragraph
        if auto_text:
            text = f"{text}\n{auto_text}" if text else auto_text
        self._label.setText(_help_html(section_key, text))
        self._scroll.setWidget(self._label)
        layout.addWidget(self._scroll)

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    def hideEvent(self, e) -> None:  # noqa: N802
        super().hideEvent(e)
        self.closed.emit(self._section_key)
