"""Offscreen screenshots of the RefGrade window (layout, no worker) at several screen sizes.

    python tools/ui_shots.py            (from core\\)  →  core\\reports\\ui_<land|port>_<w>x<h>.png

Prints the measured geometry (column, image area, tile, histogram) for each size.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "core" / "reports"
pkg = types.ModuleType("ref_grade")
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules["ref_grade"] = pkg

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
from ref_grade.dialog import RefGradeDialog  # noqa: E402

OUT.mkdir(exist_ok=True)
POOL = ROOT / "bench" / "pool"
for name, img in (("land", POOL / "07.jpg"), ("port", POOL / "10.jpg")):
    for w, h in ((1366, 720), (1920, 1032), (1920, 1152), (2560, 1392)):
        d = RefGradeDialog(QImage(str(img)))
        if "--on" in sys.argv:                       # every section ON (to judge tracks/wheels)
            for header, _rows in d.sections.values():
                header.on.setChecked(True)
        d.resize(w, h)
        d.show()
        app.processEvents()
        d.resize(w, h)
        app.processEvents()
        d.grab().save(str(OUT / f"ui_{name}_{w}x{h}{'_on' if '--on' in sys.argv else ''}.png"))
        t = d.slots.tiles[0]
        print(f"{name} {w}x{h}: column {d.panel.width()}x{d.panel.height()} "
              f"image {d.view.width()}x{d.view.height()} "
              f"mode {d._effective_mode} "
              f"tile {t.width()}x{t.height()} histogram_bottom "
              f"{d.histogram.mapTo(d, QPoint(0, d.histogram.height())).y()} "
              f"image_bottom {d.view.mapTo(d, QPoint(0, d.view.height())).y()} "
              f"histogram_in_column {d.panel.isAncestorOf(d.histogram)}")
        d.close()
