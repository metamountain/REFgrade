"""Debug: show _enforce_no_scroll internals at 1920x1080."""
import os, sys, types
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["REFGRADE_SETTINGS"] = "RefGradeDebug"

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from ref_grade.style import current_s
from ref_grade import scale
from ref_grade.dialog import RefGradeDialog

img = QImage(200, 150, QImage.Format.Format_RGB32)
img.fill(QColor(80, 120, 160))
dlg = RefGradeDialog(img)
dlg.resize(1920, 1080)
dlg.show()
app.processEvents()

# Re-expand all sections to see the layout minimum with everything expanded
for key in dlg._scroll_keys:
    if key != "reference":
        dlg._set_section_collapsed(key, False)
app.processEvents()

lay = dlg.panel.layout()
lay.activate()
ms = lay.minimumSize()
print(f"panel.height() = {dlg.panel.height()}")
print(f"panel.layout().minimumSize() (all expanded) = {ms.width()}x{ms.height()}")
print(f"fit = {ms.height() <= dlg.panel.height()}")
print()

# Check each section's contribution
for key in dlg._scroll_keys:
    hdr, rows = dlg.sections[key]
    content = dlg._contents.get(key)
    if content:
        # Get the minimum height of the content widget
        content_lay = content.layout()
        if content_lay:
            content_lay.activate()
            cm = content_lay.minimumSize()
            print(f"  {key:12s} header_min={hdr.minimumHeight():3d} content_min={cm.height():3d} "
                  f"content_total={hdr.minimumHeight() + cm.height():3d}")
        else:
            print(f"  {key:12s} header_min={hdr.minimumHeight():3d} content_min={content.minimumHeight():3d} "
                  f"content_total={hdr.minimumHeight() + content.minimumHeight():3d}")
    else:
        print(f"  {key:12s} (no content)")

# Check the scroll area's minimum size
print(f"\nscroll area min height = {dlg._scroll.minimumHeight()}")
print(f"scroll area sizeHint height = {dlg._scroll.sizeHint().height()}")
print(f"scroll area layout min height = {dlg._scroll.layout().minimumSize().height()}")

# Check each item in the scroll layout
scroll_lay = dlg._scroll.layout()
total_inner_min = 0
for i in range(scroll_lay.count()):
    item = scroll_lay.itemAt(i)
    w = item.widget()
    if w:
        inner_lay = w.layout()
        inner_min = inner_lay.minimumSize().height() if inner_lay else 0
        total_inner_min += inner_min
        print(f"  scroll item {i}: obj={w.objectName()} min_h={w.minimumHeight()} sizeHint_h={w.sizeHint().height()} "
              f"inner_lay_min_h={inner_min} visible={w.isVisible()}")
    else:
        print(f"  scroll item {i}: (spacer/stretch)")
print(f"  total inner min = {total_inner_min}")
print(f"  + spacing ({scroll_lay.count()} * {scroll_lay.spacing()}) = {total_inner_min + scroll_lay.count() * scroll_lay.spacing()}")

# Check the panel layout
panel_lay = dlg.panel.layout()
print(f"\npanel layout spacing = {panel_lay.spacing()}")
print(f"panel layout contents margins = {panel_lay.contentsMargins()}")
for i in range(panel_lay.count()):
    item = panel_lay.itemAt(i)
    w = item.widget()
    if w:
        print(f"  panel item {i}: {w.__class__.__name__} obj={w.objectName()} min_h={w.minimumHeight()} sizeHint_h={w.sizeHint().height()}")
    else:
        lay = item.layout()
        if lay:
            print(f"  panel item {i}: (layout) min_h={lay.minimumSize().height()}")
        else:
            print(f"  panel item {i}: (spacer) {item.sizeHint().height()}")

# Check each section's contribution
for key in dlg._scroll_keys:
    hdr, rows = dlg.sections[key]
    content = dlg._contents.get(key)
    if content:
        print(f"  {key:12s} header={hdr.height():3d} content={'visible' if content.isVisible() else 'HIDDEN':7s} "
              f"collapsed={hdr.is_collapsed()} content_min_h={content.minimumHeight()}")
    else:
        print(f"  {key:12s} (no content)")

print()
print(f"row_h = {scale.row_h(current_s())}")
print(f"base_px = {scale.base_px(current_s())}")
print(f"s = {current_s():.1f}")
print(f"unit = {scale.unit(1920, 1080):.1f}")

# Check curves editor
from ref_grade.curves_ui import CurvesEditor
ce = dlg._contents.get("curves")
if ce:
    # Find the CurvesEditor inside
    for child in ce.findChildren(CurvesEditor):
        print(f"\nCurvesEditor: w={child.width()} h={child.height()} "
              f"minH={child.minimumHeight()} minHFW={child.minimumHeightForWidth(child.width())} "
              f"hFW={child.heightForWidth(child.width())} hasHFW={child.hasHeightForWidth()}")
