"""Generate a screenshot of the Color Replace section in the RefGrade dialog."""
import os, sys, types, time
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from ref_grade.dialog import RefGradeDialog

app = QApplication.instance() or QApplication(sys.argv)
QSettings("RefGrade", "RefGradeTests").clear()

img = QImage(800, 600, QImage.Format.Format_RGB32)
img.fill(QColor(120, 80, 60))

dlg = RefGradeDialog(img)
dlg.resize(1920, 1080)
dlg.show()

deadline = time.time() + 5
while time.time() < deadline:
    app.processEvents()
    if dlg.width() >= 1900:
        break
    time.sleep(0.01)
app.processEvents()

cr = dlg._cr_section
cr.header.on.setChecked(True)
app.processEvents()

for i in range(3):
    cr._add_instance()
    app.processEvents()

cr._instances[0].src_wheel.set_point(0.5, 0.0)
cr._instances[0].tgt_wheel.set_point(-0.4, 0.3)
cr._instances[0].intensity.set_value(0.8)
cr._instances[0].range.set_value(45.0)
cr._instances[0].src_bright.set_value(-0.3)
cr._instances[0].tgt_bright.set_value(0.2)
app.processEvents()

time.sleep(0.3)
app.processEvents()

out = Path(__file__).resolve().parents[2] / "reports" / "shots" / "cr_screenshot.png"
out.parent.mkdir(parents=True, exist_ok=True)
dlg.grab().save(str(out))
print(f"Saved: {out}")
dlg.close()
