"""Save As: one file dialog (no folder step), written with Qt only (Krita has no numpy/PIL)."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules.setdefault("ref_grade", _pkg)

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", f"RefGradeTests_{os.getpid()}")

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402

W, H = 4, 3


class TestSaveAs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        img = QImage(W, H, QImage.Format.Format_RGB32)
        img.fill(QColor(10, 20, 30))
        self.dlg = RefGradeDialog(img, source={"width": W, "height": H,
                                               "path": str(Path(self.tmp.name) / "photo.png")})
        self.dlg._orig_w, self.dlg._orig_h = W, H

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def test_single_dialog_adds_suffix_and_remembers_folder(self):
        target = str(Path(self.tmp.name) / "out")
        with patch.object(QFileDialog, "getSaveFileName",
                          return_value=(target, "JPEG (*.jpg *.jpeg)")) as save, \
             patch.object(QFileDialog, "getExistingDirectory") as folder, \
             patch.object(type(self.dlg), "_do_ok") as ok:
            self.dlg._do_saveas()
        folder.assert_not_called()
        save.assert_called_once()
        self.assertTrue(save.call_args[0][2].endswith("photo_grad.png"))
        ok.assert_called_once()
        self.assertEqual(self.dlg._saveas_out, target + ".jpg")
        self.assertEqual(Path(self.dlg._settings.value("saveas_folder")), Path(self.tmp.name))

    def test_cancel_does_nothing(self):
        with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")), \
             patch.object(type(self.dlg), "_do_ok") as ok:
            self.dlg._do_saveas()
        ok.assert_not_called()

    def _write(self, depth: str, raw: bytes, name: str) -> QImage:
        rawp = Path(self.tmp.name) / "result.raw"
        rawp.write_bytes(raw)
        out = Path(self.tmp.name) / name
        self.dlg._source["depth"] = depth
        self.dlg._saveas_out = str(out)
        self.dlg._apply_apply({"raw": str(rawp)})
        self.assertIsNone(self.dlg._saveas_out)
        img = QImage(str(out))
        self.assertFalse(img.isNull(), f"{name} not written")
        return img

    def test_u8_bgra_written_with_correct_colours(self):
        px = bytes([30, 20, 200, 255])          # B, G, R, A
        img = self._write("U8", px * (W * H), "u8.png")
        c = img.pixelColor(1, 1)
        self.assertEqual((c.red(), c.green(), c.blue()), (200, 20, 30))

    def test_u16_bgra_written_with_correct_colours(self):
        px = struct.pack("<4H", 3000, 20000, 60000, 65535)   # B, G, R, A
        img = self._write("U16", px * (W * H), "u16.png")
        c = img.pixelColor(2, 1)
        self.assertAlmostEqual(c.redF(), 60000 / 65535, delta=0.01)
        self.assertAlmostEqual(c.greenF(), 20000 / 65535, delta=0.01)
        self.assertAlmostEqual(c.blueF(), 3000 / 65535, delta=0.01)

    def test_no_numpy_or_pil_in_saveas_path(self):
        src = (_PLUGIN_DIR / "dlg_worker.py").read_text(encoding="utf-8")
        start = src.index("def _apply_apply")
        body = src[start:src.index("clarity_info", start)]
        self.assertNotIn("import numpy", body)
        self.assertNotIn("from PIL", body)


if __name__ == "__main__":
    unittest.main()
