"""Standalone LOAD/SAVE (user 2026-10-17): start without an image, SAVE writes <name>_rg next
to the original and keeps the grading open, a pasted screenshot is never saved into temp."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _ROOT / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

_spec = importlib.util.spec_from_file_location("refgrade_gui", _ROOT / "standalone" / "gui.py")
gui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gui)

from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ref_grade.dialog import RefGradeDialog  # noqa: E402


class TestNames(unittest.TestCase):
    def test_default_save_path(self):
        self.assertEqual(gui.default_save_path(Path("C:/x/foto.jpg")), Path("C:/x/foto_rg.jpg"))
        self.assertEqual(gui.default_save_path(Path("C:/x/scan.TIF")), Path("C:/x/scan_rg.tif"))
        self.assertEqual(gui.default_save_path(Path("C:/x/pic.webp")), Path("C:/x/pic_rg.png"))
        self.assertEqual(gui.default_save_path(Path("C:/x/pic.bmp")), Path("C:/x/pic_rg.png"))

    def test_is_pasted(self):
        self.assertTrue(gui.is_pasted(Path("C:/Temp/refgrade_paste_ab12/pasted.png")))
        self.assertFalse(gui.is_pasted(Path("C:/Bilder/pasted.png")))

    def test_no_forced_file_dialog_at_start(self):
        src = (_ROOT / "standalone" / "gui.py").read_text(encoding="utf-8")
        main = src[src.index("def main("):]
        self.assertNotIn("_pick_file()", main)
        self.assertIn("start_screen", main)


class TestImageIO(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(dir=_ROOT / "tmp"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_exif_orientation_applied(self):
        from PIL import Image
        img = Image.new("RGB", (40, 20), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6                      # rotate 90° CW on display
        img.save(self.tmp / "r.jpg", exif=exif.tobytes())
        out, _ = gui.open_rgb8(self.tmp / "r.jpg")
        self.assertEqual(out.size, (20, 40))

    def test_16bit_grey_scaled_not_clipped(self):
        import numpy as np
        from PIL import Image
        a = np.full((8, 8), 32896, np.uint16)       # 0.502 of 65535
        Image.fromarray(a, mode="I;16").save(self.tmp / "g.png")
        out, _ = gui.open_rgb8(self.tmp / "g.png")
        self.assertEqual(out.mode, "RGB")
        self.assertAlmostEqual(out.getpixel((0, 0))[0], 128, delta=1)

    def test_icc_profile_kept(self):
        import numpy as np
        from PIL import Image, ImageCms
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        raw = self.tmp / "r.raw"
        np.zeros((4, 6, 4), np.uint8).tofile(raw)
        gui.write_result(str(raw), 6, 4, self.tmp / "o.png", icc)
        with Image.open(self.tmp / "o.png") as im:
            self.assertEqual(im.info.get("icc_profile"), icc)
            self.assertEqual(im.size, (6, 4))


class TestOkKeepsDialogOpen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _dialog(self, on_ok):
        img = QImage(64, 48, QImage.Format.Format_RGB32)
        img.fill(QColor(90, 90, 90))
        d = RefGradeDialog(img, on_ok=on_ok)
        d.show()
        self.app.processEvents()
        return d

    def test_on_ok_false_keeps_open(self):
        d = self._dialog(lambda raw, name, *a: False)
        d._apply_apply({"raw": "x.raw"})
        self.assertTrue(d.isVisible())
        d.close()

    def test_on_ok_none_closes(self):
        d = self._dialog(lambda raw, name, *a: None)       # Krita: layer written → close
        d._apply_apply({"raw": "x.raw"})
        self.assertFalse(d.isVisible())


if __name__ == "__main__":
    unittest.main()
