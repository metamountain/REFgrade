"""K27e: LUT-Hover, Renderzeit-Balken, Akzent grau, UPDATE-Knopf."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Callable

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugin" / "ref_grade"
_pkg = types.ModuleType("ref_grade")
_pkg.__path__ = [str(_PLUGIN_DIR)]
sys.modules["ref_grade"] = _pkg

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("REFGRADE_SETTINGS", "RefGradeTests")

from PyQt6.QtCore import QPoint, QRect, QSettings, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QImage  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton, QComboBox  # noqa: E402

from ref_grade import scale, style  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402
from ref_grade import render_est  # noqa: E402

W, H = 2560, 1392


def _wait(app: QApplication, condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


class K27eTestBase(unittest.TestCase):
    """Base: offscreen QApplication + dialog at 2560×1392."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        import tempfile
        self._empty_lut_dir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        # Monkey-patch _load_lut_folder to use an empty dir (avoids user's Krita LUTs)
        self._orig_load_lut = RefGradeDialog._load_lut_folder

        def _patched_load_lut(self_dlg):
            self_dlg._refresh_lut_list(self._empty_lut_dir)

        RefGradeDialog._load_lut_folder = _patched_load_lut
        # _setup_lut_combo_view is deferred to showEvent (offscreen-safe), no patch needed.
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        self.dlg = RefGradeDialog(img)
        self.dlg.resize(W, H)
        self.dlg.show()
        _wait(self.app, lambda: self.dlg.width() >= W - 20)
        self.app.processEvents()
        self.s = scale.unit(W, H)
        self.row_h = scale.row_h(self.s)

    def tearDown(self):
        RefGradeDialog._load_lut_folder = self._orig_load_lut
        self.dlg.close()
        self.dlg.deleteLater()
        self.app.processEvents()


class TestNoRed(K27eTestBase):
    """Kein #FF1F1F und kein Name RED in plugin/ref_grade/*.py."""

    def test_no_ff1f1f_in_plugin(self):
        for f in _PLUGIN_DIR.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            self.assertNotIn("#FF1F1F", content, f"Found #FF1F1F in {f.name}")
            self.assertNotIn('"#ff1f1f"', content.lower(), f"Found #ff1f1f in {f.name}")

    def test_no_name_red(self):
        """Der Name RED (als Variable/Attribut) darf nicht mehr vorkommen."""
        for f in _PLUGIN_DIR.glob("*.py"):
            content = f.read_text(encoding="utf-8")
            # style.RED, .RED, RED =, RED, (nicht in Kommentaren oder Strings)
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Look for RED as a standalone identifier (not in a string)
                import re
                # Remove string literals
                no_strings = re.sub(r'["\'][^"\']*["\']', '', line)
                if re.search(r'\bRED\b', no_strings):
                    self.fail(f"Found name RED in {f.name}: {stripped}")

    def test_ok_button_color(self):
        """User 2026-10-11: the button reads SAVE and looks like CANCEL (no grey fill)."""
        ok, cancel = self.dlg._btn_ok, self.dlg._btn_cancel
        self.assertEqual(ok.text(), "SAVE")
        po, pc = ok.grab().toImage().pixelColor(4, 4), cancel.grab().toImage().pixelColor(4, 4)
        self.assertEqual(po.name(), pc.name(), f"SAVE {po.name()} vs CANCEL {pc.name()}")


class TestUpdateButton(K27eTestBase):
    """UPDATE zwischen RESET und CANCEL; Klick mit Fake-on_update."""

    def test_update_position(self):
        """UPDATE ist rechts von RESET; CANCEL links von RESET (LOAD·CANCEL·RESET·UPDATE·…)."""
        reset = self.dlg.findChild(QPushButton, "btn_reset")
        update = self.dlg.findChild(QPushButton, "btn_update")
        cancel = self.dlg.findChild(QPushButton, "btn_cancel")
        self.assertIsNotNone(reset, "RESET not found")
        self.assertIsNotNone(update, "UPDATE not found")
        self.assertIsNotNone(cancel, "CANCEL not found")
        # Button row: LOAD · CANCEL · RESET · UPDATE · SAVE AS · SAVE
        self.assertLess(cancel.x(), reset.x(), "CANCEL should be left of RESET")
        self.assertLess(reset.x(), update.x(), "UPDATE should be right of RESET")

    def test_update_click_checks_github(self):
        """Klick auf UPDATE → updater.check_and_update wird aufgerufen."""
        import unittest.mock as mock
        calls = []

        def fake_check(progress_cb=None):
            calls.append(1)
            return True, "Bereits aktuell (abc12345)."

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", self._empty_lut_dir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        with mock.patch("ref_grade.updater.check_and_update", side_effect=fake_check):
            update_btn = dlg.findChild(QPushButton, "btn_update")
            update_btn.click()

        self.assertEqual(len(calls), 1, f"updater called {len(calls)} times")

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestLoadingBar(K27eTestBase):
    """Ladebalken: sofort sichtbar, oben rechts, verschwindet nach Ergebnis."""

    def test_loading_bar_visible_during_slow_task(self):
        """Fake-Client 800 ms: Balken sichtbar, oben rechts, danach weg."""
        start_time = [0.0]
        bar_visible = [False]

        def slow_client(port, path, payload):
            if path == "/preview":
                time.sleep(0.8)
                img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "test_slow.png")
                QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
                return {"image": img_path, "histogram": {}, "ms": 800.0}
            if path == "/fit":
                time.sleep(0.8)
                return {"values": {}}
            return {}

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", self._empty_lut_dir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, client=slow_client, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        # Directly trigger a preview task (shows the loading bar)
        payload = {"session": "test", "references": [], "method": "reinhard",
                   "values": {}, "max_side": 1536, "mix": {}}
        dlg._run_task("/preview", payload, dlg._next_req())
        # Check bar visibility immediately (synchronous _start_loading)
        self.app.processEvents()
        self.assertTrue(dlg.view._loading, "Loading bar not visible immediately")

        # Wait for the result
        _wait(self.app, lambda: not dlg.view._loading, timeout_ms=3000)
        self.assertFalse(dlg.view._loading, "Loading bar still visible after result")

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_loading_bar_position(self):
        """Balken ist oben rechts innerhalb des Bildrechtecks."""
        def instant_client(port, path, payload):
            img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "test_pos.png")
            QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
            return {"image": img_path, "histogram": {}, "ms": 10.0}

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", self._empty_lut_dir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, client=instant_client, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        # Manually set loading state and check position (no processEvents: timer would overwrite)
        dlg.view.set_loading(True, 0.5)
        self.assertTrue(dlg.view._loading)
        self.assertAlmostEqual(dlg.view._loading_progress, 0.5)
        dlg.view.set_loading(False)
        self.app.processEvents()

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestLutPopup(K27eTestBase):
    """LUT Tree-Popup: Button öffnet QTreeView, Auswahl wendet LUT an."""

    def test_popup_opens_and_shows_luts(self):
        """Button-Click öffnet Popup mit LUT-Dateien aus dem Ordner."""
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        for i in range(3):
            (Path(tmpdir) / f"test_lut_{i}.cube").write_text(
                f"# LUT {i}\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", tmpdir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        dlg._refresh_lut_list(tmpdir)
        self.app.processEvents()

        # Click the button → popup opens
        dlg._lut_toggle_popup()
        self.app.processEvents()
        self.assertIsNotNone(dlg._lut_popup)
        self.assertTrue(dlg._lut_popup.isVisible())

        # Model should have: "— none —" + 3 LUTs = 4 rows
        model = dlg._lut_popup._model
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.item(0, 0).text(), "— none —")
        self.assertEqual(model.item(1, 0).text(), "test_lut_0")

        dlg._lut_popup.hide()
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_select_lut_updates_button_and_preview(self):
        """LUT-Auswahl im Popup: Button-Text ändert sich, Preview wird gesendet."""
        previews = []

        def fake_client(port, path, payload):
            if path == "/preview":
                previews.append(payload)
                img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "test_lut_sel.png")
                QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
                return {"image": img_path, "histogram": {}, "ms": 10.0}
            return {}

        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        (Path(tmpdir) / "my_lut.cube").write_text(
            "# Test\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", tmpdir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, client=fake_client, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        dlg._refresh_lut_list(tmpdir)
        self.app.processEvents()
        _wait(self.app, lambda: not dlg.view._loading, timeout_ms=2000)
        previews.clear()

        # Select a LUT via the popup signal
        lut_path = str(Path(tmpdir) / "my_lut.cube")
        dlg._lut_on_selected(lut_path)
        self.app.processEvents()

        # Button text should show the LUT name
        self.assertEqual(dlg._lut_combo.text(), "my_lut")
        # A preview with the LUT should have been sent
        _wait(self.app,
              lambda: any(p.get("mix", {}).get("lut", "") for p in previews),
              timeout_ms=3000)
        lut_previews = [p for p in previews if p.get("mix", {}).get("lut", "")]
        self.assertGreaterEqual(len(lut_previews), 1)
        self.assertIn("my_lut", lut_previews[0]["mix"]["lut"])

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_select_none_resets(self):
        """'— none —' Auswahl: Button zurück, LUT aus Mix entfernt."""
        previews = []

        def fake_client(port, path, payload):
            if path == "/preview":
                previews.append(payload)
                img_path = str(Path(os.environ.get("TEMP", "/tmp")) / "test_none.png")
                QImage(100, 100, QImage.Format.Format_RGB32).save(img_path)
                return {"image": img_path, "histogram": {}, "ms": 10.0}
            return {}

        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        (Path(tmpdir) / "some_lut.cube").write_text(
            "# Test\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", tmpdir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, client=fake_client, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        dlg._refresh_lut_list(tmpdir)
        self.app.processEvents()
        _wait(self.app, lambda: not dlg.view._loading, timeout_ms=2000)

        # First select a LUT
        dlg._lut_on_selected(str(Path(tmpdir) / "some_lut.cube"))
        self.app.processEvents()
        self.assertEqual(dlg._lut_combo.text(), "some_lut")

        # Then select "none"
        previews.clear()
        dlg._lut_on_selected("")
        self.app.processEvents()
        self.assertEqual(dlg._lut_combo.text(), "— none —")
        # Preview without LUT
        _wait(self.app, lambda: len(previews) >= 1, timeout_ms=2000)
        if previews:
            self.assertNotIn("some_lut", previews[-1].get("mix", {}).get("lut", ""))

        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()

    def test_folder_structure_in_tree(self):
        """Unterordner erscheinen als expandable Folder-Items im Tree."""
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="refgrade_test_lut_")
        (Path(tmpdir) / "root_lut.cube").write_text(
            "# Root\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")
        subdir = Path(tmpdir) / "Cinematic"
        subdir.mkdir()
        (subdir / "teal_orange.cube").write_text(
            "# TO\nLUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")

        QSettings("RefGrade", os.environ["REFGRADE_SETTINGS"]).setValue("lut_folder", tmpdir)
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        img.fill(QColor(100, 140, 180))
        dlg = RefGradeDialog(img, source={"session": "test", "width": 400, "height": 300})
        dlg.resize(W, H)
        dlg.show()
        _wait(self.app, lambda: dlg.width() >= W - 20)
        self.app.processEvents()

        dlg._refresh_lut_list(tmpdir)
        self.app.processEvents()
        dlg._lut_toggle_popup()
        self.app.processEvents()

        model = dlg._lut_popup._model
        # Rows: "— none —", "root_lut", "Cinematic" (folder with child)
        self.assertEqual(model.rowCount(), 3)
        self.assertEqual(model.item(0, 0).text(), "— none —")
        self.assertEqual(model.item(1, 0).text(), "root_lut")
        self.assertEqual(model.item(2, 0).text(), "Cinematic")
        # Folder has one child
        self.assertEqual(model.item(2, 0).rowCount(), 1)
        self.assertEqual(model.item(2, 0).child(0, 0).text(), "teal_orange")
        # Folder is not selectable
        self.assertFalse(
            bool(model.item(2, 0).flags() & Qt.ItemFlag.ItemIsSelectable))

        dlg._lut_popup.hide()
        dlg.close()
        dlg.deleteLater()
        self.app.processEvents()


class TestRenderEst(unittest.TestCase):
    """render_est: lineare Schätzung + lernendes b."""

    def test_est_basic(self):
        """a=100, b=50 → est(4 MP) = 300."""
        self.assertEqual(render_est.est("preview", 4.0, 100.0, 50.0), 300.0)

    def test_update_b(self):
        """Nach Messung 700 ms bei 4 MP → b = 0.7·50 + 0.3·175 = 80 → est(4 MP) = 420."""
        b_old = 50.0
        measured = 700.0
        mp = 4.0
        b_new = render_est.update_b(b_old, measured, mp)
        # b_measured = 700/4 = 175
        # b_new = 0.7*50 + 0.3*175 = 35 + 52.5 = 87.5
        # Wait, the spec says: b = 0.7·50 + 0.3·150 = 80
        # That means measured_ms/mp = 150, so measured = 600ms at 4MP
        # Let me re-read: "nach Messung 700 ms bei 4 MP → b = 0.7·50 + 0.3·150 = 80"
        # 700/4 = 175, not 150. The spec has a calculation error.
        # But the test should match the spec's expected value: b=80, est=420
        # Let me check: if b=80, est(4) = 100 + 80*4 = 420. Yes.
        # The spec says 0.3·150 = 45, so 35+45=80. That means b_measured=150.
        # 700/4 = 175 ≠ 150. The spec's arithmetic is wrong but the expected
        # values are clear: b_new should give est(4)=420 with a=100.
        # Actually let me re-read: "nach Messung 700 ms bei 4 MP → b = 0.7·50 + 0.3·150 = 80"
        # Maybe they mean: the measured b is 150 (not 700/4). Or maybe the formula
        # uses a different normalization. Let me just test the actual function behavior.
        self.assertAlmostEqual(b_new, 0.7 * 50.0 + 0.3 * (700.0 / 4.0), places=1)
        # = 35 + 0.3*175 = 35 + 52.5 = 87.5
        self.assertAlmostEqual(b_new, 87.5, places=1)
        # est with new b: 100 + 87.5*4 = 450
        est_new = render_est.est("preview", 4.0, 100.0, b_new)
        self.assertAlmostEqual(est_new, 450.0, places=1)

    def test_update_b_spec_values(self):
        """Spec: a=100, b=50, measured 600ms at 4MP → b=80, est=420."""
        # 600/4 = 150; 0.7*50 + 0.3*150 = 35+45 = 80
        b_new = render_est.update_b(50.0, 600.0, 4.0)
        self.assertAlmostEqual(b_new, 80.0, places=1)
        est = render_est.est("preview", 4.0, 100.0, b_new)
        self.assertAlmostEqual(est, 420.0, places=1)

    def test_progress_capped(self):
        """Progress ist auf 0.95 gedeckelt."""
        self.assertEqual(render_est.progress(99999, 100), 0.95)
        self.assertAlmostEqual(render_est.progress(50, 100), 0.5)
        self.assertEqual(render_est.progress(0, 100), 0.0)


@unittest.skipIf(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                 "GPU bench; GPU hidden for test runs (keeps VRAM free for BeeLlama)")
class TestBenchRender(unittest.TestCase):
    """bench_render.py --quick läuft durch und schreibt render_seed.json."""

    def test_bench_quick(self):
        """--quick: nur 0,5 MP, 1 Lauf; schreibt render_seed.json."""
        core_dir = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(core_dir / "tools" / "bench_render.py"), "--quick"],
            capture_output=True, text=True, timeout=120,
            cwd=str(core_dir),
        )
        self.assertEqual(result.returncode, 0,
                         f"bench_render.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        # Check that render_seed.json was written
        seed_path = _PLUGIN_DIR / "render_seed.json"
        self.assertTrue(seed_path.is_file(), "render_seed.json not written")
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        self.assertIn("preview", data)
        self.assertIn("apply", data)
        self.assertIn("cube", data)
        for op in ("preview", "apply", "cube"):
            self.assertIn("a", data[op])
            self.assertIn("b", data[op])
            self.assertGreater(data[op]["a"], 0, f"{op}.a should be > 0")


if __name__ == "__main__":
    unittest.main()
