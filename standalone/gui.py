"""Standalone RefGrade GUI: volle Dialog-Oberfläche ohne Krita.

Start:
    python standalone/gui.py [bild.png]

Ohne Argument: Datei-Dialog. Mit Argument: direkt laden.
Worker wird automatisch gestartet und beim Beenden gestoppt.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CORE = _ROOT / "core"

for p in (str(_CORE), str(_ROOT / "plugin")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Krita-Mock (nur für den Import des ref_grade-Pakets) ─────────────────────
_krita_mod = types.ModuleType("krita")


class _FakeKrita:
    @staticmethod
    def instance():
        return _FakeKrita()

    def addExtension(self, *a):
        pass

    def notifier(self):
        class _N:
            def applicationClosing(self):
                class _S:
                    def connect(self, *a):
                        pass
                return _S()
        return _N()


class _FakeExtension:
    def __init__(self, parent):
        pass

    def setup(self):
        pass

    def createActions(self, window):
        pass


_krita_mod.Krita = _FakeKrita
_krita_mod.Extension = _FakeExtension
sys.modules["krita"] = _krita_mod

import numpy as np
from PIL import Image

from ref_grade import config, worker_client, worker_process
from ref_grade.dialog import RefGradeDialog


def _read_source(path: Path, tmp: Path) -> dict:
    """Bilddatei → source-Dict (wie krita_bridge.read_active, aber ohne Krita)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    a = np.array(img)
    bgra = np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)])
    raw_path = tmp / "source.raw"
    raw_path.write_bytes(bgra.tobytes())
    return {
        "raw": str(raw_path),
        "width": w,
        "height": h,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "path": str(path.resolve()),
    }


def main() -> None:
    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6.QtGui import QImage

    app = QApplication(sys.argv)
    app.setApplicationName("RefGrade Standalone")

    # 1. Bild wählen
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if not img_path.is_file():
            print(f"Fehler: Datei nicht gefunden: {img_path}")
            sys.exit(1)
    else:
        path_str = QFileDialog.getOpenFileName(
            None, "Bild öffnen", str(Path.home()),
            "Bilder (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")[0]
        if not path_str:
            print("Abgebrochen.")
            return
        img_path = Path(path_str)

    print(f"Bild: {img_path} ({img_path.stat().st_size // 1024} KB)")

    # 2. Worker starten
    port = config.PORT
    python = config.python_exe()
    print(f"Worker-Port: {port}, Python: {python}")
    worker_process.ensure(port, python, config.CORE, timeout=30.0)
    print("Worker läuft.")

    # 3. Source lesen
    tmp = Path(tempfile.mkdtemp(prefix="refgrade_gui_"))
    source = _read_source(img_path, tmp)
    print(f"Source: {source['width']}x{source['height']}")

    # 4. Session laden
    resp = worker_client.call(port, "/load", source)
    source["session"] = resp["session"]
    print(f"Session: {source['session']}")

    # 5. Thumbnail für den Dialog
    pil_img = Image.open(img_path).convert("RGB")
    pil_img.thumbnail((2048, 2048))
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGB888).copy()

    # 6. Callbacks
    def on_ok(raw_path: str, layer_name: str, clarity_info: dict | None = None) -> None:
        out = img_path.with_name(f"{img_path.stem}_grad.png")
        arr = np.fromfile(raw_path, dtype=np.uint8).reshape(
            source["height"], source["width"], 4)
        rgb = np.ascontiguousarray(arr[..., [2, 1, 0]])
        Image.fromarray(rgb, mode="RGB").save(out, quality=95)
        print(f"Gespeichert: {out}")

    def on_update() -> bool:
        print("UPDATE: noch nicht implementiert (standalone)")
        return False

    def save_lut(folder: str, filename: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(
            None, "LUT speichern", str(Path(folder) / filename), "LUT (*.cube)")
        return path or None

    # 7. Dialog öffnen
    dlg = RefGradeDialog(
        qimg, None,
        client=worker_client.call,
        source=source,
        save_lut=save_lut,
        on_ok=on_ok,
        on_update=on_update,
    )
    dlg.show_screen()

    try:
        dlg.exec()
    finally:
        worker_process.stop(port)
        print("Worker gestoppt. Tschüss.")


if __name__ == "__main__":
    main()
