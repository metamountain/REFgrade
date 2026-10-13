"""Open the RefGrade window live WITHOUT Krita, with the real worker.

The image is turned into exactly the raw format Krita delivers (BGRA, 8 bit), loaded with
/load, and the window opens over the free screen area. OK writes the result as a PNG next to
this script's report folder instead of a Krita layer.

    python tools/ui_live.py [image] [--dummy]      (from core\\)

--dummy opens the layout without worker (no computing).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "core" / "reports"
pkg = types.ModuleType("ref_grade")                 # load the plugin without its krita __init__
pkg.__path__ = [str(ROOT / "plugin" / "ref_grade")]
sys.modules["ref_grade"] = pkg

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog  # noqa: E402

app = QApplication(sys.argv)
from ref_grade import config, worker_client, worker_process  # noqa: E402
from ref_grade.dialog import RefGradeDialog  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
img_path = args[0] if args else str(ROOT / "bench" / "pool" / "07.jpg")
LOG = REPORTS / "ui_live.log"
REPORTS.mkdir(exist_ok=True)
LOG.write_text("", encoding="utf-8")


def say(*a) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(" ".join(map(str, a)) + "\n")


if "--dummy" in sys.argv:
    d = RefGradeDialog(QImage(img_path))
else:
    say("worker", worker_process.ensure(config.PORT, config.python_exe(), config.CORE, timeout=60))
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    h, w = rgb.shape[:2]
    bgra = np.concatenate([rgb[..., ::-1], np.full((h, w, 1), 255, np.uint8)], axis=2)
    raw = config.temp_dir() / "ui_live_source.raw"
    raw.write_bytes(bgra.tobytes())
    source = {"raw": str(raw), "width": w, "height": h, "depth": "U8", "x": 0, "y": 0}
    source["session"] = worker_client.call(config.PORT, "/load", source)["session"]
    say("loaded", img_path, w, h)

    def save_lut(folder: str, filename: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(None, "Save LUT", str(Path(folder) / filename),
                                              "LUT (*.cube)")
        say("save_lut", path)
        return path or None

    def on_ok(raw_path: str, layer_name: str) -> None:
        data = np.frombuffer(Path(raw_path).read_bytes(), np.uint8).reshape(h, w, 4)
        out = REPORTS / "ui_live_result.png"
        Image.fromarray(data[..., 2::-1].copy()).save(out)
        say("OK", layer_name, out)

    d = RefGradeDialog(QImage(img_path), None, client=worker_client.call, source=source,
                       save_lut=save_lut, on_ok=on_ok)
d.show_screen()
d.exec()
say("closed")
