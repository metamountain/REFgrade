"""RefGrade als MCP-Server: eine KI öffnet Bilder, sieht die Vorschau, liest Scope-Zahlen, setzt
Regler, nutzt AUTO und exportiert LUT/Bild (K32a, 2026-10-11).

Start (stdio):  python -m mcp_server        (aus core\\)
Eigener Worker auf freiem Port, beim ersten Aufruf gestartet, am Ende beendet. Mit
RG_WORKER_PORT=<port> wird ein laufender Worker benutzt (z. B. der des Krita-Plugins).

Die Logik steckt in einfachen Funktionen (testbar ohne MCP); `build_server()` registriert sie nur.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))

from refgrade import auto as _auto  # noqa: E402
from refgrade.help import HELP  # noqa: E402
from refgrade.scope_stats import stats as _stats  # noqa: E402
from refgrade.sections import SECTIONS, VISIBLE  # noqa: E402
from refgrade.sliders import SLIDERS  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="refgrade_mcp_"))
_SLIDER = {s.name: s for s in SLIDERS}
_state: dict = {"port": None, "proc": None, "images": {}}   # images: session → source path


class GradeError(ValueError):
    """Fehler, den die KI lesen soll (falscher Name, falscher Bereich …)."""


# ── Worker ──────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _call(path: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    url = f"http://127.0.0.1:{_port()}{path}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise GradeError(f"Worker {path}: {msg}") from None


def _port() -> int:
    if _state["port"]:
        return _state["port"]
    env_port = os.environ.get("RG_WORKER_PORT")
    if env_port:
        _state["port"] = int(env_port)
        return _state["port"]
    port = _free_port()
    _state["proc"] = subprocess.Popen(
        [sys.executable, "-m", "worker", "--port", str(port), "--tmp", str(_TMP / "worker")],
        cwd=str(CORE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _state["port"] = port
    for _ in range(200):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return port
        except OSError:
            time.sleep(0.2)
    raise GradeError("Worker startet nicht")


def shutdown() -> None:
    """Eigenen Worker beenden (nie einen fremden über RG_WORKER_PORT)."""
    proc = _state.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    _state.update(port=None, proc=None)


atexit.register(shutdown)


# ── Werte prüfen (die eine Stelle) ──────────────────────────────────────────

def _check_values(values: dict | None) -> tuple[dict, list[str]]:
    """Unbekannte Namen → Fehler mit Liste; außerhalb des Bereichs → geclampt + Meldung."""
    values = dict(values or {})
    unknown = [k for k in values if k not in _SLIDER or k not in VISIBLE]
    if unknown:
        raise GradeError(f"Unbekannte Regler {unknown}. Gültig: {', '.join(VISIBLE)}")
    notes, out = [], {}
    for k, v in values.items():
        s = _SLIDER[k]
        c = min(max(float(v), s.min), s.max)
        if c != float(v):
            notes.append(f"{k} {v} → {c} (Bereich {s.min}…{s.max})")
        out[k] = c
    return out, notes


def _refs(session: str, references: list[str] | None) -> tuple[list[str], dict]:
    """Ohne Vorlagen: Bild selbst als Platzhalter, Bereich REFERENCE aus."""
    weights = {k: 1.0 for k in SECTIONS}
    if references:
        return list(references), {"weights": weights, "master": 1.0, "solo": None}
    weights["reference"] = 0.0
    return [_state["images"][session]], {"weights": weights, "master": 1.0, "solo": None}


def _session(session: str) -> str:
    if session not in _state["images"]:
        raise GradeError(f"Unbekannte Session {session!r}: zuerst open_image aufrufen")
    return session


# ── Werkzeuge (einfache Funktionen) ─────────────────────────────────────────

def open_image(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise GradeError(f"Datei nicht gefunden: {path}")
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    h, w = a.shape[:2]
    raw = _TMP / f"src_{len(_state['images'])}.raw"
    np.dstack([a[..., 2], a[..., 1], a[..., 0], np.full((h, w), 255, np.uint8)]).tofile(raw)
    sid = _call("/load", {"raw": str(raw), "width": w, "height": h, "depth": "U8"})["session"]
    _state["images"][sid] = str(p.resolve())
    return {"session": sid, "width": w, "height": h}


def sliders() -> list[dict]:
    out = []
    section_of = {n: k for k, names in SECTIONS.items() for n in names}
    for name in VISIBLE:
        s = _SLIDER[name]
        text = (HELP.get(name) or {}).get("en", "") if isinstance(HELP.get(name), dict) else ""
        out.append({"name": name, "section": section_of.get(name, ""), "min": s.min, "max": s.max,
                    "default": s.default, "step": s.step, "help": text.split(". ")[0][:160]})
    return out


def _preview_img(session: str, values: dict, references, method: str, max_side: int):
    refs, mix = _refs(session, references)
    r = _call("/preview", {"session": session, "references": refs, "method": method,
                           "values": values, "max_side": max_side, "mix": mix, "stage": "full"})
    return Image.open(r["image"]).convert("RGB")


def preview(session: str, values: dict | None = None, references: list[str] | None = None,
            method: str = "reinhard", max_side: int = 768) -> tuple[bytes, str]:
    """→ (JPEG-Bytes, Text mit benutzten Werten)."""
    _session(session)
    vals, notes = _check_values(values)
    img = _preview_img(session, vals, references, method, min(int(max_side), 1600))
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    text = f"Werte: {json.dumps(vals)}" + (f"\nGeclampt: {'; '.join(notes)}" if notes else "")
    return buf.getvalue(), text


def scopes(session: str, values: dict | None = None, references: list[str] | None = None,
           method: str = "reinhard") -> dict:
    _session(session)
    vals, notes = _check_values(values)
    img = _preview_img(session, vals, references, method, 512)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
    out = _stats(t)
    if notes:
        out["clamped"] = notes
    return out


def auto(session: str, section: str, values: dict | None = None) -> dict:
    _session(session)
    if section not in _auto.AUTO_SECTIONS:
        raise GradeError(f"AUTO gibt es nur für {list(_auto.AUTO_SECTIONS)}")
    vals, _ = _check_values(values)
    return _call("/auto", {"session": session, "section": section, "values": vals})


def export_lut(session: str, values: dict | None, directory: str,
               references: list[str] | None = None, method: str = "reinhard") -> str:
    _session(session)
    vals, _ = _check_values(values)
    refs, mix = _refs(session, references)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return _call("/cube", {"session": session, "references": refs, "method": method,
                           "values": vals, "dir": directory, "mix": mix})["path"]


def export_image(session: str, values: dict | None, out_path: str,
                 references: list[str] | None = None, method: str = "reinhard") -> str:
    """Volle Auflösung. Clarity liefert der Worker nur als Krita-Filtermaske — hier ohne Clarity."""
    _session(session)
    vals, _ = _check_values(values)
    refs, mix = _refs(session, references)
    raw = _TMP / "export.raw"
    _call("/apply", {"session": session, "references": refs, "method": method,
                     "values": vals, "out": str(raw), "mix": mix})
    src = Image.open(_state["images"][session])
    w, h = src.size
    bgra = np.fromfile(raw, dtype=np.uint8).reshape(h, w, 4)
    rgb = np.ascontiguousarray(bgra[..., [2, 1, 0]])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out, quality=95) if out.suffix.lower() in (".jpg", ".jpeg") \
        else Image.fromarray(rgb, mode="RGB").save(out)
    return str(out)


# ── MCP-Hülle ───────────────────────────────────────────────────────────────

TOOL_NAMES = ("open_image", "sliders", "preview", "scopes", "auto", "export_lut", "export_image")


def build_server():
    from mcp.server.fastmcp import FastMCP, Image as McpImage

    mcp = FastMCP("refgrade", instructions=(
        "Colour grading for one image. open_image → session. sliders() lists every control with "
        "range. preview() shows the graded image, scopes() gives numbers (percentiles, clipping, "
        "colour cast per zone in Lab a*/b*, saturation). auto(section) suggests values for "
        "tone/color/wheels/rgb/curves. Values are a dict {slider: value}; unchanged sliders keep "
        "their default. Neutral cast = a*, b* near 0."))

    @mcp.tool(name="open_image", description="Open an image file; returns session id and size.")
    def _open_image(path: str) -> dict:
        return open_image(path)

    @mcp.tool(name="sliders", description="All controls: name, section, min, max, default, step, help.")
    def _sliders() -> list[dict]:
        return sliders()

    @mcp.tool(name="preview", description="Graded preview image for the given slider values.")
    def _preview(session: str, values: dict | None = None, references: list[str] | None = None,
                 method: str = "reinhard", max_side: int = 768) -> list:
        data, text = preview(session, values, references, method, max_side)
        return [McpImage(data=data, format="jpeg"), text]

    @mcp.tool(name="scopes", description="Scope numbers of the graded image: percentiles, clipping, "
                                         "colour cast per zone (Lab), saturation, 16-bin histogram.")
    def _scopes(session: str, values: dict | None = None, references: list[str] | None = None,
                method: str = "reinhard") -> dict:
        return scopes(session, values, references, method)

    @mcp.tool(name="auto", description="AUTO suggestion for one section (tone, color, wheels, rgb, curves).")
    def _auto_tool(session: str, section: str, values: dict | None = None) -> dict:
        return auto(session, section, values)

    @mcp.tool(name="export_lut", description="Write the grade as a .cube LUT into a directory.")
    def _export_lut(session: str, values: dict | None, directory: str,
                    references: list[str] | None = None, method: str = "reinhard") -> str:
        return export_lut(session, values, directory, references, method)

    @mcp.tool(name="export_image", description="Write the graded image at full resolution "
                                               "(PNG/TIFF/JPG by extension; without clarity).")
    def _export_image(session: str, values: dict | None, out_path: str,
                      references: list[str] | None = None, method: str = "reinhard") -> str:
        return export_image(session, values, out_path, references, method)

    return mcp


def main() -> None:
    try:
        build_server().run()
    finally:
        shutdown()


if __name__ == "__main__":
    main()
