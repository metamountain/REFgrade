"""Bildbrücke Standalone (ohne Krita): Bilddatei/QImage → Rohdaten, Ergebnis-Rohdaten → Bilddatei.

Rohformat wie in Krita (krita_bridge.read_active): Bytes B,G,R,A je Pixel, Tiefe "U8"
(uint8) oder "U16" (uint16), Byte-Länge = pixel_rules.byte_count = w*h*4*(1|2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import cv2
except ImportError:  # the runtime venv has no OpenCV (only this CLI bridge uses it)
    cv2 = None


def _need_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("The command line tool needs OpenCV: "
                           "python -m pip install opencv-python-headless")

if TYPE_CHECKING:  # nur für Type-Hints; PyQt6 wird erst in read_qimage geladen
    from PyQt6.QtGui import QImage


def _depth_info(depth: str) -> tuple[np.dtype, int]:
    """Tiefe → (numpy-Dtype, voller Alpha-Wert)."""
    if depth == "U8":
        return np.dtype(np.uint8), 255
    if depth == "U16":
        return np.dtype(np.uint16), 65535
    raise ValueError(f"Unbekannte Tiefe: {depth!r}")


def _write_raw(bgra: np.ndarray, tmp: Path) -> str:
    """BGRA-Array nach tmp/'source.raw' schreiben, Pfad als String zurückgeben."""
    tmp.mkdir(parents=True, exist_ok=True)
    raw_path = tmp / "source.raw"
    raw_path.write_bytes(bgra.tobytes())
    return str(raw_path)


def read_image(path: str, tmp: Path) -> dict:
    """Bilddatei → Rohdaten (B,G,R,A) + Metadaten-Dict.

    Lädt mit cv2.imdecode (np.fromfile wegen Umlauten im Pfad). Graustufen → BGR,
    3 Kanäle → voller Alpha-Kanal (255 bzw. 65535). uint8 → "U8", uint16 → "U16",
    andere Typen → ValueError. Rohbytes nach tmp/'source.raw'.

    Returns:
        Dict mit raw, width, height, depth, x, y und path.
    """
    _need_cv2()
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Konnte Bild nicht dekodieren: {path}")

    if img.dtype == np.uint8:
        depth, full = "U8", 255
    elif img.dtype == np.uint16:
        depth, full = "U16", 65535
    else:
        raise ValueError(f"Ununterstützter Pixeltyp: {img.dtype} ({path})")

    if img.ndim == 2:  # Graustufe → BGR
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 3:
            bgr = img
        elif img.shape[2] == 4:
            bgr = img[..., :3]
        else:
            raise ValueError(f"Ununterstützte Kanalanzahl: {img.shape[2]} ({path})")
    else:
        raise ValueError(f"Ununterstützte Form: {img.shape} ({path})")

    h, w = bgr.shape[:2]
    bgra = np.empty((h, w, 4), dtype=bgr.dtype)
    bgra[..., :3] = bgr
    bgra[..., 3] = full

    raw = _write_raw(bgra, tmp)
    return {
        "raw": raw,
        "width": w,
        "height": h,
        "depth": depth,
        "x": 0,
        "y": 0,
        "path": path,
    }


def read_qimage(img: "QImage", tmp: Path) -> dict:
    """Zwischenablage-QImage → Rohdaten (B,G,R,A) + Metadaten-Dict.

    Konvertiert nach Format_ARGB32 (liegt im Speicher als B,G,R,A) → "U8",
    sonst wie read_image (Rohbytes nach tmp/'source.raw').
    """
    import ctypes

    from PyQt6.QtGui import QImage

    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    addr = int(img.constBits())
    buf = (ctypes.c_char * (w * h * 4)).from_address(addr)
    bgra = np.frombuffer(buf, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4).copy()

    raw = _write_raw(bgra, tmp)
    return {
        "raw": raw,
        "width": w,
        "height": h,
        "depth": "U8",
        "x": 0,
        "y": 0,
        "path": "",
    }


def write_image(source: dict, raw_path: Path, out_path: Path) -> None:
    """Ergebnis-Rohdaten (B,G,R,A) → Bilddatei.

    Liest die Rohbytes (Länge prüfen, sonst ValueError), formt sie als BGRA-Array
    der Tiefe source['depth'] und kodiert mit cv2.imencode (tofile wegen Umlauten).
    .png/.tif/.tiff behalten 16 Bit; .jpg → 8 Bit ohne Alpha.
    """
    _need_cv2()
    w, h = source["width"], source["height"]
    dtype, _ = _depth_info(source["depth"])
    expected = w * h * 4 * dtype.itemsize
    data = Path(raw_path).read_bytes()
    if len(data) != expected:
        raise ValueError(f"Rohlänge: erwartet {expected}, gefunden {len(data)}")
    bgra = np.frombuffer(data, dtype=dtype).reshape(h, w, 4)

    ext = Path(out_path).suffix.lower()
    if ext in (".png", ".tif", ".tiff"):
        payload = bgra  # 16 Bit, mit Alpha
    elif ext in (".jpg", ".jpeg"):
        if dtype == np.uint16:
            payload = (bgra.astype(np.uint32) >> 8).astype(np.uint8)[..., :3]
        else:
            payload = bgra[..., :3]  # 8 Bit, ohne Alpha
    else:
        raise ValueError(f"Ununterstützte Endung: {ext!r}")

    ok, buf = cv2.imencode(ext, payload)
    if not ok:
        raise ValueError(f"Konnte Bild nicht kodieren: {out_path}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(buf).tofile(str(out_path))


def result_path(path: str) -> Path:
    """Ergebnispfad neben dem Original: <name>_REFgrade.png (16-Bit-Quelle → .tif)."""
    src = Path(path)
    suffix = ".tif" if _is_16bit(src) else ".png"
    return src.with_name(f"{src.stem}_REFgrade{suffix}")


def _is_16bit(path: Path) -> bool:
    """True, wenn die Bilddatei 16-Bit-Tiefe hat (Dekode-Check)."""
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img is not None and img.dtype == np.uint16
