"""Pixel-Regeln: Unterstützungs-Check, Ebenenname, Byte-Zählung. Ohne krita, ohne Qt."""

from __future__ import annotations

from pathlib import Path

_PREFIX = "REF · "
_MAX_STEM = 60


def check(color_model: str, color_depth: str, profile: str) -> str | None:
    """None wenn unterstützt, sonst englische Meldung."""
    if color_model != "RGBA" or color_depth not in ("U8", "U16") or "srgb" not in profile.lower():
        return (
            "RefGrade v1 only handles RGB images with 8 or 16 bit (sRGB). "
            f"This image: {color_model}/{color_depth}/{profile}."
        )
    return None


def layer_name(reference_name: str) -> str:
    """'REF · ' + Stamm des Dateinamens, Stamm max 60 Zeichen."""
    stem = Path(reference_name).stem
    if len(stem) > _MAX_STEM:
        stem = stem[:_MAX_STEM]
    return _PREFIX + stem


def byte_count(w: int, h: int, depth: str) -> int:
    """Byte-Anzahl: w*h*4*(1|2)."""
    itemsize = 1 if depth == "U8" else 2
    return w * h * 4 * itemsize
