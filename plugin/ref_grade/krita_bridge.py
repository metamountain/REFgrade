"""Pixelbrücke Krita ↔ Worker: Ebene lesen, Ergebnis als neue Ebene schreiben."""

from __future__ import annotations

from pathlib import Path

import krita

from . import pixel_rules


def read_active(tmp: Path) -> dict | str:
    """Read the active layer → JSON fragment, or an English message for the user."""
    doc = krita.Krita.instance().activeDocument()
    if doc is None:
        return "No image open."

    err = pixel_rules.check(doc.colorModel(), doc.colorDepth(), doc.colorProfile())
    if err is not None:
        return err

    node = doc.activeNode()
    if node is None:
        return "No active layer."

    r = node.bounds()                     # QRect (Node.h:454), not a tuple
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    if w == 0 or h == 0:
        return "The active layer is empty."

    depth = doc.colorDepth()
    data = node.projectionPixelData(x, y, w, h)
    expected = pixel_rules.byte_count(w, h, depth)
    if len(data) != expected:
        return f"Unexpected pixel data length: expected {expected}, got {len(data)}."

    source_path = tmp / "source.raw"
    source_path.write_bytes(bytes(data))

    result: dict = {
        "raw": str(source_path),
        "width": w,
        "height": h,
        "depth": depth,
        "x": x,
        "y": y,
    }

    sel = doc.selection()
    if sel is not None:
        mask_data = sel.pixelData(x, y, w, h)
        mask_path = tmp / "ignore.raw"
        mask_path.write_bytes(bytes(mask_data))
        result["ignore_mask"] = str(mask_path)

    return result


def write_layer(info: dict, result_raw: Path, name: str, clarity_info: dict | None = None) -> None:
    """Write the result as a new paint layer directly above the active layer.

    If clarity_info is provided, attach a filter mask with the unsharp filter.
    """
    data = result_raw.read_bytes()
    w, h = info["width"], info["height"]
    depth = info["depth"]
    expected = pixel_rules.byte_count(w, h, depth)
    if len(data) != expected:
        raise ValueError(f"Unexpected result length: expected {expected}, got {len(data)}.")

    doc = krita.Krita.instance().activeDocument()
    active = doc.activeNode()
    new_node = doc.createNode(name, "paintlayer")
    parent = active.parentNode()
    parent.addChildNode(new_node, active)
    new_node.setPixelData(data, info["x"], info["y"], w, h)

    if clarity_info is not None:
        _add_clarity_mask(doc, new_node, info, clarity_info)

    doc.refreshProjection()


def _add_clarity_mask(doc, layer, info: dict, clarity_info: dict) -> None:
    """Attach an unsharp filter mask to the layer for clarity."""
    x, y = info["x"], info["y"]
    w, h = info["width"], info["height"]

    # Create the unsharp filter with the given parameters
    f = krita.Krita.instance().filter("unsharp")
    cfg = f.configuration()
    cfg.setProperty("halfSize", clarity_info["halfSize"])
    cfg.setProperty("amount", clarity_info["amount"])
    cfg.setProperty("threshold", 0)
    cfg.setProperty("lightnessOnly", True)
    f.setConfiguration(cfg)

    # Load the midtone mask as a selection
    mask_data = Path(clarity_info["mask"]).read_bytes()
    # doc.selection() is None without a global selection (Document.cpp:357): build a fresh one.
    sel = krita.Selection()
    sel.setPixelData(mask_data, x, y, w, h)

    # Create the filter mask and attach it to the layer
    mask = doc.createFilterMask("Clarity (RefGrade)", f, sel)
    layer.addChildNode(mask, None)
