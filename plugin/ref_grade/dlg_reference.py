"""ReferenceMixin: reference tiles, method, slots, manual tiles (K26)."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from . import config, scale, style
from .widgets import IMAGE_SUFFIXES, RefSlots, small_button
from .dlg_common import METHODS

class ReferenceMixin:
    """Mixin for RefGradeDialog: reference tiles, method, slots, manual tiles."""

    def _method_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        _rh = scale.row_h(style.current_s())
        self.methods = QButtonGroup(self)
        _method_tips = {
            "REINHARD": "Reinhard colour transfer: matches the mean and spread of the reference in Lab. Robust default.",
            "MKL": "MKL colour transfer (Monge-Kantorovich): matches the full colour covariance of the reference.",
            "ZONAL": "Zonal colour transfer: matches shadows, midtones and highlights of the reference separately.",
            "PAIR": "PAIR: learns a LUT from a before/after pair (two images, or one side-by-side image) and applies it.",
        }
        for i, name in enumerate(METHODS):
            b = QPushButton(name)
            b.setObjectName(f"method_{name.lower()}")
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setFont(style.font("base"))
            b.setProperty("font_role", "base")
            b.setFixedHeight(_rh)
            b.setToolTip(_method_tips[name])
            self.methods.addButton(b, i)
            row.addWidget(b)
        self.fit_btn = QPushButton("FIT")
        self.fit_btn.setObjectName("fit_btn")
        self.fit_btn.setFont(style.font("base"))
        self.fit_btn.setProperty("font_role", "base")
        self.fit_btn.setFixedHeight(_rh)
        self.fit_btn.setToolTip("Auto-fit Tone, Color and Wheels sliders to match the active reference")
        self.fit_btn.clicked.connect(self._do_fit)
        row.addWidget(self.fit_btn)
        self.pair_hint = QLabel()
        self.pair_hint.setObjectName("pair_hint")
        self.pair_hint.setProperty("muted", True)
        self.pair_hint.setFont(style.font("small"))
        self.pair_hint.setProperty("font_role", "small")
        self.pair_hint.setVisible(False)
        row.addWidget(self.pair_hint, 1)
        self.methods.buttonClicked.connect(self._on_method_changed)
        return w


    def _on_method_changed(self, *_a) -> None:
        """FIT disable bei PAIR, Tiles-Labels aktualisieren, Preview anstoßen."""
        pair = self.methods.checkedId() == 3
        self.fit_btn.setEnabled(not pair)
        if self.slots is not None:
            self.slots.set_pair_labels(pair)
        if self._client:
            self._schedule_preview()


    def _wire_tile(self, tile) -> None:
        """Connect signals and event filter for one tile, then refresh names/tooltips."""
        tile.clicked.connect(self._slot_clicked)
        tile.menu_requested.connect(self._slot_menu)
        tile.dropped.connect(self._set_slot)
        tile.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        tile.installEventFilter(self)
        self._refresh_tile_names()


    def _refresh_tile_names(self) -> None:
        """Set objectName and tooltip on every tile from its current index (UI_HELP)."""
        from .ui_help import UI_HELP
        for t in self.slots.tiles:
            name = f"tile_{t.index}"
            t.setObjectName(name)
            t.setToolTip(UI_HELP.get(name, ""))


    def _insert_ref_tile(self):
        """Insert a new tile before the trailing '+' tile (which stays last); return it."""
        n = len(self.slots.tiles)
        at = n - 1 if (n and self.slots.tiles[-1].path is None) else n
        return self.slots.insert_tile(at)


    def _sync_pair_labels(self) -> None:
        """Re-apply BEFORE/AFTER labels after the tile list changed."""
        if self.slots is not None:
            self.slots.set_pair_labels(self.methods.checkedId() == 3)


    def _ref_count(self) -> int:
        """Number of tiles with a loaded reference."""
        return sum(1 for t in self.slots.tiles if t.path is not None)


    def _sync_tile_count(self) -> None:
        """Adjust the tile list: n_refs images + 1 '+' if < MAX_REFS, else MAX_REFS."""
        n = self._ref_count()
        target = n + 1 if n < RefSlots.MAX_REFS else RefSlots.MAX_REFS
        # Remove trailing empty tiles beyond target, or add new '+' tiles
        while len(self.slots.tiles) > target:
            last = self.slots.tiles[-1]
            if last.path is None:
                self.slots.remove_tile(len(self.slots.tiles) - 1)
            else:
                break
        while len(self.slots.tiles) < target:
            self.slots.insert_tile(len(self.slots.tiles))
        self._refresh_tile_names()
        self._sync_pair_labels()


    def _on_dropped_multi(self, paths: list[str]) -> None:
        """Drag & drop of one or more files onto the grid: load all (up to MAX_REFS)."""
        self._ref_on_if_off()
        room = RefSlots.MAX_REFS - self._ref_count()
        for path in paths[:room]:
            if path in [t.path for t in self.slots.tiles if t.path]:
                continue
            tile = self._insert_ref_tile()
            tile.set_image(path, Path(path).stem)
        self._sync_tile_count()
        self._save_manual_tiles()
        if self._client is not None:
            self._schedule_preview()


    def _ref_on_if_off(self) -> None:
        """If Reference is OFF, turn it ON (loading a ref activates the section)."""
        header = self.sections.get("reference", (None, None))[0]
        if header is not None and not header.is_on():
            header.on.setChecked(True)


    def _fill_slots_from_pool(self) -> None:
        files = sorted(p for p in config.POOL.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        n = min(len(files), RefSlots.MAX_REFS)
        self.slots.set_count(n + 1 if n < RefSlots.MAX_REFS else RefSlots.MAX_REFS)
        for tile, path in zip(self.slots.tiles, files):
            tile.set_image(str(path), path.stem)
        if files:
            self.slots.tiles[0].set_active(True)
        self._sync_pair_labels()


    def _set_slot(self, i: int, path: str | None) -> None:
        """Replace the image on tile i (drop on an existing tile)."""
        self.slots.tiles[i].set_image(path, Path(path).stem if path else "")
        if not path:
            self.slots.tiles[i].set_active(False)
        self._save_manual_tiles()
        if self._client is not None:
            self._schedule_preview()


    def _slot_clicked(self, i: int, ctrl: bool) -> None:
        tile = self.slots.tiles[i]
        if tile.path is None:
            self._ref_on_if_off()
            path = self.open_image()
            if path:
                tile.set_image(path, Path(path).stem)
                self._sync_tile_count()
                self._save_manual_tiles()
                if self._client is not None:
                    self._schedule_preview()
            return
        if not ctrl:
            for t in self.slots.tiles:
                t.set_active(False)
        tile.set_active(not tile.active if ctrl else True)
        if self._client is not None:
            self._schedule_preview()


    def _slot_menu(self, i: int, action: str) -> None:
        if action == "replace":
            path = self.open_image()
            if path:
                self._set_slot(i, path)
        elif action == "reload":
            self._set_slot(i, self.slots.tiles[i].path)
        elif action == "clear":
            self.slots.remove_tile(i)
            self._sync_tile_count()
            self._save_manual_tiles()
            if self._client is not None:
                self._schedule_preview()


    def _ask_image(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference", str(config.POOL),
            "Images (" + " ".join("*" + s for s in IMAGE_SUFFIXES) + ")")
        return path or None

    # ── QSettings: manual tiles ───────────────────────────────────────────────

    def _load_manual_tiles(self) -> None:
        """Restore the reference list from QSettings (migrates old 6-slot format)."""
        import json
        raw = self._settings.value("refs_list", "")
        if raw:
            try:
                paths = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                paths = []
        else:
            # Migrate old tile_0..tile_5 format
            paths = []
            for i in range(6):
                p = self._settings.value(f"tile_{i}", "")
                if p:
                    paths.append(p)
            self._settings.remove("tile_0")
            for i in range(1, 6):
                self._settings.remove(f"tile_{i}")
        # Keep only paths that still exist
        paths = [p for p in paths if Path(p).exists()][:RefSlots.MAX_REFS]
        self.slots.set_count(len(paths) + 1 if len(paths) < RefSlots.MAX_REFS else RefSlots.MAX_REFS)
        for tile, path in zip(self.slots.tiles, paths):
            tile.set_image(path, Path(path).stem)


    def _save_manual_tiles(self) -> None:
        """Save the reference list (no gaps) to QSettings as JSON."""
        import json
        paths = [t.path for t in self.slots.tiles if t.path]
        self._settings.setValue("refs_list", json.dumps(paths))

    # ── tile fill from /references ────────────────────────────────────────────

    def _fill_empty_slots(self) -> None:
        """Fill empty tiles via /references."""
        if self._client is None or self._session_id is None:
            return
        n = self._ref_count()
        if n >= RefSlots.MAX_REFS:
            return
        occupied = [t.path for t in self.slots.tiles if t.path]
        payload = {
            "session": self._session_id,
            "pool": str(config.POOL),
            "k": RefSlots.MAX_REFS - n,
            "exclude": occupied,
        }
        req_id = self._next_req()
        self._run_task("/references", payload, req_id)

    # ── state → payload ───────────────────────────────────────────────────────

    def _references(self) -> list[str]:
        """Paths of active tiles."""
        return [t.path for t in self.slots.tiles if t.path and t.active]


    def _method(self) -> str:
        """Active method name (lowercase)."""
        idx = self.methods.checkedId()
        return METHODS[idx].lower() if idx >= 0 else "reinhard"


    def _first_ref_name(self) -> str:
        """Name of the first active reference (for LUT / layer naming)."""
        refs = self._references()
        return Path(refs[0]).stem if refs else "unnamed"

    # ── preview system ────────────────────────────────────────────────────────

    def _apply_references(self, result: dict) -> None:
        """Fill tiles from /references response (adds new tiles dynamically)."""
        items = result.get("items", [])
        for item in items:
            if self._ref_count() >= RefSlots.MAX_REFS:
                break
            tile = self._insert_ref_tile()
            # A pre-generated thumbnail speeds up display; fall back to the source path
            # when the worker did not provide one (e.g. a PAIR before/after pair).
            tile.set_image(item.get("thumb") or item["path"], item["name"])
            tile.path = item["path"]
            tile.label = item["name"]
            tile.update()
        self._sync_tile_count()
        if items and not any(t.active for t in self.slots.tiles):
            first = next((t for t in self.slots.tiles if t.path), None)
            if first is not None:
                first.set_active(True)
        self._save_manual_tiles()
        self._update_mark()
        self._schedule_preview()              # first grading as soon as a reference is active


    def _tile_hover(self, tile) -> None:
        """Hover over a tile → /prerender with that reference."""
        if self._client is None or self._session_id is None or tile.path is None:
            return
        payload = {
            "session": self._session_id,
            "reference": tile.path,
            "method": self._method(),
            "values": self._values(),
            "max_side": self._max_side(),
            "mix": self.mode(),
        }
        req_id = self._next_req()
        self._run_task("/prerender", payload, req_id)
        self._hover_tile = tile
        self.view.set_status(f"PREVIEW · {tile.label}", f"PREVIEW · {tile.label}")


    def _tile_leave(self) -> None:
        """Leave a tile → restore current grading."""
        if self._client is None:
            return
        self._hover_tile = None
        self._restore_computed()

