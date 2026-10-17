# RefGrade — Funktionsreferenz

Stand: 2026-10-15. Detail-Referenz je Funktion. Architektur → `KONZEPT.md`.

---

## Worker-Endpunkte

Worker: `ThreadingHTTPServer` auf `127.0.0.1:47123`. JSON rein/raus.
Session-LRU: max 2. GPU-Lock serialisiert Rechenaufträge.
Errors: 400 (Bad Request), 409 (unbekannte Session), 500 (sonstiges).

### GET

| Endpoint | Params | Returns |
|----------|--------|---------|
| `/health` | — | `{ok, version, device, auto_sections}` |
| `/sliders` | — | `{sliders: [...], help: {...}}` |

### POST

| Endpoint | Params | Returns |
|----------|--------|---------|
| `/load` | `raw, width, height, depth, ignore_mask?` | `{session, width, height, depth}` |
| `/references` | `session, pool, k=6, exclude=[]` | `{items: [{path, name, score, thumb}]}` |
| `/thumb` | `path, session?` | `{thumb, score?}` |
| `/preview` | `session, references, method, max_side, values, mix?, stage?, scopes?, lut_size?, clip_view?, timing?` | `{image, histogram, ms, clip, vectorscope?, parade?, wave?, pair_error?, timing?}` |
| `/crop` | `session, references, method, values, rect=[x,y,w,h], max_side, mix?` | `{image, scale, vectorscope, parade}` |
| `/prerender` | `session, reference, method, values, max_side, mix?` | `{image}` |
| `/whitebalance` | `session, mode="auto"\|"pick", point=[x,y]?` | `{temperature, tint}` |
| `/auto` | `session, section, values, references, method, mix?` | `{values: {...}}` oder `{curves: {...}}` |
| `/fit` | `session, references, method` | `{values: {...}}` (rejected bei method="pair") |
| `/apply` | `session, references, method, values, out, mix?` | `{raw, ms, clarity?}` |
| `/cube` | `session, references, method, values, dir, mix?` | `{path}` |
| `/shutdown` | — | `{ok}` |

**Daten-Handoff:** Bilder als BMP auf Disk (`config.temp_dir()`), Pfade im JSON.
`/apply` schreibt `result.raw` (BGRA), Host-Callback liest mit numpy → RGB.

**Two-stage Preview:** `stage="lut"` + `lut_size` 17 (drag) / 33 (release).
Bei Clarity/Texture/Dehaze: `stage="full"` Follow-up.

**Request-Staleness:** `_latest[route]` + `_req_path` — alte Replies werden dropped.

---

## LOAD (Standalone)

**Datei:** `dialog.py:575-624`

| Funktion | Was |
|----------|-----|
| `_do_load_menu` | QMenu: Open file… / Paste screenshot (Ctrl+V) / Folder… |
| `_load_from_file` | QFileDialog → `_load_path` |
| `_load_from_clipboard` | Clipboard → temp PNG → `_load_path` |
| `_load_from_folder` | Ordner-Picker → erste Bilddatei → `_load_path` |
| `_load_path(path)` | `source["path"] = path` → `on_update()` |

- Krita: Button disabled
- Ctrl+V: QShortcut, nur wenn `source["path"]` gesetzt
- Filter: `*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp`

---

## SAVE AS

**Datei:** `dialog.py:626-635` + `dlg_worker.py:436-456`

1. `_do_saveas`: Ordner-Picker (QSettings `saveas_folder` merkt sich letzten)
2. Ziel: `<folder>/<stem>_rg.<ext>` (Format des Originals, sonst .png)
3. `self._saveas_out = target` → `_do_ok()`
4. `_apply_apply`: wenn `_saveas_out` gesetzt → raw lesen → RGB → PNG speichern → Status → Flag clear → **kein** `accept()` (Dialog bleibt offen)

---

## SAVE

**Datei:** `dlg_worker.py:483-497` + `dlg_worker.py:436-456`

1. `_do_ok`: `/apply` mit `out=config.temp_dir()/"result.raw"`
2. `_apply_apply` (normaler Pfad): `_on_ok(raw, layer_name, clarity_info?)` → `accept()`
3. Krita: `on_ok` → `write_layer` (neue Layer)
4. Standalone: `on_ok` → `<stem>_rg.<ext>` neben Source

---

## Color Replace

**Core:** `core/refgrade/color_replace.py`
**UI:** `plugin/ref_grade/dlg_color_replace.py`
**Dialog-Wiring:** `dialog.py:730-775`

### Werte (je Instanz, max 8)

| Key | Range | Default | Bedeutung |
|-----|-------|---------|-----------|
| `cr_{i}_src_hue` | 0-360 | 0 | Quell-Hue |
| `cr_{i}_src_sat` | 0-1 | 1 | Quell-Sättigung (skaliert Maske) |
| `cr_{i}_src_bright` | 0-1 | 1 | Schwacher Helligkeits-Modulator (0.7..1.0): V≈src_bright→1.0, V weit→0.7. Farbe primär, Helligkeit nur feine Abstimmung |
| `cr_{i}_tgt_hue` | 0-360 | 0 | Ziel-Hue |
| `cr_{i}_tgt_sat` | 0-1 | 1 | Ziel-Sättigung |
| `cr_{i}_tgt_bright` | 0-1 | 1 | OUTPUT-Helligkeit-Skalierung (1=normal, <1=dunkler, >1= heller, geclampt 0..1) |
| `cr_{i}_intensity` | 0-1 | 0.0 (Core) / 0.5 (UI-Start) | Effekt-Stärke (0 = skip) |
| `cr_{i}_range` | 1-180 | 30 | Hue-Halbbreite in Grad |

Core-Default 0.0 = inaktiv: Der Worker erhält nur Werte für existierende UI-Instanzen,
Instanzen 1–7 fallen auf den Core-Default zurück und müssen nicht wirken (siehe
`core/cr_notes.md`, 2026-10-15). In der LUT-Kette ist `color_replace` ein eigener
Bereich (`session._section_fns`, Position nach HSL, vor LUT).

### Algorithmus (`_replace_one`)

1. `h, S, V = rgb_to_hsv(x)`
2. Hue-Distanz (kreisförmig, 0..180)
3. Plateau-Maske: `plateau=0.6`, weicher Abfall am Rand
4. Saturation-Guard: smoothstep, graue Pixel (S≈0) unberührt
5. Helligkeits-Proximität (0..1): `bright_gate = 1 - |V - src_bright| * 4` clamped
   → schwacher Modulator `bright_mod = 0.7 + 0.3 * bright_gate` (0.7..1.0)
6. Finale Maske: `plateau × sat_guard × bright_mod × src_sat × intensity`
7. Ziel: `hsv_to_rgb(tgt_hue, tgt_sat, (V * tgt_bright).clamp(0,1))` (OUTPUT-Helligkeit)
8. Blend: `(x + mask * (tgt_rgb - x)).clamp(0, 1)` (Final-Clamp: intensity>1 = Overshoot)

### Pipette

| Ziel | Cursor | Auslöser |
|------|--------|----------|
| Source | Eyedropper (Ballon **hohl**, 45°, Cross am Sample-Punkt) | Klick auf SRC-Farbfeld oder "Source" Button |
| Target | Crosshair (Sniper: cross + ring + dot) | Klick auf TGT-Farbfeld oder "Target" Button |

- Cursor auf **ganzem Dialog** (`self.setCursor`), nicht nur Bild
- Nach Pick: `self.unsetCursor()`
- Farbfelder: 44×44 QLabel, klickbar (eventFilter), PointingHandCursor
- Pick-Mechanik: `eventFilter` auf `self.view` → `_cr_pick_at(pos)` → Pixel lesen → `set_pipette_color`

---

## Cursor

**Datei:** `cursors.py`

Beide: 32×32, monochrom (`style.ACCENT_HI`), kein Antialiasing.

| Cursor | Design | Hotspot |
|--------|--------|---------|
| `pipette()` | Eyedropper 45°, Ballon hohl (NoBrush), thin tube, Cross am Sample-Punkt | Cross-Zentrum (9, 23) |
| `crosshair()` | Sniper: cross lines + ring + center dot | Zentrum (16, 16) |

---

## Standalone

**Datei:** `standalone/gui.py` + `plugin/ref_grade/runtime.py` (gemeinsamer Installer)

**Start:** `python standalone/gui.py [bild.png] [--setup]`

Flow:
1. Krita-Mock in `sys.modules` injizieren
2. **torch-Check** (`importlib.util.find_spec("torch")`, reine Python, **vor** dem PyQt6-Import)
3. torch vorhanden: torch importieren (**vor** PyQt6 — DLL-Regel), dann PyQt6 + `ref_grade.dialog`
4. torch fehlt (oder `--setup`): PyQt6 + `dlg_install.InstallDialog` (K38d-Dialog,
   `PyQt6` kommt als Extra-Paket in die venv); danach **Neustart** mit dem
   Runtime-Python (`subprocess.Popen([venv_python, gui.py, *argv])`) — torch darf
   nicht nach PyQt6 im selben Prozess geladen werden. „Use existing Python“ → mit
   diesem Python neu starten. Abbruch → Meldung, Ende.
5. Bild wählen (Arg oder QFileDialog)
6. Worker starten (`worker_process.ensure`; braucht torch)
7. Source lesen (`_read_source` → BGRA raw)
8. `/load` → Session-ID
9. Thumbnail (PIL, ≤2048px) → QImage
10. Callbacks: `on_ok` (speichert `_grad.png`), `on_update` (Stub), `save_lut`
11. `RefGradeDialog(...)` → `exec()`
12. `finally`: Worker stoppen

**Identisch mit Krita:** Dialog, Worker-Client, alle Funktionen.
**Unterschied:** nur Host-Callbacks + Source-Reader + Runtime-Setup.

### Runtime-Setup (`runtime.py`, K38e)

**Warum vor Qt:** Auf Windows führt ein PyQt6-Import **vor** torch zu
`WinError 1114` (DLL-Konflikt). Darum wird torch (falls vorhanden) zuerst geladen;
der Install-Dialog braucht Qt und läuft deshalb nur in einem Prozess **ohne** torch —
der Prozess endet danach über Neustart oder Abbruch.

Installer ist derselbe wie im Krita-Plugin (`plugin/ref_grade/runtime.py`):
uv + Python 3.12 + torch (Flavor) + numpy/kornia/pillow; die Standalone hängt
`PyQt6` an (`runtime.install(..., extra=("PyQt6",))` → PACKAGES-Schritt;
Marker-`packages` bleibt `list(PACKAGES)`).

CLI (ohne Qt, `plugin/` als cwd):
- `python -m ref_grade.runtime --detect` → Erkennung als JSON (installiert nichts)
- `python -m ref_grade.runtime --install <flavor> [--dir <ort>]` → Installation, Log auf stdout, Exit 0/1

---

## Config

**Datei:** `plugin/ref_grade/config.py`

| Wert | Typ | Bedeutung |
|------|-----|-----------|
| `ROOT` | Path | Repo-Root (oder Package-Dir bei ZIP-Install) |
| `CORE` | Path | `ROOT / "core"` |
| `POOL` | Path | `ROOT / "bench" / "pool"` |
| `LUTS` | Path | LUT-Verzeichnis |
| `PORT` | int | `47123` |
| `python_exe()` | str\|None | Worker-Interpreter (REFGRADE_PYTHON → venv → PATH → newest install) |
| `temp_dir()` | Path | `%TEMP%\ref_grade` |

---

## UI-Farb-Schema (2026-10-16)

**Datei:** `plugin/ref_grade/schemes.py`. 5 Farben pro Schema: `bg, surface, text, accent, dim`.
Schemas: **Black** (Default), **Dark Grey**, **Light Grey**, **White** (Legacy-Namen:
Carbon→Black, Anthrazite→Dark Grey, Paper→White). Umschaltung per Dropdown in der
MASTER-Zeile; Persistenz QSettings `scheme`.

| API | Wirkung |
|-----|---------|
| `SCHEMES` | `dict[name, ColorScheme]` — die 4 Schemas |
| `current_scheme()` / `current_scheme_name()` | aktives Schema |
| `set_current(name)` | Schema setzen (unbekannt → `DEFAULT`) |
| `role_color(role)` | alte Chrome-Rolle (`TEXT`, `ACCENT`, …) → Farbe des aktiven Schemas |
| `apply_scheme(widget, scheme?)` | QPalette + Fenster-Stylesheet auf das Haupt-Widget setzen |

`style.py` löst die 9 Chrome-Constants live über `role_color` auf (Module-`__getattr__`);
`style.stylesheet()` baut das QSS daraus. Funktionale Farben (`TRACKS`, `overlay_style`,
`MARK_ERROR`, Curves-Kanäle, CR-Swatches) sind schemafrei.

---

## Hilfe pro Wert (K40a, 2026-10-16)

**Dateien:** `core/refgrade/help.py` (Texte), `plugin/ref_grade/widgets.py` (SliderRow),
`plugin/ref_grade/curves_ui.py` (Curves-Editor).

- **HELP-Einträge** haben 3 Schlüssel: `short` (knapp, Tooltip), `what` (was der Wert bewirkt,
  20–320 Zeichen), `tip` (Handhabung, kursiv). 137 Einträge, alle mit `what`.
- **SliderRow:** Jede Row, deren Name einen HELP-Eintrag hat, zeigt einen quadratischen "?"-Button
  (row_h × row_h) zwischen Slider und Wert-Label. Klick → Popup unter dem Button
  (`Qt.Popup`, schließt bei Klick daneben): **short** (fett) / *what* / tip (kursiv).
  Breite `round(2.2 * current_s())`. Rows ohne HELP-Eintrag (`master`, `amount_*`) haben keinen.
- **Curves:** "?"-Glyph links vom ↺-Glyph (gleiche Größe). Hover zeigt das Hilfe-Panel
  (`CHANNEL_HELP` des aktiven Kanals); Klick **pinnt** das Panel (bleibt auch ohne Hover),
  zweiter Klick löst den Pin. ↺ setzt weiterhin nur den aktiven Kanal zurück.
| `lut_dir(kritarc)` | Path | LUT-Dir aus Krita-Config |
