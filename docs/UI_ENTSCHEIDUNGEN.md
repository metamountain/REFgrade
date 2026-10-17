# UI-Entscheidungen (2026-10-15)

## Hilfe pro Wert (K40a, 2026-10-16)

- **"?"-Button pro Slider-Row:** quadratisch (row_h × row_h — wie Rad-↺ und allgemeiner "?"-Button,
  nicht schmaler, sonst verletzt er `test_button_sizes_all_resolutions`), zwischen Slider und
  Wert-Label, objectName `help_{name}`, Tooltip = `short`. Nur bei Rows mit HELP-Eintrag.
- **Popup statt Tooltip:** frameless QLabel mit `Qt.Popup`-Flag (schließt von selbst bei Klick
  daneben, kein eigener Dismiss-Code), lazies erzeugt, Breite `round(2.2 * current_s())`,
  Inhalt `<b>short</b> / what / <i>tip</i>`. Position: `bottomLeft` des Buttons (global).
- **Scale-Change:** `SliderRow.set_value_width` (der Scale-Hook aus `_update_value_widths`)
  aktualisiert auch die "?"-Button-Größe — sonst bleibt die Build-Zeit-Größe (2 px versetzt).
- **Curves:** "?"-Glyph links vom ↺-Glyph (gleiche Größe, `_help_rect`). Hover zeigt das
  `CHANNEL_HELP`-Panel; Klick **pinnt** es (`_help_pinned`), zweiter Klick löst. Klick auf "?"
  ändert nichts am Curve; ↺ setzt weiterhin nur den aktiven Kanal zurück.

## UI-Farb-Schema-System (2026-10-16)

**Maximal 5 Farbwerte** pro Schema, umschaltbar per **Dropdown** (ersetzt den alten 🎨-Cycle-Button).
Definition: `plugin/ref_grade/schemes.py` (`ColorScheme`-Dataclass, `SCHEMES`, `apply_scheme`).

| Rolle | Bedeutung | Black | Dark Grey | Light Grey | White |
|-------|-----------|-------|-----------|------------|-------|
| `bg`      | Haupt-Hintergrund | `#0a0a0a` | `#1e1e1e` | `#c8c8c8` | `#f7f7f7` |
| `surface` | Panel/Section, Borders | `#1a1a1a` | `#2c2c2c` | `#d8d8d8` | `#ffffff` |
| `text`    | Primär-Text | `#e0e0e0` | `#d6d6d6` | `#1a1a1a` | `#1a1a1a` |
| `accent`  | Slider, Buttons, aktive Elemente | `#4a9eff` | `#4a9eff` | `#2563eb` | `#2563eb` |
| `dim`     | Sekundär-Text, inaktiv | `#666666` | `#8a8a8a` | `#5a5a5a` | `#8c8c8c` |

- **Persistenz:** QSettings `scheme` (Registry `HKCU\Software\RefGrade\RefGrade`), Default `Black`.
- **Mechanik:** `style.py` löst die 9 alten Chrome-Constants (`BLACK`, `TEXT`, `MUTED`, `LINE`,
  `LINE_STRONG`, `DIM`, `ACCENT`, `ACCENT_HI`, `ACCENT_HOVER`) **live** über `schemes.role_color`
  auf (Module-`__getattr__`) → Fenster-Stylesheet + QPalette färben die ganze UI um.
- **Funktionale Farben bleiben fix** (kein Schema): `style.TRACKS` (Slider-Gradienten),
  `style.overlay_style` (Overlay-States), Curves-Kanalfarben, Color-Replace-Swatches,
  `style.MARK_ERROR` (graue Fehler-Markierung, K27e).
- Dropdown in der MASTER-Zeile (rechts), Breite = längster Name + Padding, skaliert mit `row_h`.

---

## Runtime-Installer-Dialog (K38d, 2026-10-16)

`plugin/ref_grade/dlg_install.py` (ersetzt die Radio-Buttons von K37b durch eine Combo).
Aufbau oben nach unten:

1. **„Detected:“** + Adapter aus `detect()["adapters"]` (komma-getrennt), darunter die
   `message` von `detect()`, wenn nicht leer.
2. **Build-Combo** (`QComboBox`): Eintrag 0 = `Auto (recommended): <label>` (erkannter
   Flavor), dann alle `FLAVORS` als `<label> - approx. <size_mb/1000:.1f> GB download`.
   Item-Daten = Flavor-Schlüssel.
3. **„Install to:“** `QLineEdit` (Startwert `runtime_dir()`) + **Browse…**
   (`QFileDialog.getExistingDirectory`). Darunter Hinweis-Label: bei jeder Änderung von
   Ort oder Build `check_target` (rot, wenn nicht leer) und sperrt „Install“.
4. **„Use existing Python…“** (`QFileDialog.getOpenFileName`, Filter `python.exe`) →
   `use_existing`; Erfolg → `accept()`, Fehler → Meldung im Hinweis-Label.
5. Log, Fortschritt, Install/Cancel/Retry (wie K37b).

- **Install:** `set_runtime_dir(ort)` (nur wenn Ort ≠ aktuellem `runtime_dir()`), dann
  Installation mit dem gewählten Flavor (Auto = erkannter).
- **Keine festen Pixelwerte** (k27d): Höhen über `scale.button_h(style.current_s())`,
  Fonts über `style.font` (Pattern `dlg_lut.py`).
- **Verborgene Kompatibilitäts-Radios** (`radio_gpu`/`radio_cpu`, `setVisible(False)`):
  `test_plugin_k37b.py` referenziert sie; entfernen, wenn die k37b-Tests auf die Combo
  umgestellt sind (Paket 97, offene Frage 1).
- **Zielordner wird vor `check_target` angelegt** (wenn fehlt), sonst sperrt der
  nicht existierende Default-Ort „Install“ beim Start.

---

## Torch-Setup-Dialog (Standalone, 2026-10-15)

Modal, **vor** dem Hauptfenster, nur wenn `ensure_torch()` False liefert.

- **Titel:** „PyTorch nicht gefunden"
- **Text:** „GPU-Features (Color Replace) brauchen PyTorch. / Ohne PyTorch läuft RefGrade nur mit klassischen Features."
- **3 Buttons** (dunkles Theme, `style.STYLESHEET`):
  1. **Automatisch installieren** (`main="ok"`) → `pip install torch --index-url …/cu118`,
     Progress-Label „Installiere PyTorch… bitte warten (ca. 2–5 min)", Button-Disable während Lauf,
     danach Re-Check; bei Fehler → Fehler-Label, fallthrough zu 2.
  2. **Pfad manuell wählen…** → `QFileDialog.getExistingDirectory` (Ordner mit `torch/`),
     Re-Check, bei Erfolg Pfad in QSettings `refgrade/torch_path`
  3. **Ohne GPU weiter** (`main="cancel"`) → `has_torch=False`, CR-Sektion ausgeblendet

**Persistenz:** `refgrade/torch_path` (QSettings, Registry `HKCU\Software\RefGrade\RefGrade`).
Beim Start **vor** dem torch-Check in `sys.path` legen — einmal gesetzt, nie wieder Dialog.

**WinError 1114:** torch muss vor Qt laden. `ensure_torch()` ist Qt-frei; der Dialog (Qt) kommt
dahinter. Re-Checks nach Qt-Import laufen in frischem Subprozess (`_torch_importable`).

---

## Button-Leiste (unten)

**Reihenfolge (fix, nicht Krita-adaptiv):**
```
LOAD · CANCEL · RESET · UPDATE · SAVE AS · SAVE
```

- U-förmige Aufmerksamkeitskurve: Primäraktionen an den Enden
- Workflow-Reihenfolge: laden → grade → speichern
- LOAD in Krita: disabled (bleibt aber an Position 0)
- Keine Icons, nur Text

### LOAD
- Standalone-only (Krita: disabled)
- Menü: Open file… / Paste screenshot (Ctrl+V) / Folder…
- Ctrl+V = QShortcut, nur wenn `source["path"]` gesetzt
- Mechanik: `source["path"] = new_path` → `on_update()`

### SAVE AS
- Krita + Standalone
- Ordner-Picker, merkt sich letzten Ordner (QSettings `saveas_folder`)
- Default: `config.ROOT`
- Speichert `<name>_grad.png` im gewählten Ordner
- Dialog bleibt offen (kein `accept()`)

### SAVE
- Krita: neue Layer (bestehendes Verhalten)
- Standalone: `<name>_grad.png` neben Source (Suffix `_grad`, nicht `_REFgrade`)

---

## Color Replace

### Aktiv-Zustand
- Section startet **OFF** (`on=False`, 2026-10-15) — wie alle anderen Sektionen
- Inhalt gedimmt (DimGuard), erster Klick schaltet die Section ON

### Farbfelder (ColorSquares)
- Kleine Quadrate unten-links am Wheel (`_ColorSquare`, `wheel_size × 0.25`)
- **Einziger Pick-Trigger**: Source/Target-Buttons entfernt (2026-10-15)
- Klick auf SRC-Farbfeld → Pipette-Modus (Source)
- Klick auf TGT-Farbfeld → Pipette-Modus (Target)
- Cursor: **Loupe** (Lupe) auf Hover, normaler Pfeil sonst (`cursors.loupe()`)
- ToolTip: "Click to pick source/target color"

### Instance-Steuerung (2026-10-15)
- **Close-Button (×)**: `QToolButton` 16×16, oben-rechts an jeder Instanz **außer #1**
  - Klick → `intensity = 0` (Core skippt `intensity ≤ 0`) + Instanz wird ausgeblendet
- **Intensity**: Range 0–**2.0**, Default **0.5** (war 0–1.0)
- **Wheels**: auf 0.7 skaliert (`wheel_size = 0.85·s·0.7`), bei 3+ Instanzen auf **max. 100 px** gekappt (kompakt, 2026-10-16)

### Pipette-Cursor
- **Source**: Eyedropper (45°, Ballon **hohl**, thin tube, Cross am Sample-Punkt)
- **Target**: Crosshair (Sniper: cross + ring + center dot)
- Sichtbar auf **ganzem Dialog** (`self.setCursor`), nicht nur auf Bild
- Nach Pick: `self.unsetCursor()` → normaler Pfeil
- Corporate Design: monochrom, `style.ACCENT_HI` (#B4B4B4), kein Blau

### Brightness-Bars (Fix 2026-10-15: „nicht korrekt verknüpft, soll Farbton beeinflussen")
- `src_bright` (0..1, Default 1.0): **schwacher** Helligkeits-Modulator (2026-10-16, war hartes SELECTION-Gate)
  - `bright_gate = 1 - |V - src_bright| * 4` clamped 0..1 → `bright_mod = 0.7 + 0.3 * bright_gate` (0.7..1.0)
  - V≈src_bright → volle Selektion (1.0), V weit entfernt → nur 0.7. Farbe (Hue+Sat) bleibt primär;
    dunkle + helle Varianten derselben Farbe werden beide getroffen (kein multiplikativer Kill mehr)
- `tgt_bright` (0..1, Default 1.0): skaliert die OUTPUT-Helligkeit
  - `V_tgt = (V * tgt_bright).clamp(0, 1)` (war ungeclampt → RGB > 1.0 möglich)
  - 1.0 = Pixel-V beibehalten, 0.5 = halb so hell, 0.0 = schwarz, >1 = heller (bis Weiß)
- Final-Clamp am `_replace_one`-Ergebnis: `(x + mask * (tgt_rgb - x)).clamp(0, 1)`
  (intensity > 1 = Overshoot kann über 0..1 hinausreichen)
- **Pipette-Verknüpfung**: Pick setzt die zugehörige Brightness-Bar auf `V = max(r,g,b)`
  des gepickten Pixels (`set_src_from_rgb` → `src_bright`, `set_tgt_from_rgb` → `tgt_bright`)
  → gepickte Farbe matcht sich selbst; Ersetzung bekommt die gepickte Helligkeitsstufe
- Beide in `sliders.py` + `sections.py` registriert (Worker akzeptiert sie)
- **Styling (2026-10-15, Slim-Fix 2026-10-15)**: `BrightBar` = **schmale Leiste, fix 8px**
  (war `scale.row_h` wie die anderen CR-Slider) — kompakter Gradient-Strip, kein voller Slider.
  Hintergrund = `QLinearGradient` Schwarz→Weiß (volle Breite),
  Handle = 2px vertikale Linie (`ACCENT_HI`-Kern mit dunklem Rand) bei `value·width`
  → auf beiden Gradient-Enden sichtbar
- **Value-Sync (2026-10-15)**: Das Farbfeld (src/tgt) zeigt **immer** die Wheel-Hue/Sat
  bei der Helligkeit (V) der zugehörigen Brightness-Bar: `_update_squares()` leitet beide
  Quadrate aus `_hsv_to_rgb(wheel_h, wheel_s, bright.value())` ab. Bar bewegen → Quadrat
  aktualisiert sich in Echtzeit (Helligkeit des Quadrats = Handle-Position). Pipette-Pick
  bleibt konsistent, weil Wheel(H,S) + Bar(V) exakt den gepickten Pixel reproduzieren.
  (Früher: Quadrat aus gecachtem `_src_rgb`/`_tgt_rgb` bei V=1.0 → Bar hatte keinen Einfluss.)

### Werte-Pipeline
```
UI (Wheels, BrightBars, Sliders)
  → _values() sammelt cr_{i}_* Keys
  → Worker /preview + /apply
  → core/refgrade/grade.py → color_replace.apply()
  → _replace_one(hue, sat, bright, intensity, range)
```

---

## Technische Fixes (diese Session)

| Datei | Change |
|-------|--------|
| `sliders.py` | `src_bright` + `tgt_bright` registriert; `cr_{i}_intensity` max 1.0→**2.0** |
| `sections.py` | Brightness-Keys in color_replace Section |
| `color_replace.py` | `_replace_one` nimmt `src_bright` + `tgt_bright` |
| `dlg_color_replace.py` | `on=False`, `_ColorSquare` (Loupe-Cursor, `clicked`-Signal), Source/Target-Buttons entfernt, Close-Button (×) oben-rechts, Wheels ×0.7 |
| `dlg_color_replace.py` | **BrightBar-Fix (2026-10-15)**: Höhe `row_h`→**8px** (Slim); Value-Sync — `_update_squares()` leitet Quadrate aus Wheel(H,S)+Bar(V) ab (Bar→Quadrat Echtzeit); gecachte `_src_rgb`/`_tgt_rgb` entfernt; `set_values()` ruft `_update_squares()` auf; `_rgb_hex()`-Helper |
| `dialog.py` | LOAD/SAVE AS Buttons, Custom Cursor, Ctrl+V |
| `dlg_worker.py` | `_saveas_out` Flag in `_apply_apply` |
| `cursors.py` | Ballon hohl (NoBrush); `loupe()` (Lupe) |
| `standalone/gui.py` | Suffix `_REFgrade` → `_grad` |

## 2026-10-16 — Save As
- SAVE AS öffnet **einen** Datei-Speichern-Dialog (PNG/JPEG/TIFF), Vorgabe `<Quelle>_grad.png`
  im zuletzt benutzten Ordner. Der frühere Ordner-Schritt (`getExistingDirectory`) entfällt
  (Nutzer: „das nervt“). Fehlt die Endung, kommt sie aus dem gewählten Filter.
- Schreiben nur mit Qt (`QImage`, U8 = ARGB32, U16 = RGBA64 + `rgbSwapped`), weil Kritas Python
  weder numpy noch PIL hat. Test: `core/tests/test_saveas_qt.py`.

## 2026-10-17 — Installer-Dialog Optik (K38h)
- **Layout:** alle Zeilen mit natürlicher Höhe von oben, **ein** `addStretch(1)` über der
  Knopfzeile (freier Raum sammelt sich unten). Abstände über `scale`/`style`, keine Pixelwerte.
- **Reihenfolge:** Titelzeile „REFgrade – set up runtime" (Branding: „REF" Akzent) ·
  „Detected: …" + message · „Build" + Combo · **Download-Label** „Download: approx. X.X GB ·
  needs about Y.Y GB free" (Y = `size_mb * 2`, wie `check_target`; kaufmännisch gerundet,
  250 MB → 0.3) · „Install to:" + Pfad + „Browse…" · Hinweis-Label (rot bei Problem) ·
  Log/Fortschritt (erst sichtbar bei Install) · Knopfzeile.
- **Download-Größe:** nur im Label, nicht mehr in den Combo-Einträgen (eine Quelle).
- **Knopfzeile:** „Use existing Python…" links, „Cancel" + „Install" rechts.
  **Entscheidung:** „Install" = `main="ok"` (Akzent-Füllung) — Spec sagt „Akzent (wie SAVE im
  Hauptdialog)", aber SAVE hat seit 2026-10-11 `main="cancel"` (Umriss). „Akzent" + Problem
  „nicht als Hauptknopf erkennbar" → Akzent-Füllung. (Offene Frage in `core/reports/paket_105.md`.)
- **Schema:** `schemes.apply_scheme(self)` im `__init__`, `win_chrome.apply` in `showEvent`
  (gleicher Weg wie Hauptdialog/Standalone).

## 2026-10-17 — Color Replace startet wieder ON
- **Bug:** Commit `89270b5` (2026-10-15, Debug-Runde „CR fixes … pipeline fix") hatte
  `dlg_color_replace.py` von `on=True` auf `on=False` umgestellt und so stehen lassen —
  die Sektion startete seither deaktiviert. Nutzerwunsch (seit `afd2dc6`, 2026-10-14):
  CR startet **ON**. Fix: `SectionHeader("color_replace", "Color Replace", on=True)`.

## 2026-10-17 — Offene Nutzerwünsche vom 2026-10-16 (Opus)
- **TONE immer offen:** TONE (Exposure … Dehaze = Lightroom „Basic") wird wie WHEELS **nie**
  automatisch eingeklappt (`_enforce_no_scroll`). Name bleibt TONE. Position bleibt an 3. Stelle,
  denn Panel-Reihenfolge = Verarbeitungsreihenfolge (REFERENCE/COLOR gleichen erst an, TONE
  danach). Wer TONE oben will, zieht es per Drag nach oben. Das Wiederöffnen nach Größenänderung
  probiert jetzt **alle** auto-eingeklappten Sektionen (passt eine nicht, wird die nächste versucht).
  Bei 1366×720 fehlen 12 px, sichtbar wird dabei nichts abgeschnitten.
- **Regler-Griffe:** In zu schmalen Zeilen (RGB-Raster, „Channels act on", Wheels-Sektion,
  Rad-Master) legte Qt das „?" über das rechte Reglerende. Bei Max war der Griff dann unsichtbar.
  `SliderRow._fit_slider` behält die Mindestbreite von 6 Zeilenhöhen, wo Platz ist, und gibt sie
  sonst auf. Ein kurzer Regler ist besser als ein verdeckter Griff.
- **Antialiasing:** Die Vorschau wird in der **dargestellten** Größe angefordert (längere Seite
  × `fit_zoom`), nicht mehr mit der längeren Seite des Bildbereichs. Vorher kam ein Hochformat in
  breiter Ansicht bis zu 2,5× zu groß an, und Qt skalierte bilinear (aliast ab etwa 2×) statt
  `area` im Core. Beim Ziehen bleibt es bei halber Auflösung (K29c). Das finale Bild ist
  ohnehin volle Auflösung.
- **SAVE AS-Name (2026-10-17):** Vorschlag `<Originalname>_rg.<Format des Originals>`
  (png/jpg/tif, sonst .png), in Plugin und Standalone. Die CLI nimmt `_rg` als Standard-Endung.
- **COLOR REPLACE-Helligkeitsbalken (2026-10-17):** Der Verlauf bleibt ein dünner Streifen, der
  Griff ist der runde Standard-Griff (10 px, TEXT, beim Hover ACCENT_HI) mit dunklem Rand, damit er
  auch am weißen Ende sichtbar ist.
