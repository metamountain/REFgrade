# Color Replace – Notes & Pitfalls

## Stale Worker (PITFALL)

Vor jedem Krita-Test: `python core/tools/kill_stale_workers.py`

Verwaiste Worker halten VRAM und senden alte Werte. Immer zuerst killen.

## 2026-10-14: CR-Logik überarbeitet

### Neue Semantik

| Parameter | Bedeutung | Range | Default (Core) | Default (UI) |
|-----------|-----------|-------|----------------|--------------|
| `src_hue` | Source-Farbe (Hue) | 0–360 | 0 | 0 |
| `src_sat` | Source-Farbe (Sat) | 0–1 | 1.0 | 1.0 |
| `src_bright` | Source-Farbe (Value) | 0–1 | 1.0 | 1.0 |
| `tgt_hue` | Target-Farbe (Hue) | 0–360 | 0 | 240 |
| `tgt_sat` | Target-Farbe (Sat) | 0–1 | 1.0 | 1.0 |
| `tgt_bright` | Target-Farbe (Value) | 0–1 | 1.0 | 1.0 |
| `intensity` | Stärke der Ersetzung | 0–**2** (2026-10-15) | **0.0** (inaktiv) | 0.5 |
| `range` | Hue-Bereich in Grad | 5–180 | 60 | 60 |

### Maske

```
t = hue_dist / range
plateau = 0.6
mask = 1.0                          wenn t < plateau
mask = (1 - t) / (1 - plateau)      sonst
mask *= sat_guard * src_sat * intensity
```

- `src_sat=0` (Center/Schwarz) → **kein Effekt** (PITFALL: ohne src_sat in Maske werden Pixel trotzdem verändert)
- `intensity=0` → Instanz inaktiv (Core-Default, verhindert dass leere Instanzen 1–7 stören)

### UI-Layout

- 2-Spalten-Grid, füllt links→rechts, neue Reihe wenn voll
- Start: 1 Instance, `+ Add color` fügt 1 hinzu, Max 8
- Wheels: ~136px (8×row_h), Badge 24×24 oben-links (Pipette-Trigger)
- Cursor: Pipette (Source), Sniper-Fadenkreuz (Target)
- Add-Button in `_content` → versteckt wenn Section minimiert

### Hue-Mapping (PITFALL)

Qt ConicalGradient geht im Screen-CW (0°=rechts, 90°=unten, 270°=oben).
`atan2(v, u)` gibt Math-CCW. **Invertierung nötig:**

```python
# wheel → hsv
hue = (-math.degrees(math.atan2(v, u))) % 360.0

# hsv → wheel
wheel.set_point(r * cos(rad), -r * sin(rad))
```

Ohne Invertierung: Wheel oben-links = Grün statt Blau.

### Headless-Test

```bash
cd core && python -m pytest tests/test_color_replace.py -v
```

5 manuelle Checks (src_sat=0, red→blue, plateau, edge, out_of_range) – alle PASS.

## 2026-10-15: UI-Änderungen

- **Section startet OFF** (`on=False`), wie alle anderen Sektionen (DimGuard schaltet bei
  erstem Klick ON).
- **Close-Button (×)**: `QToolButton` 16px oben-rechts an jeder Instanz **außer #1**.
  Klick → `intensity = 0` + Instanz ausgeblendet. Nutzt die bestehende Core-Regel
  `intensity ≤ 0 → Instanz wird übersprungen` (`color_replace.apply`).
- **Source/Target-Buttons entfernt**: Farbfelder (`_ColorSquare`) sind der einzige
  Pick-Trigger; Loupe-Cursor (`cursors.loupe()`) auf Hover.
- **Intensity**: Range 0–2, Default 0.5. **Wheels**: ×0.7 skaliert.

## 2026-10-15: Brightness-Fix (Nutzer: „CR brightness nicht korrekt verknüpft, soll Farbton beeinflussen!")

### Problem

1. `src_bright`-Gate zu breit (Halbfenster 0.5): bei 0.5 matchte der **ganze**
   Helligkeitsbereich (nicht „nur Midtones"), bei 1.0 die Hälfte (nicht „nur Highlights").
2. `tgt_bright`-Skalierung ungeclampt: `V_tgt = V * tgt_bright` > 1 → RGB-Werte > 1.0
   (z. B. tgt_bright=1.5 bei V=1 → R=1.5).
3. **Pipette verknüpfte die Helligkeit des gepickten Pixels nicht**: `set_src_from_rgb` /
   `set_tgt_from_rgb` setzten nur Hue+Sat, die Brightness-Bars blieben auf 1.0 →
   gepickte Midtones/Schatten matchten nie (Gate=0).

### Fix

- **`src_bright` (SELECTION)**: Proximitätsfilter mit Halbfenster **0.25**:
  `bright_gate = 1 - |V - src_bright| * 4` clamped 0..1
  - 1.0 → V∈[0.75, 1.0] (volle Helligkeit), 0.5 → V∈[0.25, 0.75] (nur Midtones),
    0.0 → V∈[0, 0.25] (nur Schatten)
- **`tgt_bright` (OUTPUT)**: `V_tgt = (V * tgt_bright).clamp(0, 1)` — skaliert die
  OUTPUT-Helligkeit (1.0=normal, 0.5=dunkler, >1.0= heller bis Weiß).
- **Final-Clamp** am Ergebnis von `_replace_one`: intensity > 1 (Overshoot) kann über
  0..1 hinausreichen → `(x + mask * (tgt_rgb - x)).clamp(0, 1)`.
- **Pipette-Verknüpfung** (`dlg_color_replace.py`): Pick setzt die zugehörige
  Brightness-Bar auf `V = max(r, g, b)` des gepickten Pixels → die gepickte Farbe
  matcht sich selbst (src) bzw. die Ersetzung bekommt die gepickte Helligkeitsstufe (tgt).

### Tests

- `tests/test_color_replace.py`: 2 neue Tests (`test_src_bright_selects_brightness_range`,
  `test_tgt_bright_scales_output_brightness`) → 10/10 PASS.
- Funktionstest: reines Blau (V=1) → Rot: tgt_bright 0.5/1.0/1.5 → R=0.5/1.0/1.0 (geclampt).
