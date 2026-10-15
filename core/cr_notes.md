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
| `intensity` | Stärke der Ersetzung | 0–1 | **0.0** (inaktiv) | 1.0 |
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
