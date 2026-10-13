# RefGrade MCP-Server

Eine KI (Qwen Code, Claude Code, …) graded ein Bild über RefGrade: öffnen, Vorschau **sehen**,
Scope-Zahlen lesen, Regler setzen, AUTO je Kapitel, LUT/Bild exportieren.

## Start

```
cd <REFgrade>\core
python -m mcp_server
```

stdio-Server. Startet beim ersten Aufruf einen eigenen Worker (freier Port) und beendet ihn am
Ende. `RG_WORKER_PORT=<port>` → einen laufenden Worker benutzen statt einen eigenen zu starten.

## Eintragen

**Qwen Code** — `~/.qwen/settings.json`, unter `mcpServers`:

```json
"refgrade": {
  "command": "<path to python.exe>",
  "args": ["-m", "mcp_server"],
  "cwd": "<REFgrade>\\core"
}
```

**Claude Code** — `.mcp.json` im Projekt (oder `claude mcp add refgrade -- python -m mcp_server`
aus `core\`):

```json
{ "mcpServers": { "refgrade": {
  "command": "<path to python.exe>",
  "args": ["-m", "mcp_server"],
  "cwd": "<REFgrade>\\core" } } }
```

## Werkzeuge

| Werkzeug | Zweck |
|---|---|
| `open_image(path)` | Bild öffnen → `session`, Breite, Höhe |
| `sliders()` | alle Regler: Name, Kapitel, min, max, Standard, Schritt, Kurzhilfe |
| `preview(session, values, references?, method?, max_side=768)` | Vorschau als Bild + benutzte Werte (geclampt gemeldet) |
| `scopes(session, values, …)` | Perzentile je Kanal, Clipping, Farbstich je Zone (Lab a*/b*), Sättigung, Histogramm 16 |
| `auto(session, section, values?)` | AUTO-Vorschlag für tone / color / wheels / rgb / curves |
| `export_lut(session, values, directory)` | Grade als `.cube` |
| `export_image(session, values, out_path)` | volle Auflösung, PNG/TIFF/JPG nach Endung (ohne Clarity) |

`values` = `{"exposure": 0.3, "temperature": -0.1, …}`; nicht genannte Regler bleiben auf Standard.
Unbekannte Namen → Fehler mit der Liste der gültigen.

## Beispiel-Ablauf (für die KI)

1. `open_image("D:/bilder/haus.jpg")`
2. `scopes(s, {})` → Farbstich `cast.all.b = +8` (gelblich), Lichter-Clipping 3 %
3. `auto(s, "color")` → temperature/tint übernehmen; `scopes` erneut prüfen (b* ≈ 0?)
4. `preview(s, values)` ansehen, nachjustieren
5. `export_lut(s, values, "D:/luts")`

Tests: `core\tests\test_mcp_server.py` (inkl. Protokolltest über `mcp.client.stdio`).
