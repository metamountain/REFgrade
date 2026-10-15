# RefGrade — Konzept Krita-Plugin

Name: **RefGrade** (Nutzerentscheid 2026-10-07).

Stand: 2026-10-15. Voll gebaut und im Einsatz (Krita + Standalone).
Funktionsreferenz: `docs/FUNKTIONEN.md`. UI-Entscheidungen: `docs/UI_ENTSCHEIDUNGEN.md`.

## Aktueller Status

- **Kern** (`core/refgrade/`): Lab, Reinhard, MKL, Zonal, Fit, Color Replace, Clarity, Curves, HSL, LUT, Scopes — fertig, getestet.
- **Worker** (`core/worker.py`): HTTP-Dienst, alle Endpunkte live (siehe `docs/FUNKTIONEN.md`).
- **Plugin** (`plugin/ref_grade/`): Vollbild-Dialog, alle Sektionen, Color Replace mit Pipette, LOAD/SAVE AS, Cursor, Chain-Order.
- **Standalone** (`standalone/gui.py`): Gleicher Dialog ohne Krita, `_grad.png` Export.
- **Deployment**: Junction `%APPDATA%\krita\pykrita\ref_grade` → `plugin/ref_grade`.

## 1. Ausgangslage (geprüft)

- **Der Farbkern ist fertig und eigenständig.** `core/refgrade/` (Lab, Reinhard,
  MKL, Zonal, Fit, Galerie-Ranking, .cube-Export) mit 18 Testmodulen. Der letzte bekannte
  Stand war 1 roter Test (`test_known_sliders_recovery`, ΔE00 1.095 statt < 1.0).
- **Ziel ist Krita 6.0.4 (Beta, Qt6).** Der Nutzer testet es als Beta-Tester. Die
  API-Doku (`krita-dev-docs`) ist deshalb auf den Tag `v6.0.4`
  festgelegt; der installierte Build meldet `git c5b23e3` und kann leicht davon abweichen.
  Die Doku auf docs.krita.org nennt nur PyQt5 und ist für Qt6 veraltet. Die
  mitgelieferten Plugins importieren beides, PyQt6 und PyQt5. Das Plugin wird PyQt6-only
  geschrieben. API-Änderungen in späteren 6.x-Betas sind möglich, deshalb soll die
  Krita-Schicht dünn bleiben.
- **Kritas Python (3.13) enthält nur PyQt6, also weder numpy noch torch.** Der Kern
  braucht torch und läuft deshalb nicht im Krita-Prozess.
- **Das Systempython 3.12 hat torch 2.12 + CUDA und kornia** und kann den Kern also
  ausführen. Es fehlt nur `colour-science`, das ausschließlich die Referenztests brauchen.
- Vorbilder auf dieser Maschine:
  - `ai_diffusion` ist ein Krita-Client und redet mit einem externen Server. Dieses Muster
    übernehmen wir.
  - `vision_tools` ist natives C++ (ggml). Das ist für uns zu aufwendig.

## 2. Architektur: Plugin + Worker

```
Krita (Python 3.13, PyQt6)                 Worker (System-Python 3.12, torch/CUDA)
┌──────────────────────────┐   HTTP auf    ┌──────────────────────────────┐
│ ref_grade/ (pykrita)     │  127.0.0.1    │ core/worker.py  (stdlib-HTTP)│
│  Vollbild-Dialog (G'MIC) │ ────────────▶ │  importiert core/refgrade    │
│  Pixel lesen/schreiben   │ ◀──────────── │  Vorschau, Fit, Ranking, LUT │
│  startet Worker (QProcess)│  Bilder über  └──────────────────────────────┘
└──────────────────────────┘  Temp-Dateien
```

- **Worker:** Ein schlanker HTTP-Dienst nur mit `http.server` aus der
  Standardbibliothek. Endpunkte:
  - `health`
  - `references` (Pool ranken, Thumbnails)
  - `preview` (Grade auf Vorschaugröße)
  - `fit` (Regler aus Vorlage schätzen)
  - `apply` (volle Auflösung)
  - `cube` (.cube schreiben, Name nach Vorlage)

  Eine GPU-Sperre serialisiert die Rechenaufträge. Das Ranking der Vorlagen läuft
  außerhalb der Sperre, weil es sonst Vorschau und Fit blockiert. Gemessen ist das am
  Vorgänger mit einem simulierten 4-s-Ranking: Die Vorschau wartete dort 3,7 s.
- **Plugin:** Es ist dünn und enthält keine Mathematik.
  - Es liest Pixel mit `Node.pixelData` / `projectionPixelData` und verkleinert sie
    für die Vorschau mit `QImage.scaled`, was ohne numpy geht.
  - Bilder tauscht es über Temp-Dateien aus, weil beide Prozesse auf derselben Maschine
    laufen. Das ist einfacher als Bytes über HTTP.
  - Das Ergebnis schreibt es per `setPixelData` als **neue Ebene** über das Original.
    So bleibt das Original unangetastet, und die Ebenen-Deckkraft dient als globale Stärke.
- **Worker-Lebenszyklus:** Das Plugin startet den Worker beim ersten Gebrauch per
  QProcess und beendet ihn mit Krita. Ein Port, eine Instanz. Läuft schon einer, wird er
  wiederverwendet (Prüfung über `health`).
- Warum nicht im Krita-Prozess rechnen: Python ohne numpy ist für Millionen Pixel
  unbrauchbar langsam. numpy/torch in Kritas Python zu installieren ist fragil,
  versioniert sich mit jedem Krita-Update neu und lässt sich nicht testen.

## 3. Oberfläche: Vollbild-Dialog wie G'MIC (entschieden 2026-10-07)

Kein Docker. RefGrade öffnet sich wie G'MIC-Qt als **eigenes Fenster im Vollbild** (F11 wechselt, Esc = Cancel)
über dem Krita-Fenster (Menü *Filter → RefGrade…* und ein Tastenkürzel).
Das Fenster ist modal; solange es offen ist, wird in Krita nichts geändert.

Layout nach Nutzerskizze 2026-10-07, zweite Fassung (`layout landscape .png`): links
eine **Bedienspalte** mit etwa 1/3 der Breite, rechts die **Bildfläche** mit etwa 2/3, die
mit dem Fenster mitwächst.

```
┌──────┬──────┬──────┬─┬───────────────────────────────────────────┐
│ ▢    │ ▢    │ ▢    │ │░░░░░░░░░░░ Rand (Querformat) ░░░░░░░░░░░░░│
│ ▢    │ ▢    │ ▢    │ ├───────────────────────────────────────────┤
├──────┴───┬──┴──────┤ │                                           │
│ Methode  │ Regler  │ │                                           │
│ Stärke   │ …       │ │          BILD (Vorschau), eingepasst      │
│ FIT      │         │ │          Hochformat → Rand links/rechts   │
│ Regler … │ LUT…    │ │                                           │
│          │ ABBR OK │ ├───────────────────────────────────────────┤
├──────────┴─────────┤ │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ HISTOGRAMM         │ │                                           │
└────────────────────┴─┴───────────────────────────────────────────┘
```

- **Links oben: 6 Vorlagen-Plätze (3×2, quadratisch, Größe flex, Vorschaubild füllt die Kachel per Mittelausschnitt; gerechnet wird immer mit dem ganzen Bild)**, der aktive rot umrandet. Bedienung:
  - **Klick auf einen leeren Platz** öffnet den Dateidialog (Bild als Vorlage laden).
  - **Klick auf einen belegten Platz** aktiviert diese Vorlage.
  - **Rechtsklick** öffnet ein Menü: *Ersetzen…* (Dateidialog), *Neu laden*, *Leeren*.
  - **Drag & Drop** einer Bilddatei auf einen Platz belegt oder ersetzt ihn. Ein Drop auf
    die freie Fläche des Rasters belegt den nächsten freien Platz.
  - Die Belegung bleibt zwischen Sitzungen erhalten (Krita-Settings).
  - Leere Plätze füllt das Fenster beim Öffnen mit den bestgerankten Bildern (§6.6).
- **Darunter zwei Spalten:** Spalte 1 beginnt mit Methode (Reinhard/MKL/Zonal), Stärke und
  FIT, dann Regler; Spalte 2 enthält Regler und endet mit LUT… · ABBRECHEN · OK. Alle
  Regler aus `sliders.py`, gruppiert, ohne Schublade.
- **Links unten über die ganze Bedienbreite:** Histogramm (Quelle, Vorlage, Ergebnis).
- **Rechts:** das Bild, eingepasst. Leerraum neben dem Bild bleibt neutral.
  Maus gedrückt zeigt das Original, zuerst ganz eingepasst; Mausrad zoomt um den Zeiger, mittlere Maustaste verschiebt, ab 100 % pixelgenau aus dem Vollbild (`/crop`).
- **Gestaltung minimal: alles schwarz, einzige Farbe ist Metamountain-Rot `#FF1F1F`**, nur
  für die aktuelle Wahl. Text und Linien in neutralem Grau. **Schrift: Inter** für alles,
  Zahlen mit Tabellenziffern (`tnum`).
- Die Vorschau rechnet auf die Bildschirmgröße des Bildfelds und beim Zoomen auf den
  sichtbaren Ausschnitt, wie G'MIC.
- Fenstergröße und -position merkt sich das Plugin (Krita-Settings); Start im Vollbild.

Ablauf:

1. Ebene in Krita wählen, *Filter → RefGrade…*.
2. Links oben die Vorlagenbibliothek (`bench/pool`), gerankt nach Passung. Eine Vorlage
   anklicken, die gewählte ist rot markiert.
3. Die Regler kommen aus der einen Tabelle in `sliders.py`. Methode ist standardmäßig
   Reinhard (Lab), MKL ist wählbar. Dazu gibt es „Fit“ (Regler schätzen) und Stärke.
4. **LUT-Ordner:** Krita hat keinen eigenen LUT-Ordner. Es kennt nur den OCIO-LUT-Pfad im
   Docker *LUT-Verwaltung*, und der wirkt nur auf die Anzeige; bei dir ist er leer
   (`kritarc`: `OcioLutPath=`, `UseOcio=false`). **Standard:** `<Krita-Ressourcenordner>\luts`,
   also `%APPDATA%\krita\luts`. Den Ressourcenordner liest das Plugin
   aus `kritarc` (`ResourceDirectory`), statt ihn fest einzutragen. **Manuell änderbar:** im
   Speichern-Dialog von *LUT…*; der zuletzt gewählte Ordner wird gemerkt.
5. **OK** erzeugt die Ebene „REF · <Vorlagenname>“ in voller Auflösung und schließt das
   Fenster. **LUT…** schreibt `<Vorlagenname>.cube`. **Abbrechen** lässt alles, wie es
   war.
6. Optional: Eine **Krita-Auswahl** wird als Ignorier-Maske für die Statistik verwendet,
   nicht für die Anwendung. Das entspricht der Messung aus `spec/00`: Die Maske gilt
   nur für die Statistik.

## 4. Bewusst nicht in Version 1

- Vorschau auf der Krita-Leinwand selbst (G'MIC kann das optional; wir zeigen nur im
  eigenen Fenster).
- Float- und Linear-Dokumente (F16/F32, Nicht-sRGB). Version 1 kann RGBA U8/U16 in sRGB;
  alles andere lehnt sie mit einer klaren Meldung ab.
- Die .cube-Datei in Krita als Filterebene anwenden. Krita kann .cube nur über OCIO als
  Anzeige-Transformation nutzen, nicht als Ebene. Deshalb werden die Pixel geschrieben.

## 5. Pakete (Pipeline, Economy-Modus)

Specs: `spec\21_K1_…` bis `spec\26_K6_…`, Reihenfolge und Regeln in `QWEN.md`.

| Nr | Paket | Test ohne Krita? |
|---|---|---|
| K1 | Rohdaten-I/O + Sitzungslogik: ein LUT-Rechenweg für Vorschau, Anwenden, `.cube` | ja |
| K2 | `core/worker.py`: HTTP-Endpunkte, GPU-Sperre, Ranking außerhalb der Sperre | ja |
| K3 | Plugin-Gerüst: `.desktop`, Menü + Kürzel, Worker-Start, Client, leeres Fenster | Logik ja |
| K4 | Pixelbrücke: aktive Ebene → Worker → neue Ebene „REF · …“ | Regeln + Ende-zu-Ende ja |
| K5 | Das Fenster nach Skizze (offscreen getestet, Screenshots) | Logik + Geometrie ja |
| K6 | Tooltips, `install.py`, README | ja |

Die Plugin-Logik liegt in Modulen ohne `krita`-Import und ist mit dem Systempython testbar.
Dafür ist dort PyQt6 6.8.1 installiert, dieselbe Version wie in Krita. Was nur in Krita
prüfbar ist, prüft der Nutzer per Auge. Installation: `install.py` legt die Junction
`%APPDATA%\krita\pykrita\ref_grade` → `plugin\ref_grade` an.

## 6. Entscheidungen (2026-10-07; „mach mal“ = Empfehlungen übernommen)

1. Oberfläche: Vollbild-Fenster wie G'MIC, Layout nach Skizze (§3).
2. Ergebnis: **neue Pixelebene** „REF · <Vorlage>“ über der aktiven Ebene, plus `.cube`-Export.
3. Quelle: **aktive Ebene** (ihre Projektion, also inklusive Masken).
4. Worker-Interpreter: **Systempython 3.12** (torch + CUDA).
5. Vorlagen: Bibliothek `bench\pool` + von Hand belegte Plätze; keine offenen Krita-Dokumente
   in Version 1.
6. Leere Vorlagenplätze werden beim Öffnen mit den bestgerankten Bildern aus `bench\pool`
   gefüllt; von Hand belegte bleiben.
7. LUT… / ABBRECHEN / OK sitzen unten in Reglerspalte 2; die Bildfläche zeigt nur das Bild.
8. Vorschläge-Raster 3×3 (Methode × Stärke) kommt nicht in Version 1.
