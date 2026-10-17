# RefGrade — Torch & ONNX Faktensammlung

Stand: 2026-10-16. Nur lesen/analysieren, keine Änderungen.
Quellen: `core/refgrade/{session,grade,lut,color,onnx_lut,clarity,texture,dehaze,rawio}.py`,
`standalone/gui.py`, `core/worker.py`, `plugin/ref_grade/dlg_worker.py`,
`core/tests/test_onnx_lut.py`, `build/build_all.py`.

---

## 1. Torch-Implementierung

### Fakten (was ist)

**1.1 Preview-Kette (Torch-Operationen)**

Die Preview (`session.py:preview`, 672–777) rechnet so:

1. `_preview_source` (780–795): Source auf ≤ `max_side` skalieren — `F.interpolate`
   **bilinear**, `align_corners=False` (788–791). Gecacht pro Größe.
2. `build_lut` (514–578) → `_build_lut_cached` (599–644): LUT-Gitter `[N,N,N,3]` aus
   `torch.linspace`/`expand`/`stack` (604–610), dann die Kette der Bereichsfunktionen
   **auf dem Gitter** durchlaufen, Präfix-Cache (LRU, max 24).
3. `_lut_apply` → `lut.apply` (lut.py:48–79): **`F.grid_sample` mode="bilinear",
   `align_corners=True`** auf der 5D-LUT (75–77) — trilineare 3D-Interpolation.
4. Räumliche Effekte (nur `stage="full"`, 730–740): `clarity_apply`, `texture_apply`,
   `dehaze_apply` — keine LUT, sondern Konvolutionen.

Bereichsfunktionen (`_section_fns`, 399–490), alle sRGB→sRGB pro Pixel:

| Bereich | Funktion | Torch-Op |
|---------|----------|----------|
| reference | `protect(x, method_fn(x), v)` | reinhard/mkl (`transfer.py`), zonal (Lab), `fit_pair` |
| tone/color/wheels/hsl | `grade(x, v)` (grade.py) | `srgb_to_linear`/`linear_to_srgb`, kornia `rgb_to_lab`/`lab_to_rgb`, `rgb_to_hsv`/`hsv_to_rgb`, `luma` |
| rgb | `rgb_grade` (rgb.py) | Lift/Gamma/Gain, Zonen |
| lut | `lut.apply` | `grid_sample` bilinear |
| curves | `curves.apply` + `huecurves.apply` | Punkt-Interpolation |
| color_replace | `color_replace.apply` | HSL-Distanz-Maske, mehrere Instanzen |

**1.2 GPU (CUDA) vs CPU**

- Device-Wahl: `open_session` (90–91) — `cuda:0` wenn `torch.cuda.is_available()`,
  sonst `cpu`. Source wird per `.to(device)` (118) auf das Device gezogen.
- LUT-Gitter + LUT-Bau + LUT-Apply + räumliche Effekte: **alles auf
  `s.source.device`** (603–604). Also CUDA, falls verfügbar.
- `torch.cuda.synchronize()` nur für Timing (425, 575, 719, 724, 742, 771).
- Histogramme: `.cpu()` + `torch.histogram` (651, 659–661) — **CPU**.
- Referenzen: `rawio.load_image` lädt auf CPU, Pooling zieht `.to(device)` (212).
- `apply_full` (804–850): läuft auf **CPU** (`source_full` ist CPU), chunked in
  ≤ `APPLY_CHUNK`=4 M Pixel (41, 827–837).
- Torch-Setup (K38e, `standalone/gui.py` + `plugin/ref_grade/runtime.py`): Torch wird
  **vor** dem PyQt6-Import geladen, sonst WinError 1114 (DLL-Konflikt). Ohne torch zeigt
  die Standalone den `dlg_install.InstallDialog` (PyQt6) und startet danach mit dem
  Runtime-Python neu. Torch-Builds von `download.pytorch.org/whl/<flavor>`
  (`runtime.FLAVORS`: cu130/cu126/rocm/xpu/cpu).

**1.3 Preview-Skalierung (PREVIEW_SIDE=1536)**

- `PREVIEW_SIDE = 1536` (session.py:36), `REF_SIDE = 768` (37), `LUT_SIZE = 33` (38).
- **Zwei** Skalierungsschritte:
  1. `open_session` (96–105): `source_full` (volle Auflösung) → `source` ≤ 1536,
     **bilinear** `align_corners=False`. Das ist die gecachte Preview-Basis.
  2. `preview` → `_preview_source` (784–794): `source` (≤1536) → ≤ `max_side`,
     **bilinear** `align_corners=False`.
- `max_side` kommt aus dem Dialog `_max_side()` (dlg_worker.py:98–102): längste Seite
  des **Bildbereichs in Device-Pixeln**, beim Ziehen halbiert (K29c). Also: 1536 ist
  nur die Session-Cap, die eigentliche Preview-Auflösung ist die View-Größe.
- Beim Zoomen über die Preview-Auflösung: `_trigger_crop` (207–235) → `/crop` mit
  Vollbild-Ausschnitt, dann bilinear auf `max_side` (session.py:986–989).

**1.4 Interpolations-Methode**

- **Bildskalierung (Source/Preview/Crop/Maske):** überall `F.interpolate`
  **bilinear**, `align_corners=False` (session.py:102, 114, 312, 790, 988; masks.py:35;
  fit_pair.py:389, 397).
- **LUT-Apply:** `F.grid_sample` **bilinear**, `align_corners=True` (lut.py:76).
- **Referenz- & Thumbnail-Loading:** PIL **LANCZOS** (rawio.py:122, session.py:914,
  reference_rank.py:46) — das ist der **einzige** Lanczos im System, und er greift
  **nicht** auf den Source-Preview-Pfad.
- Torch `F.interpolate` unterstützt `nearest`, `linear/bilinear`, `bicubic`, `trilinear`,
  `area` — **kein Lanczos**.

**1.5 Antialiasing**

- **Kein explizites Antialiasing / Multi-Phase-Downscale** im Source-Preview-Pfad.
- Bilinear ohne Low-Pass vor der Dekimation → Aliasing (Moiré, Detailverlust),
  sobald der Downscale-Faktor ≫ 2× liegt.
- Wo es nötig wäre: `open_session` (full→1536, bei 6000 px ≈ 4×), `_preview_source`
  (1536→view), `crop` (Vollbild→max_side).
- LUT-Apply ist ein pro-Pixel-Farbtransform (kein Downscale) → dort kein AA nötig.
- Räumliche Effekte (clarity/texture) blurren Gaussian, aber das ist der Effekt,
  kein AA für den Downscale.

### Probleme (was schrottig ist)

- **P1:** Großer Downscale (full→1536) mit Bilinear ohne AA → Aliasing bei großen
  Quellen. Das ist die Hauptursache für „pixelig/weich".
- **P2:** 1536-Cap: große Quellen verlieren Detail **vor** dem Grading. Auf großen
  Monitoren/hoher DPI wird die Preview dann von Qt hochskaliert (weich).
- **P3:** Doppelte Bilinear-Kette (full→1536→view) ohne AA in beiden Stufen.
- **P4:** (K38e behoben) `torch_check.py` gelöscht; der irreführende Text „GPU-Features
  (Color Replace) brauchen PyTorch" existiert nicht mehr.
- **P5:** `apply_full` auf CPU (827–837) — bei großen Bildern langsam, obwohl CUDA
  verfügbar ist (LUT-Apply selbst ist device-agnostisch).

### Optionen (was möglich wäre)

- **O1:** `F.interpolate(mode="area")` für den großen Downscale (full→1536). `area`
  ist ein echtes AA-Filter (Box-Over-Sampled), läuft auf GPU, 1-Zeilen-Änderung.
  Bester Effort/Aufwand.
- **O2:** `mode="bicubic"` statt bilinear — schärfer als bilinear, aber kein echtes AA
  (kann Ringe erzeugen).
- **O3:** Multi-Phase-Downscale (wie Photoshop/GIMP): Schrittweise Halbieren mit
  Box/Gaussian-Low-Pass dazwischen, bis ≤ ~1,5× Ziel, dann ein finaler bilinear/bicubic.
  Goldstandard für AA, mehr Code.
- **O4:** `PREVIEW_SIDE` erhöhen (1536→2560/3072) — mehr Detail, mehr VRAM + langsamer.
  Hilft gegen P2, **nicht** gegen P1 (Aliasing).
- **O5:** `apply_full` auf CUDA (device-agnostischer LUT-Apply) — schneller bei großen
  Bildern.

### Empfehlung (was ich tun würde)

1. **O1 zuerst:** `open_session` (100–103) und `crop` (986–989) auf `mode="area"`
   umstellen. Das behebt das Aliasing (P1/P3) mit minimalem Aufwand und GPU-Performance.
   `_preview_source` (1536→view, meist < 2×) kann bilinear bleiben.
2. **O4 ergänzend:** `PREVIEW_SIDE` auf 2560, wenn die Ziel-Monitore groß sind (P2).
3. **O5:** `apply_full` auf CUDA ziehen, wenn Full-Res-Apply zu langsam ist (P5).
4. **O3** nur, wenn O1+O4 nicht ausreichen (z. B. 8K-Quellen).

---

## 2. ONNX-Status

### Fakten (was ist)

- **`onnx_lut.py` ist ein Prototyp** (Commit `89270b5`), **nicht** in die
  Preview/Apply-Pipeline eingebunden. Nur `test_onnx_lut.py` nutzt ihn.
- **Zwei Modelle** (beide opset 17, IR 8):
  - `create_identity_model` (25–35): `result = image` — reiner Overhead-Smoke-Test.
  - `create_lut_apply_model(lut_size=33)` (41–179): **echte trilineare 3D-LUT-Interpolation**,
    äquivalent zu `refgrade.lut.apply` (grid_sample, align_corners=True).
    Ops: Shape, Gather, Mul, Concat, Transpose, Reshape, Floor, Add, Sub, Clip, Cast,
    Squeeze, Expand, GatherND, Constant.
- **Konvention:** Input `image [1,3,H,W]` + `lut [1,3,N,N,N]` (Raumachsen D=b, H=g, W=r),
  Output `result [1,3,H,W]`. Die LUT-Tabelle ist ein **Runtime-Input**, nicht gebacken.
- **Keine `.onnx`-Dateien im Repo.** `core/refgrade/models/` existiert nicht.
  Modelle werden in Tests in-memory (tmp) erzeugt; `run_onnx_inference` (215–223)
  speichert on-demand, wenn die Datei fehlt.
- **Provider:** `run_onnx_inference` default `CPUExecutionProvider` (216). Kein CUDA-Provider
  verdrahtet.
- **Abhängigkeiten optional:** `onnx` / `onnxruntime` — Tests skippen, wenn nicht installiert
  (test_onnx_lut.py:20–21, 51, 77).

### Was kann ONNX aktuell?

- **Nur der LUT-Apply-Schritt** (trilineare 3D-Lookup). Das ist der **letzte** Schritt
  der Kette.
- **Nicht** möglich mit dem aktuellen Modell:
  - LUT-**Bau** (Kette der Bereichsfunktionen auf dem Gitter) — bleibt Torch.
  - Color Replace, Curves, Zonal, Transfer, räumliche Effekte — keine ONNX-Exporte.
  - Die LUT muss also trotzdem von Torch gebaut und als Input übergeben werden.

### Was fehlt für eine vollständige ONNX-Pipeline?

- **F1:** ONNX-Export der Bereichsfunktionen (grade, transfer, zonal, curves,
  color_replace, rgb) — sonst bleibt der LUT-Bau in Torch.
- **F2:** Color Replace (HSL-Distanz-Maske, mehrere Instanzen) — komplex, nicht exportiert.
- **F3:** Räumliche Effekte (clarity/texture/dehaze) — nicht LUT-basiert, nicht exportiert.
- **F4:** CUDA-Provider (DirectML/CUDA ExecutionProvider) für GPU-ONNX.
- **F5:** Integration in `worker.py`/`session.py` — kein Code-Pfad ruft das Modell auf.
- **F6:** Build-Schritt: `build/build_all.py` baut nur Plugin-ZIP + PyInstaller-Standalone,
  **kein** ONNX-Model-Export. `tools/build_zip.py` ebenfalls.

### Performance-Vergleich Torch vs ONNX

- **Keine committed Benchmarks-Zahlen** im Repo.
- `test_onnx_lut.py::test_benchmark` (113–165): 1024×768, LUT 33³, 5 Runs, misst
  ONNX(CPU) / Torch(CPU) / Torch(CUDA) in ms + Ratio ONNX/Torch-CPU. **Print-basiert**,
  Ergebnis wird nicht gespeichert.
- Korrektheit: `test_correctness_vs_torch` (101–110) — ONNX vs Torch max err < 1e-5 (bestanden).
- → Konkreter Performance-Vergleich ist **nicht dokumentiert**; nur das Harness existiert.

### Probleme

- **P6:** Prototyp ohne Integration — kein Mehrwert in der laufenden App.
- **P7:** Nur LUT-Apply, nicht der LUT-Bau → ONNX kann die Pipeline nicht ersetzen,
  höchstens den Apply-Schritt beschleunigen (falls CUDA-ONNX schneller als Torch-CUDA).
- **P8:** Kein CUDA-Provider → ONNX nur auf CPU, dort in der Regel **nicht** schneller
  als Torch-CUDA.
- **P9:** Keine Benchmark-Zahlen → kein Beleg für einen Performance-Gewinn.
- **P10:** `onnx`/`onnxruntime` sind optionale Zusatz-Abhängigkeiten ohne Nutzen in der App.

### Optionen

- **O6:** ONNX **einfach löschen** (Prototyp, kein Nutzen, keine Integration) — reduziert
  Abhängigkeiten + Test-Fläche.
- **O7:** ONNX **fertig ausbauen**: Bereichsfunktionen exportieren, CUDA-Provider,
  Integration, Build-Schritt, echte Benchmarks. Großer Aufwand.
- **O8:** ONNX **behalten als Referenz/Benchmark-Harness**, aber klar als Prototyp markieren.

### Empfehlung

- **O6** (löschen), **es sei denn** es gibt einen konkreten Plan für eine
  Torch-freie/leichte Inference (z. B. Krita-Integration ohne PyTorch, Web-Export).
  Dann: **O7** mit Priorität auf F1+F4+F5+F6 und **echten** Benchmarks (P9), bevor
  weiter investiert wird. Solange kein Use-Case steht: Prototyp ist Ballast.

---

## 3. Preview-Qualität

### Fakten

- `PREVIEW_SIDE = 1536` (session.py:36) cappt die Source **vor** dem Grading.
- Skalierung: full→1536 (bilinear, `open_session` 100–103) → 1536→view (bilinear,
  `_preview_source` 788–791). Beide `align_corners=False`, **kein AA**.
- LUT-Apply (`lut.apply` 75–77): `grid_sample` bilinear — pro-Pixel-Farbtransform,
  **kein** Blur, also **nicht** die Ursache von Pixeln.
- LUT-Größe: 33³ normal, **17³ beim Ziehen** (dlg_worker.py:104–106) — gröbere
  Farbquantisierung während des Drags (Banding, nicht räumliche Pixelung).
- Referenzen werden mit PIL **LANCZOS** auf 768 geladen (rawio.py:122) — schärfer
  als der bilineare Source-Pfad, aber nur für Referenzen.

### Warum ist die Preview pixelig?

- **Hauptursache:** Bilinear-Downscale ohne Antialiasing. Bei 6000 px → 1536 (≈ 4×)
  fehlts dem Low-Pass → Aliasing (Moiré, Treppchen, Detailverlust). Das wirkt „pixelig".
- **Zweitursache:** 1536-Cap. Große Quellen verlieren Detail vor dem Grading; auf
  großen Monitoren wird die Preview von Qt hochskaliert (weich/pixelig).
- **Nicht** die Ursache: LUT-Apply (Farbtransform, kein Blur) und die 17³-Drag-LUT
  (das ist Farb-Banding, ein anderes Symptom).

### Optionen

| Option | Wirkung | Aufwand | GPU |
|--------|---------|---------|-----|
| Höheres `PREVIEW_SIDE` (2560/3072) | mehr Detail, behebt Cap (P2) | niedrig | ja, mehr VRAM |
| `F.interpolate(mode="area")` | echtes AA, behebt Aliasing (P1) | **1 Zeile** | ja |
| `mode="bicubic"` | schärfer als bilinear, kein echtes AA | 1 Zeile | ja |
| Multi-Phase-Downscale (Halbieren + Low-Pass) | Goldstandard-AA | mittel | ja |
| GPU-Antialiasing (Gaussian vor Bilinear) | echtes AA, mehr Kontrolle | mittel | ja |
| LANCZOS (PIL) für Source | schärfer, aber CPU + kein GPU-Pfad | mittel | nein |

### Empfehlung

1. **`mode="area"`** für den großen Downscale (`open_session` 100–103, `crop` 986–989).
   Das ist der größte Qualitätsgewinn pro Aufwand (1 Zeile, GPU, echtes AA).
2. **`PREVIEW_SIDE` → 2560**, wenn die Ziel-Monitore > 1536 px in Device-Pixeln sind.
3. **Multi-Phase-Downscale** nur als Nachschlagewerk für 8K-Quellen, wenn 1+2 nicht
   reichen.
4. LANCZOS für die Source **nicht** — bricht den GPU-Pfad und ist für den Downscale
   nicht besser als `area`.

---

## Zusammenfassung (kern)

- **Torch:** Preview = bilineare Skalierung (2×, kein AA) + LUT-Bau auf dem Gitter +
  `grid_sample`-Apply + optionale räumliche Effekte. Alles auf CUDA (falls verfügbar),
  nur Histogramme + `apply_full` auf CPU.
- **ONNX:** Prototyp, **nur** LUT-Apply (trilinear), **nicht** integriert, keine
  `.onnx`-Dateien, kein CUDA-Provider, keine Benchmark-Zahlen. Ballast ohne Use-Case.
- **Preview-Qualität:** pixelig wegen **Bilinear ohne AA** beim großen Downscale +
  **1536-Cap**. Fix: `mode="area"` (1 Zeile) + ggf. höheres `PREVIEW_SIDE`.
- **Top-3 Aktionen:** (1) `area`-Interpolation im Downscale, (2) `PREVIEW_SIDE`→2560,
  (3) ONNX-Prototyp löschen oder mit echtem Plan + Benchmarks ausbauen.

---

## 4. Messung LUT-Apply ohne Torch (2026-10-16, Pilot)

Bester von 3 Läufen. Rechner: RTX 4090 (VRAM durch BeeLlama zu 22,4/24,5 GB belegt) + AMD Radeon iGPU.
Skripte im Session-Scratchpad (gs3d.py, dmldev.py, torch_ref.py); ORT-CPU 1.31, ORT-DirectML 1.24.4.

| Variante | 1536×1024, 33³ | 6000×4000, 33³ | Genauigkeit vs Torch |
|---|---|---|---|
| Torch CPU (`lut.apply`) | 81 ms | 1234 ms | Referenz |
| Torch CUDA (4090) | 0,6 ms | 15 ms | Referenz |
| ONNX CPU, Op GridSample opset 20 (5D) | 89 ms | 1094 ms | 1,8e-7 |
| ONNX DirectML 4090, GatherND-Graph (`onnx_lut.py`) | ~670 ms (17³) | 11 s | – |
| ONNX DirectML Radeon iGPU, GatherND | 460 ms (17³) | GPU-Hang (887A0006) bei 33³ | – |

- DirectML kann GridSample in 3D nicht und fällt auf die CPU zurück (per Profiling geprüft).
- Der GatherND-Graph läuft auf DirectML, ist aber langsamer als die CPU. Auf der iGPU löst er
  bei 33³ einen Treiber-Reset aus (TDR).
- Ergebnis: ONNX ist auf der CPU gleichwertig mit Torch CPU. Einen GPU-Gewinn bringt es in dieser
  Form nicht. Der schnelle GPU-Weg für eine 3D-LUT wäre eine 3D-Textur mit Hardware-Trilinear im
  Shader (Qt/OpenGL), nicht ONNX.
- Noch offen: eine numpy-Implementierung zum Vergleich.

**Korrektur 2026-10-16:** Die Aussagen in §1.3–§1.5 und §3 zu „bilinear ohne Antialiasing“ sind
überholt — alle Downscales in `session.py` (Z. 101, 113, 311, 789, 987) nutzen inzwischen
`mode="area"` (= Empfehlung O1, umgesetzt).
