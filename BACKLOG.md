# RefGrade — Backlog (wishes not yet specified)

One line per wish: date, wish, status. Once it becomes a package, it gets its spec number
here.

- 2026-10-08 — **Reset per colour curve:** a ↺ per curve channel (RGB, R, G, B, HvH, HvS,
  HvL, LvS, SvS) that resets only that curve; the section's ↺ still resets all of them.
  *Spec'd 2026-10-10 → see QWEN.md K28a/K28b.*
- 2026-10-08 — **Non-destructive (K16):** decide between the annotation workaround (spec 39)
  and our own C++ "3D LUT" filter plugin for Krita (needs the Krita build to be recompiled).
  *Waiting on the user's decision.*
- 2026-10-08 — **PAIR warning via alignment** instead of fit error: phase-correlation shift of
  the edge images > 5 px → "Before and after do not line up" (measured in
  `reports/pair_fivek.md`). *Spec'd 2026-10-10 → K28c (spec/69).*
- 2026-10-09 — **Histogram area tool (own rectangle):** add a small "rectangle" tool that lets
  the user draw a rectangle on the image; the histogram is then computed only from that
  rectangle (not the whole image, not the Krita selection). Drag to move/resize, a button to
  clear it back to full image. *Spec'd 2026-10-10 → K28b. **Rejected by the user 2026-10-12** (wrong place, no use) — removed.*
- 2026-10-09 — **Clipping warning in histogram with value:** a small warning indicator inside
  the histogram when a channel clips (values pinned at 0 or 255), showing the clipped count
  (e.g. "R clip 12 %"). *Spec'd 2026-10-10 → see QWEN.md K28a/K28b.*
- 2026-10-12 — **Bereiche als Kette statt parallel** (Räder sehen die Helligkeit nach TONE), Reihenfolge per Drag & Drop. *Kern von Opus gebaut (`sections.chain_fn`, Präfix-Cache); UI → K35a (spec/90).*
- 2026-10-13 — **FR: Layout V — scope block flush with the wheel column, pixel-exact.** The
  right edge of the histogram/parade block must line up exactly with the right edge of the
  wheel column (user: "knapp daneben tut weh"). Seen in the 2026-10-13 screenshot at 2560×1392:
  the WHEELS chapter (Intensity/Blending/Balance + `?`) reaches ~20 px past the scope block,
  because `_wheel_squares` (`dlg_layout.py`, V branch) lets `sec_w` grow to its `sizeHint`.
  Not measured exactly yet — measure all right edges (scope block, wheel discs, ↺, masters,
  WHEELS chapter) first, then make them one edge; pin with a test. *Open, not started.*
- 2026-10-15 - **GPU-ONNX Pipeline (Fortsetzung):** LUT-Modul ist als ONNX-Prototyp gebaut
  (`core/refgrade/onnx_lut.py`, Trilinear == Torch, see KONZEPT.md). Offen: `onnxruntime-gpu`
  (CUDAExecutionProvider) installieren, dann die übrigen Module (Curves, HSL, …) portieren und
  gegen Torch CUDA benchmarken. *Open, not started.*
- 2026-10-16 — **Wheels-Fix Lücken:** alle 3 Punkte bereits gefixt (verifiziert 2026-10-16). *Erledigt.*

## Debug-Runde 2026-10-17 — offen (Sonnet-Review, Berichte in tmp/debug/report_*.md)
- `protect_midtones`: Regler + Hilfe existieren, fehlt aber in SECTIONS/Dialog → wirkungslos.
  Aufnehmen braucht: SECTIONS, Plugin-Kopie (help_structure), Dialog-Zeile, Zähltests (131→132).
- color_strength wirkt bei Reinhard/MKL nicht (session._build_method_fn).
- Worker-Temp-Dateien (preview_/parade_/wave_/thumb_/clarity_mask_) werden nie gelöscht.
- Session-Caches (s.luts, s.previews, s.method_fns) ohne Obergrenze (VRAM).
- Ignore-Maske wirkt nicht auf FIT (fit_sliders._subsample verwirft Gewichte).
- reference_rank lädt jedes Bild vor der Cache-Prüfung (langsam bei großen Pools).
- MINOR-Funde aus allen 6 Berichten noch nicht abgearbeitet.
