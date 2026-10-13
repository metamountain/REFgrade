# REFgrade — Colour Grading by Reference (Krita plugin · standalone in progress)

> **Prototype.** Tested only on Windows 11 with Krita 6.0.4 (beta) at 2560×1440. Expect rough edges —
> comments, bug reports and ideas are very welcome via GitHub Issues. A detailed README follows.


Transfers the light and color mood of a reference image onto your image in Krita.
Adjust the result with grading sliders, apply it as a new layer, or export as a 3D LUT (.cube).

## Prerequisites

- Krita 6.0.4 (Beta, Qt6)
- System Python 3.12 with torch (CUDA), torchvision, numpy

## Installation

1. Run `python install.py` from this directory.
2. In Krita: **Settings → Configure Krita → Python Plugin Manager → RefGrade** — enable it.
3. Restart Krita.

To remove: `python install.py --uninstall`

## Usage (6 lines)

1. Open the RefGrade dialog from the Scripts menu.
2. Load one or more reference images into the slots (drag & drop or click).
3. Choose a method (Reinhard / MKL / Zonal) and press FIT to auto-match.
4. Fine-tune with the sliders and color wheels; use SOLO to isolate a section.
5. Press OK to apply as a new layer, or LUT… to export a .cube file.
6. CANCEL closes without applying.

## LUT output

Exported .cube files are saved to the folder you choose in the file dialog
(default: `%APPDATA%\krita\luts`).

## Troubleshooting

- Worker log: `%TEMP%\ref_grade\worker.log`
- If the dialog shows "ERROR · …", check the log for the full traceback.
- GPU out-of-memory: the worker falls back to CPU automatically.
