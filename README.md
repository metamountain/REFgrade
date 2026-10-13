# REFgrade — Colour Grading by Reference (Krita plugin · standalone in progress)

> **Hobby project — not bug-free.** Tested only on Windows 11 with Krita 6.0.4 (beta) at
> 2560×1440. Expect rough edges, crashes, and things that simply don't work yet.
> If you hit a problem, please file an issue on GitHub — bug reports, crash logs and
> "this button does nothing" are all equally welcome.


Transfers the light and color mood of a reference image onto your image in Krita.
Adjust the result with grading sliders, apply it as a new layer, or export as a 3D LUT (.cube).

<a href="docs/screenshot.jpg"><img src="docs/screenshot.jpg" width="100%" alt="REFgrade dialog: grading controls, colour wheels and scopes beside the graded image"></a>

## Prerequisites

- Krita 6.0.4 (Beta, Qt6), Windows 11
- An NVIDIA GPU is recommended for reasonable render times. Without one, RefGrade
  falls back to CPU (much slower, especially on large images).

That's it. On first use, RefGrade downloads and sets up its own Python runtime
(PyTorch + dependencies) into `%LOCALAPPDATA%\RefGrade\runtime`. No manual Python
installation is needed.

## Installation

1. Download `ref_grade-x.y.z.zip` from the GitHub Releases page. Do not unzip it.
2. In Krita: **Tools → Scripts → Import Python Plugin from File…** and choose the ZIP.
3. Confirm the activation question, then restart Krita.
4. Open RefGrade from the **Filter → RefGrade** menu with a paint layer active.
5. **First time only:** a dialog appears asking to install the runtime. Choose
   **GPU** (recommended, several GB download) or **CPU** (smaller, slower) and click
   **Install**. This downloads `uv`, creates an isolated Python 3.12 environment with
   PyTorch, and stores it in `%LOCALAPPDATA%\RefGrade\runtime`. Your system Python is
   not modified. The dialog shows a live log; you can cancel at any time.
6. Once the install finishes, the RefGrade dialog opens normally.

### Using your own Python (optional)

If you already have a Python 3.12 with `torch`, `numpy`, `kornia` and `pillow`
installed, you can skip the built-in runtime. Set the environment variable
`REFGRADE_PYTHON` to the full path of your interpreter (e.g. `C:\Python312\python.exe`)
as a **user** environment variable, then restart Krita. RefGrade will verify the
packages on startup and show an error if something is missing.

For GPU use with your own Python, install the CUDA build of torch that matches your
driver ([pytorch.org](https://pytorch.org/get-started/locally/)):

```
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cu130
py -3.12 -m pip install numpy kornia pillow
```

**Update:** restart Krita and, *before* opening RefGrade, import the new ZIP the same way, confirm the
overwrite, restart Krita. (While RefGrade has been opened, its worker keeps files in the plugin folder busy
until Krita closes, and the overwrite can fail.)

**Uninstall:** **Settings → Configure Krita → Python Plugin Manager**, untick RefGrade, restart.
To remove it completely, also delete `pykrita\ref_grade`, `pykrita\ref_grade.desktop` and
`actions\ref_grade.action` from Krita's resource folder (**Settings → Manage Resources → Open Resource Folder**;
usually `%APPDATA%\krita`). To free the disk space used by the runtime, delete
`%LOCALAPPDATA%\RefGrade\runtime` (several GB with the GPU build).

## Usage (6 lines)

1. Open the RefGrade dialog from the Filter menu.
2. Load one or more reference images into the slots (drag & drop or click).
3. Choose a method (Reinhard / MKL / Zonal) and press FIT to auto-match.
4. Fine-tune with the sliders and color wheels; use SOLO to isolate a section.
5. Press OK to apply as a new layer, or LUT… to export a .cube file.
6. CANCEL closes without applying.

## LUT output

Exported .cube files are saved to the folder you choose in the file dialog
(default: `%APPDATA%\krita\luts`).

## Troubleshooting

- **Plugin missing from the menu:** enable it in **Settings → Configure Krita → Python Plugin Manager** and restart Krita.
- **Install dialog appears every time:** the runtime check failed. Open a terminal and run
  `%LOCALAPPDATA%\RefGrade\runtime\venv\Scripts\python.exe -c "import torch, numpy, kornia, PIL"`.
  If that fails, delete `%LOCALAPPDATA%\RefGrade\runtime` and let RefGrade reinstall.
- **Install fails (network error):** the download of `uv` or PyTorch was interrupted.
  Close the dialog, check your connection, and retry. Partial downloads are cleaned up automatically.
- **GPU not detected:** make sure your NVIDIA driver is up to date. RefGrade detects the
  CUDA version via `nvidia-smi`; if it reports CUDA < 12.6 the CPU build is used instead.
  The install dialog shows which flavor was chosen and why.
- **Worker does not start** (dialog shows an error): check the worker log at
  `%TEMP%\ref_grade\worker.log` for the full traceback.
- **Dialog shows "ERROR · …":** check the same log for details.

## For developers

- `python install.py` links the repository into Krita (a directory junction to `plugin/ref_grade`, plus the
  `.desktop` and `.action` files); `python install.py --uninstall` removes it; `--dry-run` only prints the targets.
  In this layout the plugin uses `core/` from the repository.
- `python tools/build_zip.py` builds `dist/ref_grade-<version>.zip` (version from `core/refgrade/version.py`),
  with the worker bundled as `ref_grade/core`.
- **Never import the ZIP while the dev junction exists** — the importer would overwrite the repository's
  plugin files through the junction. Run `python install.py --uninstall` first.
