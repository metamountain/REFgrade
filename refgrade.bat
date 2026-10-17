@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM RefGrade bootstrap (K38f): pick a Python, install one with uv if missing, start the GUI.
REM Python:  config.json["python_exe"] -> <runtime>\venv\Scripts\python.exe -> uv bootstrap
REM Runtime: REFGRADE_RUNTIME -> config.json["runtime_dir"] -> %LOCALAPPDATA%\RefGrade\runtime

set "CFG=%APPDATA%\RefGrade\config.json"

REM --- 1. runtime dir
set "RUNTIME_DIR="
if defined REFGRADE_RUNTIME goto :have_runtime
for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "try { (Get-Content -Raw '%CFG%' | ConvertFrom-Json).runtime_dir } catch { '' }"`) do set "RUNTIME_DIR=%%L"
if defined RUNTIME_DIR goto :have_runtime
set "RUNTIME_DIR=%LOCALAPPDATA%\RefGrade\runtime"
:have_runtime
echo [refgrade] runtime dir: %RUNTIME_DIR%

REM --- 2. python: config.json["python_exe"], then the runtime venv
set "PYTHON="
for /f "usebackq delims=" %%L in (`powershell -NoProfile -Command "try { (Get-Content -Raw '%CFG%' | ConvertFrom-Json).python_exe } catch { '' }"`) do set "PYTHON=%%L"
if defined PYTHON (
    if exist "!PYTHON!" goto :have_python
    echo [refgrade] python_exe from config.json not found, ignoring: !PYTHON!
    set "PYTHON="
)
if exist "%RUNTIME_DIR%\venv\Scripts\python.exe" (
    set "PYTHON=%RUNTIME_DIR%\venv\Scripts\python.exe"
    goto :have_python
)

REM --- 3. no python: bootstrap uv + Python 3.12 venv + base packages
echo [refgrade] no python found, bootstrapping uv + Python 3.12 into %RUNTIME_DIR%
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
if exist "%RUNTIME_DIR%\uv\uv.exe" goto :have_uv
echo [refgrade] downloading uv 0.13.0 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-pc-windows-msvc.zip' -OutFile '%RUNTIME_DIR%\uv.zip' -UseBasicParsing; Expand-Archive -LiteralPath '%RUNTIME_DIR%\uv.zip' -DestinationPath '%RUNTIME_DIR%\uv' -Force; Remove-Item -LiteralPath '%RUNTIME_DIR%\uv.zip' -Force"
if errorlevel 1 goto :err_uv_dl
if not exist "%RUNTIME_DIR%\uv\uv.exe" goto :err_uv_dl
:have_uv
set "UV=%RUNTIME_DIR%\uv\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%RUNTIME_DIR%\python"
set "UV_CACHE_DIR=%RUNTIME_DIR%\cache"
set "UV_LINK_MODE=copy"
echo [refgrade] creating python 3.12 venv ...
"%UV%" venv --python 3.12 --managed-python "%RUNTIME_DIR%\venv"
if errorlevel 1 goto :err_venv
set "PYTHON=%RUNTIME_DIR%\venv\Scripts\python.exe"
echo [refgrade] installing PyQt6, numpy, pillow ...
"%UV%" pip install --python "%PYTHON%" PyQt6 numpy pillow
if errorlevel 1 goto :err_pip

:have_python
echo [refgrade] starting gui with %PYTHON%
"%PYTHON%" "%~dp0standalone\gui.py" %*
exit /b %errorlevel%

:err_uv_dl
echo [refgrade] ERROR: downloading uv failed.
goto :fail

:err_venv
echo [refgrade] ERROR: creating the python venv failed.
goto :fail

:err_pip
echo [refgrade] ERROR: installing packages failed.
goto :fail

:fail
pause
exit /b 1
