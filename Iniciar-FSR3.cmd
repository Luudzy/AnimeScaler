@echo off
setlocal
cd /d "%~dp0"
if not defined ANIMESCALER_FSR3_PYTHON if exist "native\.venv-fsr3\Scripts\python.exe" set "ANIMESCALER_FSR3_PYTHON=native\.venv-fsr3\Scripts\python.exe"
if exist "native\launcher-paths.cmd" call "native\launcher-paths.cmd"
if not defined ANIMESCALER_FSR3_PYTHON (
  echo Set ANIMESCALER_FSR3_PYTHON to a Python 3.10 environment with aiohttp and numpy.
  pause
  exit /b 1
)
if not exist "%ANIMESCALER_FSR3_PYTHON%" (
  echo Python FSR3 nao encontrado em "%ANIMESCALER_FSR3_PYTHON%".
  pause
  exit /b 1
)
if not defined ANIMESCALER_FSR_SDK (
  echo Set ANIMESCALER_FSR_SDK to a local FidelityFX SDK 2.3.0 directory.
  pause
  exit /b 1
)
if not exist "%ANIMESCALER_FSR_SDK%" (
  echo FidelityFX SDK not found at "%ANIMESCALER_FSR_SDK%".
  pause
  exit /b 1
)
echo Verificando ambiente FSR3 isolado. A primeira execucao pode baixar dependencias.
node scripts\setup-fsr3.mjs
if errorlevel 1 goto :finish
set "ANIMESCALER_FSR3_PYTHON=native\.venv-fsr3\Scripts\python.exe"
"%ANIMESCALER_FSR3_PYTHON%" -I native\backend_preflight.py --backend fsr3 --sdk "%ANIMESCALER_FSR_SDK%"
if errorlevel 1 goto :finish
if not exist "native\fsr3-probe\build\fsr3-worker.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "native\fsr3-probe\build.ps1" -Sdk "%ANIMESCALER_FSR_SDK%"
  if errorlevel 1 goto :finish
)
"%ANIMESCALER_FSR3_PYTHON%" -u -I native\fsr3_server.py

:finish
set "ANIMESCALER_EXIT_CODE=%errorlevel%"
pause
exit /b %ANIMESCALER_EXIT_CODE%
