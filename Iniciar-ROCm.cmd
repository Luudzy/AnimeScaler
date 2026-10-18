@echo off
setlocal
set ANIMESCALER_BACKEND=rocm
set MIOPEN_FIND_MODE=FAST
set MIOPEN_DISABLE_CACHE=1
set ANIMESCALER_MIOPEN_BENCHMARK=0
set ANIMESCALER_CHANNELS_LAST=0
cd /d "%~dp0"
if not defined ANIMESCALER_ROCM_PYTHON if exist "native\.venv-rocm\Scripts\python.exe" set "ANIMESCALER_ROCM_PYTHON=native\.venv-rocm\Scripts\python.exe"
if exist "native\launcher-paths.cmd" call "native\launcher-paths.cmd"
if not defined ANIMESCALER_ROCM_PYTHON (
  echo Set ANIMESCALER_ROCM_PYTHON to the Python executable for this ROCm installation.
  echo Example: set ANIMESCALER_ROCM_PYTHON=path\to\rocm\python.exe
  pause
  exit /b 1
)
if not exist "%ANIMESCALER_ROCM_PYTHON%" (
  echo Python ROCm nao encontrado em "%ANIMESCALER_ROCM_PYTHON%".
  pause
  exit /b 1
)
"%ANIMESCALER_ROCM_PYTHON%" -I native\backend_preflight.py --backend rocm
if errorlevel 1 goto :finish
"%ANIMESCALER_ROCM_PYTHON%" -u -I native\server.py

:finish
set "ANIMESCALER_EXIT_CODE=%errorlevel%"
pause
exit /b %ANIMESCALER_EXIT_CODE%
