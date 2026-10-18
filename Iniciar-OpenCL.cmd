@echo off
setlocal
set ANIMESCALER_BACKEND=opencl
cd /d "%~dp0"
if not defined ANIMESCALER_OPENCL_PYTHON (
  echo Verificando ambiente OpenCL isolado. A primeira execucao pode baixar dependencias.
  node scripts\setup-opencl.mjs
  if errorlevel 1 goto :finish
  set "ANIMESCALER_OPENCL_PYTHON=native\.venv-opencl\Scripts\python.exe"
)
if not exist "%ANIMESCALER_OPENCL_PYTHON%" (
  echo Python OpenCL nao encontrado em "%ANIMESCALER_OPENCL_PYTHON%".
  pause
  exit /b 1
)
"%ANIMESCALER_OPENCL_PYTHON%" -I native\backend_preflight.py --backend opencl
if errorlevel 1 goto :finish
"%ANIMESCALER_OPENCL_PYTHON%" -u -I native\server.py

:finish
set "ANIMESCALER_EXIT_CODE=%errorlevel%"
pause
exit /b %ANIMESCALER_EXIT_CODE%
