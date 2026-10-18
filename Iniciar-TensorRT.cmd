@echo off
setlocal
cd /d "%~dp0"
if not exist "native\.venv-nvidia\Scripts\python.exe" (
  echo TensorRT exige GPU NVIDIA e native\.venv-nvidia com PyTorch CUDA, TensorRT 10.x, spandrel, numpy e aiohttp.
  echo Esta versao foi validada em AMD RX 5500 XT; confirme GPU NVIDIA e configure o ambiente CUDA.
  echo O ambiente ROCm nao sera alterado.
  pause
  exit /b 1
)
set ANIMESCALER_BACKEND=tensorrt
"native\.venv-nvidia\Scripts\python.exe" -I native\backend_preflight.py --backend tensorrt
if errorlevel 1 goto :finish
"native\.venv-nvidia\Scripts\python.exe" -u -I native\prepare_tensorrt.py
if errorlevel 1 goto :finish
"native\.venv-nvidia\Scripts\python.exe" -u -I native\server.py

:finish
set "ANIMESCALER_EXIT_CODE=%errorlevel%"
pause
exit /b %ANIMESCALER_EXIT_CODE%
