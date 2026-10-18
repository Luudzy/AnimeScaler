@echo off
setlocal
cd /d "%~dp0"
echo AnimeScaler - NVIDIA
echo [1] TensorRT
echo [2] OpenCL (Anime4KCPP)
echo [3] FSR3 (DirectX 12)
choice /c 123 /n /m "Selecione o backend: "
if errorlevel 3 goto :fsr3
if errorlevel 2 goto :opencl
if errorlevel 1 goto :tensorrt
exit /b 1

:fsr3
call "%~dp0Iniciar-FSR3.cmd"
exit /b
:opencl
call "%~dp0Iniciar-OpenCL.cmd"
exit /b
:tensorrt
call "%~dp0Iniciar-TensorRT.cmd"
exit /b
