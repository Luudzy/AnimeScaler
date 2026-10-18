@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
echo.
if errorlevel 1 (
  echo Setup falhou. Veja a mensagem acima.
) else (
  echo Setup concluido.
)
pause
