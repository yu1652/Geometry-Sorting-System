@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing local environment. See README.md for setup.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
