@echo off
REM Windows launcher — double-click this file to run setup.
REM Requires Python 3.10+ to be installed: https://python.org/downloads

cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python not found. Download it from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

python setup.py
pause
