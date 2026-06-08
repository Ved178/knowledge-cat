@echo off
REM Launch the interactive semantic search REPL.
cd /d "%~dp0"

if not exist "env\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

env\Scripts\python query.py --embedding-model models/e5-large-v2
pause
