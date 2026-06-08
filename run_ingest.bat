@echo off
REM Ingest documents from .\data into ChromaDB.
REM Edit PATHS below to point at your folder(s).
cd /d "%~dp0"

set PATHS=./data
REM set PATHS=./data "C:\path\to\another\folder"

if not exist "env\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

env\Scripts\python ingest.py --paths %PATHS% --embedding-model models/e5-large-v2
pause
