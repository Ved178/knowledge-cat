@echo off
REM Generate PCA and t-SNE embedding plots from ChromaDB.
cd /d "%~dp0"

if not exist "env\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

env\Scripts\python plot_embeddings.py --chroma-path ./chroma_db
echo.
echo Plots saved to embedding_plots/
pause
