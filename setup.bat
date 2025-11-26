@echo off
echo ============================================
echo    RAG Learning Project - Setup
echo ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/4] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo To activate the virtual environment, run:
echo    venv\Scripts\activate
echo.
echo Then run the project:
echo    python main.py
echo.
echo Don't forget to:
echo    1. Install Ollama from https://ollama.ai
echo    2. Run: ollama pull llama3.2
echo    3. Add your PDFs to the data/ folder
echo.
pause
