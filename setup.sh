#!/bin/bash

echo "============================================"
echo "   RAG Learning Project - Setup"
echo "============================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.10+ first"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv

echo
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo
echo "[3/4] Upgrading pip..."
pip install --upgrade pip

echo
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt

echo
echo "============================================"
echo "   Setup Complete!"
echo "============================================"
echo
echo "To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo
echo "Then run the project:"
echo "   python main.py"
echo
echo "Don't forget to:"
echo "   1. Install Ollama from https://ollama.ai"
echo "   2. Run: ollama pull llama3.2"
echo "   3. Add your PDFs to the data/ folder"
