# RAG Learning Project - Getting Started Guide

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that combines:
- **Your documents** (PDFs, text files, etc.)
- **Vector search** (finding relevant content)
- **LLM** (generating answers based on found content)

This allows you to build an AI that answers questions specifically about YOUR data!

## Quick Start (5 minutes)

### Step 1: Setup Virtual Environment & Install Dependencies

**Windows:**
```bash
cd rag_learning_project

# Option A: Run the setup script (recommended)
setup.bat

# Option B: Manual setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
cd rag_learning_project

# Option A: Run the setup script (recommended)
chmod +x setup.sh
./setup.sh

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Important:** Always activate the venv before running the project:
> - Windows: `venv\Scripts\activate`
> - Mac/Linux: `source venv/bin/activate`

### Step 2: Setup Ollama LLM Server

You have two options for running Ollama:

#### Option A: Local Ollama (on your PC)

1. Download from: https://ollama.ai
2. Install and run
3. Pull a model:
   ```bash
   ollama pull llama3.2:3b
   ```

#### Option B: Remote Ollama on Jetson Orin NX (Recommended for GPU Acceleration)

If you have a **NVIDIA Jetson Orin NX** or similar edge device, you can offload LLM inference to it:

**On the Jetson Orin NX:**
```bash
# Install Ollama on Jetson (ARM64)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model optimized for Jetson
ollama pull llama3.2:3b

# Start Ollama server (binds to all interfaces)
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**Configure your environment:**
```bash
# Create .env file from template
cp .env.example .env

# Edit .env and set your Jetson's IP address
# OLLAMA_HOST=http://192.168.178.124:11434
```

**Verify connection:**
```bash
# Test from your PC (replace with your Jetson's IP)
curl http://192.168.178.124:11434/api/tags
```

> **Jetson Orin NX Benefits:**
> - GPU-accelerated inference using CUDA
> - Dedicated AI processing, freeing your PC
> - ~3x faster than CPU-only inference
> - Runs 24/7 as an AI server

### Step 3: Add Your PDFs

Put your PDF files in the `data/` folder:
```
rag_learning_project/
├── data/
│   ├── your_document_1.pdf
│   ├── your_document_2.pdf
│   └── ...
```

### Step 4: Run!

**Option A: Menu-based Interface (Recommended for Learning)**
```bash
# Make sure venv is activated first!
python main.py
```
Choose option `8` to build the index, then `9` for interactive Q&A!

**Option B: Command Line Interface**
```bash
# Run with local Ollama
python src/rag_system.py --pdf-dir ./data --rebuild --interactive

# Or with remote Ollama (e.g., Jetson Orin NX)
python src/rag_system.py --pdf-dir ./data --rebuild --interactive --ollama-host http://192.168.178.124:11434
```

The `--rebuild` flag indexes your PDFs. Once indexed, you can omit it for faster startup.

---

## Learning Path

For a complete understanding, go through the lessons in order:

| Lesson | File | Topics |
|--------|------|--------|
| 1 | `01_introduction.py` | What is RAG, pipeline overview |
| 2 | `02_pdf_loading.py` | Loading PDFs with PyPDF, LangChain |
| 3 | `03_text_chunking.py` | Chunking strategies, chunk size |
| 4 | `04_embeddings.py` | Vector embeddings, similarity |
| 5 | `05_vector_store.py` | ChromaDB, FAISS, retrieval |
| 6 | `06_rag_chain.py` | Complete RAG chain, LLM options |
| 7 | `07_query_interface.py` | Interactive CLI, sources |

---

## Project Structure

```
rag_learning_project/
├── main.py                 # Menu-based interface (run lessons + app)
├── requirements.txt        # Dependencies
├── setup.bat / setup.sh    # Setup scripts (creates venv)
├── .env.example           # Environment variables template
├── GETTING_STARTED.md     # This guide
├── PROJECT_REPORT.md      # Development report
│
├── src/
│   └── rag_system.py      # CLI RAG application (production)
│
├── lessons/               # Step-by-step learning modules
│   ├── 01_introduction.py
│   ├── 02_pdf_loading.py
│   ├── 03_text_chunking.py
│   ├── 04_embeddings.py
│   ├── 05_vector_store.py
│   ├── 06_rag_chain.py
│   └── 07_query_interface.py
│
├── data/                  # Put your PDFs here
├── vector_db_v2/          # Vector database (auto-created)
└── venv/                  # Virtual environment (created by setup)
```

---

## LLM Options

### Free (Local)
**Ollama** - Install: https://ollama.ai
```bash
ollama pull llama3.2
```

### Paid (Cloud)
- **OpenAI**: Set `OPENAI_API_KEY` env variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` env variable

---

## Command Line Usage

```bash
# Activate venv first!
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Interactive mode (local Ollama)
python src/rag_system.py --pdf-dir ./data --interactive

# Interactive mode (remote Jetson Ollama)
python src/rag_system.py --pdf-dir ./data --interactive --ollama-host http://192.168.178.124:11434

# Single query
python src/rag_system.py --pdf-dir ./data --query "Your question"

# Rebuild index (after adding/removing PDFs)
python src/rag_system.py --pdf-dir ./data --rebuild --interactive

# Use OpenAI instead
python src/rag_system.py --pdf-dir ./data --llm openai --interactive
```

### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pdf-dir` | Directory containing PDF files | Required |
| `--db-dir` | Vector database directory | `./vector_db_v2` |
| `--query` | Single question to ask | - |
| `--interactive` | Start interactive Q&A mode | - |
| `--rebuild` | Force rebuild the index | - |
| `--llm` | LLM provider (ollama/openai/anthropic) | `ollama` |
| `--model` | Model name | `llama3.2:3b` |
| `--ollama-host` | Remote Ollama server URL | `http://localhost:11434` |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "No module named 'langchain'" | Activate venv, then `pip install -r requirements.txt` |
| "No module named 'fitz'" | Install PyMuPDF: `pip install pymupdf` |
| "Ollama not found" | Install Ollama, run `ollama pull llama3.2:3b` |
| "No PDF files found" | Add PDFs to `data/` folder |
| "Connection refused" (remote Ollama) | Check Jetson IP, ensure `OLLAMA_HOST=0.0.0.0:11434 ollama serve` is running |
| Corrupted/old index data | Delete `vector_db_v2/` folder and rebuild with `--rebuild` |
| Poor PDF text extraction | Using PyMuPDF (fitz) for best quality extraction |

---

## Troubleshooting Remote Ollama

If you're having issues connecting to remote Ollama on Jetson:

1. **Check Ollama is running on Jetson:**
   ```bash
   ssh jetson@192.168.178.124
   systemctl status ollama  # or check if ollama serve is running
   ```

2. **Verify network connectivity:**
   ```bash
   ping 192.168.178.124
   curl http://192.168.178.124:11434/api/tags
   ```

3. **Check firewall settings on Jetson:**
   ```bash
   sudo ufw allow 11434/tcp
   ```

4. **Ensure Ollama binds to all interfaces:**
   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

Happy learning!
