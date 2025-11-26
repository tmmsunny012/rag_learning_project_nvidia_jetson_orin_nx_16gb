# RAG Learning Project - NVIDIA Jetson Orin NX 16GB

A complete **Retrieval-Augmented Generation (RAG)** learning project for building PDF Q&A systems. Optimized for **NVIDIA Jetson Orin NX** with remote Ollama support for GPU-accelerated inference.

## What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:
- **Your documents** (PDFs, text files, etc.)
- **Vector search** (finding relevant content)
- **LLM** (generating answers based on found content)

This allows you to build an AI that answers questions specifically about YOUR data!

## Features

- **7 Step-by-step Learning Lessons** - From basics to complete RAG pipeline
- **PyMuPDF Integration** - Superior PDF text extraction quality
- **ChromaDB Vector Store** - Persistent vector storage with similarity search
- **Remote Ollama Support** - Offload LLM inference to Jetson Orin NX
- **Multiple LLM Options** - Ollama (free), OpenAI, Anthropic
- **Two Interfaces** - Menu-based GUI and CLI for flexibility

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/tmmsunny012/rag_learning_project_nvidia_jetson_orin_nx_16gb.git
cd rag_learning_project_nvidia_jetson_orin_nx_16gb

# Windows
setup.bat

# Mac/Linux
chmod +x setup.sh && ./setup.sh
```

### 2. Setup Ollama

**Option A: Local Ollama**
```bash
# Install from https://ollama.ai
ollama pull llama3.2:3b
```

**Option B: Remote Ollama on Jetson Orin NX (Recommended)**
```bash
# On your Jetson:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 3. Add Your PDFs

Place your PDF files in the `data/` folder.

### 4. Run!

**Menu-based Interface (Recommended for Learning)**
```bash
python main.py
```

**Command Line Interface**
```bash
# Local Ollama
python src/rag_system.py --pdf-dir ./data --rebuild --interactive

# Remote Ollama (Jetson Orin NX)
python src/rag_system.py --pdf-dir ./data --rebuild --interactive --ollama-host http://192.168.178.124:11434
```

## Project Structure

```
rag_learning_project/
├── main.py                 # Menu-based interface (lessons + app)
├── src/
│   └── rag_system.py       # Production CLI RAG application
├── lessons/                # Step-by-step learning modules
│   ├── 01_introduction.py  # What is RAG, pipeline overview
│   ├── 02_pdf_loading.py   # PDF loading with PyMuPDF
│   ├── 03_text_chunking.py # Chunking strategies
│   ├── 04_embeddings.py    # Vector embeddings
│   ├── 05_vector_store.py  # ChromaDB, FAISS
│   ├── 06_rag_chain.py     # Complete RAG chain
│   └── 07_query_interface.py # Interactive CLI
├── data/                   # Your PDF files here
├── vector_db_v2/           # Vector database (auto-created)
├── requirements.txt        # Dependencies
└── GETTING_STARTED.md      # Detailed guide
```

## Learning Path

| Lesson | Topics |
|--------|--------|
| 01 | Introduction to RAG, pipeline overview |
| 02 | PDF loading with PyMuPDF (recommended) |
| 03 | Text chunking strategies, chunk size optimization |
| 04 | Vector embeddings, similarity concepts |
| 05 | Vector stores (ChromaDB, FAISS) |
| 06 | RAG chain with LangChain, LLM options |
| 07 | Interactive query interface |

## Jetson Orin NX Benefits

Running Ollama on NVIDIA Jetson Orin NX provides:
- **GPU-accelerated inference** using CUDA
- **Dedicated AI processing** - frees your PC
- **~3x faster** than CPU-only inference
- **24/7 AI server** capability

## LLM Options

| Provider | Model | Cost |
|----------|-------|------|
| Ollama | llama3.2:3b, mistral, phi3 | Free (local) |
| OpenAI | gpt-4o-mini, gpt-4o | Paid |
| Anthropic | claude-3-5-sonnet | Paid |

## Key Technologies

- **LangChain** - RAG pipeline orchestration
- **PyMuPDF (fitz)** - PDF text extraction
- **ChromaDB** - Vector database
- **HuggingFace Embeddings** - all-MiniLM-L6-v2
- **Ollama** - Local/remote LLM inference

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--pdf-dir` | PDF files directory | Required |
| `--db-dir` | Vector database directory | `./vector_db_v2` |
| `--query` | Single question | - |
| `--interactive` | Interactive Q&A mode | - |
| `--rebuild` | Force rebuild index | - |
| `--llm` | LLM provider | `ollama` |
| `--model` | Model name | `llama3.2:3b` |
| `--ollama-host` | Remote Ollama URL | `localhost:11434` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'langchain'" | `pip install -r requirements.txt` |
| "No module named 'fitz'" | `pip install pymupdf` |
| "Ollama not found" | Install Ollama, run `ollama pull llama3.2:3b` |
| "Connection refused" (remote) | Check Jetson IP, ensure Ollama is running with `OLLAMA_HOST=0.0.0.0:11434` |

## License

MIT License - Feel free to use for learning and projects!

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

Built with LangChain, ChromaDB, and Ollama
