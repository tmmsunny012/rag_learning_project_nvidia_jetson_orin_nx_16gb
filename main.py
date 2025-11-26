"""
=============================================================================
RAG Learning Project - Main Entry Point
=============================================================================

This is your complete RAG (Retrieval-Augmented Generation) learning project!

Quick Start:
------------
1. Install dependencies:    pip install -r requirements.txt
2. Add PDFs to:             ./data/
3. Run:                     python main.py

For learning, run the lessons in order:
    python lessons/01_introduction.py
    python lessons/02_pdf_loading.py
    ...etc
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print welcome banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║           🤖 RAG Learning Project                             ║
    ║           ═══════════════════════                             ║
    ║                                                               ║
    ║   Build your own AI that answers questions from PDFs!        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print main menu."""
    print("""
    What would you like to do?

    LEARNING MODE:
    ──────────────
    [1] Run Lesson 1: Introduction to RAG
    [2] Run Lesson 2: PDF Loading
    [3] Run Lesson 3: Text Chunking
    [4] Run Lesson 4: Embeddings
    [5] Run Lesson 5: Vector Stores
    [6] Run Lesson 6: RAG Chain
    [7] Run Lesson 7: Query Interface

    APPLICATION MODE:
    ─────────────────
    [8] Build RAG Index (process your PDFs)
    [9] Interactive Q&A Session
    [0] Quick Query

    [q] Quit

    """)


def run_lesson(lesson_number: int):
    """Run a specific lesson."""
    import subprocess

    lesson_files = list(PROJECT_ROOT.glob(f"lessons/0{lesson_number}_*.py"))

    if not lesson_files:
        print(f"Lesson {lesson_number} not found!")
        return

    lesson_path = lesson_files[0]
    print(f"\n{'=' * 60}")
    print(f"Running: {lesson_path.name}")
    print('=' * 60 + "\n")

    # Run as subprocess to avoid exec() scope issues
    subprocess.run([sys.executable, str(lesson_path)], cwd=str(PROJECT_ROOT))


def check_dependencies():
    """Check if required dependencies are installed."""
    required = [
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
    ]

    missing = []

    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\n⚠️  Missing dependencies detected!")
        print("Install with:")
        print(f"    pip install {' '.join(missing)}")
        print("\nOr install all:")
        print("    pip install -r requirements.txt")
        return False

    return True


def build_index():
    """Build the RAG index from PDFs."""
    from src.rag_system import RAGSystem

    data_dir = PROJECT_ROOT / "data"
    db_dir = PROJECT_ROOT / "vector_db_v2"  # Updated to match rag_system.py default

    # Check for PDFs
    data_dir.mkdir(exist_ok=True)
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n[!] No PDF files found in {data_dir}")
        print("\nPlease add your PDF files to the 'data' folder.")
        print("Then run this option again.")
        return None

    print(f"\n[*] Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files[:5]:
        print(f"   - {pdf.name}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more")

    # Build index
    rag = RAGSystem(
        pdf_directory=str(data_dir),
        persist_directory=str(db_dir)
    )

    rebuild = input("\nRebuild index even if it exists? (y/n): ").strip().lower() == 'y'
    rag.build_index(force_rebuild=rebuild)

    return rag


def interactive_session():
    """Start an interactive Q&A session."""
    import os
    from src.rag_system import RAGSystem

    data_dir = PROJECT_ROOT / "data"
    db_dir = PROJECT_ROOT / "vector_db_v2"  # Updated to match rag_system.py default

    # Check if index exists
    if not db_dir.exists():
        print("\n[!] No index found. Building index first...")
        rag = build_index()
        if rag is None:
            return
    else:
        rag = RAGSystem(
            pdf_directory=str(data_dir),
            persist_directory=str(db_dir)
        )
        rag.build_index()

    # Choose LLM
    print("\nChoose LLM:")
    print("  [1] Ollama (free, local) - requires Ollama installed")
    print("  [2] Ollama (remote, e.g., Jetson Orin NX)")
    print("  [3] OpenAI (paid) - requires OPENAI_API_KEY")
    print("  [4] Anthropic (paid) - requires ANTHROPIC_API_KEY")

    choice = input("\nChoice (1-4) [1]: ").strip() or "1"

    llm_map = {
        "1": ("ollama", "llama3.2:3b", None),
        "2": ("ollama", "llama3.2:3b", None),  # Will prompt for host
        "3": ("openai", "gpt-4o-mini", None),
        "4": ("anthropic", "claude-3-5-sonnet-20241022", None)
    }

    llm_type, model, ollama_host = llm_map.get(choice, ("ollama", "llama3.2:3b", None))

    if choice == "1":
        model = input("Ollama model name [llama3.2:3b]: ").strip() or "llama3.2:3b"
    elif choice == "2":
        default_host = os.getenv("OLLAMA_HOST", "http://192.168.178.124:11434")
        ollama_host = input(f"Ollama server URL [{default_host}]: ").strip() or default_host
        model = input("Ollama model name [llama3.2:3b]: ").strip() or "llama3.2:3b"

    try:
        rag.setup(llm_type=llm_type, model=model, ollama_host=ollama_host)
        rag.interactive()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        if "ollama" in str(e).lower():
            print("\nMake sure Ollama is running:")
            print("  1. Install from https://ollama.ai")
            print("  2. Run: ollama pull llama3.2:3b")
            if choice == "2":
                print(f"  3. For remote: ensure server at {ollama_host} is accessible")


def quick_query():
    """Answer a single question."""
    import os
    from src.rag_system import RAGSystem

    data_dir = PROJECT_ROOT / "data"
    db_dir = PROJECT_ROOT / "vector_db_v2"  # Updated to match rag_system.py default

    if not db_dir.exists():
        print("\n[!] No index found. Please build index first (option 8).")
        return

    question = input("\nYour question: ").strip()
    if not question:
        return

    rag = RAGSystem(
        pdf_directory=str(data_dir),
        persist_directory=str(db_dir)
    )

    rag.build_index()

    # Check for remote Ollama in environment
    ollama_host = os.getenv("OLLAMA_HOST")

    try:
        rag.setup(llm_type="ollama", model="llama3.2:3b", ollama_host=ollama_host)
        result = rag.query_with_sources(question)

        print(f"\n[Answer]\n{result['answer']}")
        print("\n[Sources]")
        for i, src in enumerate(result['sources'], 1):
            print(f"   {i}. {Path(src['source']).name} (page {src['page']})")

    except Exception as e:
        print(f"\n[ERROR] {e}")


def main():
    """Main entry point."""
    print_banner()

    if not check_dependencies():
        print("\nPlease install dependencies and try again.")
        return

    while True:
        print_menu()
        choice = input("Your choice: ").strip().lower()

        if choice == 'q':
            print("\n👋 Goodbye! Happy learning!")
            break

        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            run_lesson(int(choice))
            input("\nPress Enter to continue...")

        elif choice == '8':
            build_index()
            input("\nPress Enter to continue...")

        elif choice == '9':
            interactive_session()

        elif choice == '0':
            quick_query()
            input("\nPress Enter to continue...")

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
