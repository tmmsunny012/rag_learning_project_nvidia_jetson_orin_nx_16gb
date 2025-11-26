"""
=============================================================================
LESSON 7: Interactive Query Interface
=============================================================================

In this lesson, you'll learn:
- Building an interactive CLI for your RAG system
- Adding conversation history
- Displaying sources
- Error handling and edge cases

This lesson brings everything together into a usable application!
"""

import os
from pathlib import Path
from typing import List, Optional


import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class RAGApplication:
    """
    Complete RAG application with interactive query interface.
    """

    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_type: str = "ollama",  # "ollama", "openai", or "anthropic"
        llm_model: str = "llama3.2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ollama_host: str = None,  # For remote Ollama (e.g., Jetson Orin NX)
    ):
        """
        Initialize the RAG application.

        Args:
            pdf_directory: Path to directory containing PDFs
            persist_directory: Where to store the vector database
            embedding_model: Name of the embedding model
            llm_type: Type of LLM to use ("ollama", "openai", or "anthropic")
            llm_model: Specific model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            ollama_host: Remote Ollama server URL (e.g., "http://192.168.178.124:11434")
                         If None, uses localhost or OLLAMA_HOST environment variable
        """
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_host = ollama_host

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.conversation_history = []

    def load_documents(self) -> List:
        """Load all PDF documents from the directory using PyMuPDF for better extraction."""
        import fitz  # PyMuPDF - better text extraction than PyPDF
        from langchain_core.documents import Document

        print(f"\n[*] Loading PDFs from: {self.pdf_directory}")

        documents = []
        pdf_dir = Path(self.pdf_directory)
        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text.strip():  # Only add non-empty pages
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "page": page_num,
                                "total_pages": len(doc)
                            }
                        ))
                doc.close()
                print(f"  - {pdf_path.name}: {len(doc)} pages")
            except Exception as e:
                print(f"  - Error loading {pdf_path}: {e}")

        print(f"[OK] Loaded {len(documents)} pages from {len(pdf_files)} PDFs")

        return documents

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print(f"\n✂️ Splitting documents (chunk_size={self.chunk_size})...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        chunks = splitter.split_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")

        return chunks

    def create_vectorstore(self, chunks: List):
        """Create or load the vector store."""
        from langchain_community.vectorstores import Chroma
        # Modern import (recommended)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        print(f"\n[*] Creating embeddings with {self.embedding_model}...")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Check if vectorstore exists
        if Path(self.persist_directory).exists():
            print(f"📂 Loading existing vectorstore from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
        else:
            print(f"🆕 Creating new vectorstore at {self.persist_directory}")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={"k": 4, "fetch_k": 10}
        )

        print("✓ Vectorstore ready")

    def create_llm(self):
        """Create the LLM based on configuration."""
        print(f"\n[*] Initializing LLM: {self.llm_type} ({self.llm_model})")

        if self.llm_type == "ollama":
            from langchain_ollama import ChatOllama

            # Support for remote Ollama (e.g., Jetson Orin NX)
            ollama_url = self.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

            llm = ChatOllama(
                model=self.llm_model,
                temperature=0,
                base_url=ollama_url
            )
            print(f"    Ollama server: {ollama_url}")

        elif self.llm_type == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self.llm_model, temperature=0)

        elif self.llm_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=self.llm_model, temperature=0)

        else:
            raise ValueError(f"Unknown LLM type: {self.llm_type}")

        print("[OK] LLM ready")
        return llm

    def create_rag_chain(self, llm):
        """Create the RAG chain."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based on the provided context.
If the answer is not in the context, say "I don't have information about that in my documents."
Always be concise and accurate.

Context:
{context}

Question: {question}

Answer:""")

        def format_docs(docs):
            return "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source', 'Unknown')}, "
                f"Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            )

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("✓ RAG chain ready")

    def initialize(self):
        """Initialize the entire RAG system."""
        print("\n" + "=" * 60)
        print("🚀 INITIALIZING RAG SYSTEM")
        print("=" * 60)

        # Load and process documents
        documents = self.load_documents()

        if not documents:
            print("\n⚠️ No PDF documents found!")
            print(f"Please add PDF files to: {self.pdf_directory}")
            return False

        chunks = self.split_documents(documents)

        # Create vector store
        self.create_vectorstore(chunks)

        # Create LLM and chain
        llm = self.create_llm()
        self.create_rag_chain(llm)

        print("\n" + "=" * 60)
        print("✅ RAG SYSTEM READY")
        print("=" * 60)

        return True

    def query(self, question: str, show_sources: bool = True) -> str:
        """
        Query the RAG system.

        Args:
            question: The user's question
            show_sources: Whether to show source documents

        Returns:
            The answer from the RAG system
        """
        if not self.rag_chain:
            return "Error: RAG system not initialized. Call initialize() first."

        # Get answer
        answer = self.rag_chain.invoke(question)

        # Get source documents for reference
        if show_sources:
            source_docs = self.retriever.invoke(question)

            print("\n📚 Sources used:")
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"  {i}. {Path(source).name} (Page {page})")

        # Add to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })

        return answer

    def interactive_mode(self):
        """Run an interactive query session."""
        print("\n" + "=" * 60)
        print("💬 INTERACTIVE RAG SESSION")
        print("=" * 60)
        print("Commands:")
        print("  'quit' or 'exit' - End session")
        print("  'clear' - Clear conversation history")
        print("  'history' - Show conversation history")
        print("  'sources on/off' - Toggle source display")
        print("=" * 60)

        show_sources = True

        while True:
            try:
                question = input("\n❓ Your question: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if question.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation history cleared")
                    continue

                if question.lower() == 'history':
                    if not self.conversation_history:
                        print("No conversation history yet.")
                    else:
                        print("\n📜 Conversation History:")
                        for i, item in enumerate(self.conversation_history, 1):
                            print(f"\n{i}. Q: {item['question']}")
                            print(f"   A: {item['answer'][:100]}...")
                    continue

                if question.lower() == 'sources on':
                    show_sources = True
                    print("✓ Sources will be shown")
                    continue

                if question.lower() == 'sources off':
                    show_sources = False
                    print("✓ Sources hidden")
                    continue

                # Process the question
                print("\n🔍 Searching documents...")
                answer = self.query(question, show_sources=show_sources)

                print("\n📝 Answer:")
                print("-" * 40)
                print(answer)
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


# =============================================================================
# SIMPLE USAGE EXAMPLE
# =============================================================================

def simple_usage_example():
    """Show simple usage of the RAG application."""
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │              SIMPLE USAGE EXAMPLE                            │
    └─────────────────────────────────────────────────────────────┘

    from lessons.07_query_interface import RAGApplication

    # Option A: Local Ollama
    app = RAGApplication(
        pdf_directory="./data",           # Your PDFs here
        persist_directory="./chroma_db",  # Vector DB storage
        llm_type="ollama",                # or "openai", "anthropic"
        llm_model="llama3.2"              # Model name
    )

    # Option B: Remote Ollama (e.g., NVIDIA Jetson Orin NX)
    app = RAGApplication(
        pdf_directory="./data",
        persist_directory="./chroma_db",
        llm_type="ollama",
        llm_model="llama3.2",
        ollama_host="http://192.168.178.124:11434"  # Remote server
    )

    # Build the RAG system
    app.initialize()

    # Query
    answer = app.query("What is machine learning?")
    print(answer)

    # Or start interactive mode
    app.interactive_mode()
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LESSON 7: Interactive Query Interface")
    print("=" * 60)

    simple_usage_example()

    # Check if we should run a demo
    print("\n" + "-" * 40)
    print("To run a live demo, ensure you have:")
    print("1. PDFs in the ./data directory")
    print("2. Ollama installed with a model (or OpenAI API key)")
    print("3. Required packages installed")
    print("-" * 40)

    response = input("\nRun interactive demo? (y/n): ").strip().lower()

    if response == 'y':
        # Set up paths
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        db_dir = project_root / "chroma_db"

        # Create data directory if it doesn't exist
        data_dir.mkdir(exist_ok=True)

        # Check for PDFs
        pdf_files = list(data_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"\n⚠️ No PDF files found in {data_dir}")
            print("Please add some PDF files and try again.")
            print("\nYou can:")
            print("1. Add your own PDF files")
            print("2. Run lesson 02 to create a sample PDF")
            sys.exit(1)

        print(f"\n✓ Found {len(pdf_files)} PDF files")

        # Create and run the application
        try:
            app = RAGApplication(
                pdf_directory=str(data_dir),
                persist_directory=str(db_dir),
                llm_type="ollama",
                llm_model="llama3.2"
            )

            if app.initialize():
                app.interactive_mode()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nMake sure you have:")
            print("- Ollama installed (https://ollama.ai)")
            print("- A model pulled: ollama pull llama3.2")
            print("- Required packages: pip install -r requirements.txt")
