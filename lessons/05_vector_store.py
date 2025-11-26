"""
=============================================================================
LESSON 5: Vector Stores - Storing and Retrieving Embeddings
=============================================================================

In this lesson, you'll learn:
- What vector stores are and why we need them
- Different vector store options
- How to store and retrieve embeddings
- Similarity search techniques

What is a Vector Store?
-----------------------
A vector store (or vector database) is a specialized database designed to
store, index, and search vector embeddings efficiently.

    ┌─────────────────────────────────────────────────────────────┐
    │                     VECTOR STORE                            │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   ID    │  Vector (Embedding)          │  Metadata          │
    │  ───────┼──────────────────────────────┼─────────────────── │
    │   1     │  [0.2, -0.5, 0.8, ...]       │  {page: 1, src:..} │
    │   2     │  [0.3, -0.4, 0.7, ...]       │  {page: 2, src:..} │
    │   3     │  [-0.1, 0.9, -0.3, ...]      │  {page: 3, src:..} │
    │   ...   │  ...                          │  ...               │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
            Query: "What is machine learning?"
                              │
                    ┌─────────┴─────────┐
                    │  Similarity       │
                    │  Search           │
                    └─────────┬─────────┘
                              │
                              ▼
            Returns: Top K most similar documents
"""

from typing import List, Optional
from pathlib import Path


import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# VECTOR STORE OPTIONS
# =============================================================================

def vector_store_comparison():
    """
    Compare different vector store options.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              VECTOR STORE COMPARISON                           ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  CHROMA (Recommended for Learning)                            ║
    ║  ├─ ✓ Easy to use, no setup required                         ║
    ║  ├─ ✓ Runs locally, embeds in your app                       ║
    ║  ├─ ✓ Persistent storage available                           ║
    ║  ├─ ✗ Not suitable for very large datasets                   ║
    ║  └─ pip install chromadb                                      ║
    ║                                                                ║
    ║  FAISS (Facebook AI Similarity Search)                        ║
    ║  ├─ ✓ Very fast similarity search                            ║
    ║  ├─ ✓ Good for larger datasets                               ║
    ║  ├─ ✓ GPU support available                                  ║
    ║  ├─ ✗ More complex to set up persistence                     ║
    ║  └─ pip install faiss-cpu (or faiss-gpu)                     ║
    ║                                                                ║
    ║  PINECONE (Cloud-based)                                       ║
    ║  ├─ ✓ Fully managed, scales automatically                    ║
    ║  ├─ ✓ Good for production                                    ║
    ║  ├─ ✗ Requires API key and internet                          ║
    ║  └─ pip install pinecone-client                              ║
    ║                                                                ║
    ║  WEAVIATE (Open Source, Self-hosted)                          ║
    ║  ├─ ✓ Feature-rich, GraphQL API                              ║
    ║  ├─ ✓ Hybrid search (vector + keyword)                       ║
    ║  └─ pip install weaviate-client                              ║
    ║                                                                ║
    ║  QDRANT (Open Source)                                         ║
    ║  ├─ ✓ Fast, Rust-based                                       ║
    ║  ├─ ✓ Good filtering capabilities                            ║
    ║  └─ pip install qdrant-client                                ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# METHOD 1: ChromaDB (Recommended for Learning)
# =============================================================================

def create_chroma_store(documents, embeddings, persist_directory: Optional[str] = None):
    """
    Create a ChromaDB vector store.

    ChromaDB is perfect for learning because:
    - Zero configuration
    - Works in-memory or with persistence
    - Built-in embedding support
    """
    from langchain_community.vectorstores import Chroma

    if persist_directory:
        # Persistent storage
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        # In-memory (lost when program ends)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )

    return vectorstore


def load_existing_chroma(persist_directory: str, embeddings):
    """
    Load an existing ChromaDB store from disk.
    """
    from langchain_community.vectorstores import Chroma

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectorstore


# =============================================================================
# METHOD 2: FAISS (Fast Similarity Search)
# =============================================================================

def create_faiss_store(documents, embeddings, save_path: Optional[str] = None):
    """
    Create a FAISS vector store.

    FAISS is great for:
    - Fast similarity search
    - Larger datasets
    - GPU acceleration
    """
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    if save_path:
        # Save to disk
        vectorstore.save_local(save_path)

    return vectorstore


def load_existing_faiss(save_path: str, embeddings):
    """
    Load an existing FAISS store from disk.
    """
    from langchain_community.vectorstores import FAISS

    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required for loading
    )

    return vectorstore


# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def demonstrate_similarity_search(vectorstore, query: str, k: int = 3):
    """
    Demonstrate different types of similarity search.
    """
    print(f"\nQuery: '{query}'")
    print("=" * 50)

    # Method 1: Basic similarity search
    print("\n1. BASIC SIMILARITY SEARCH")
    print("-" * 40)
    results = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")

    # Method 2: Similarity search with scores
    print("\n2. SIMILARITY SEARCH WITH SCORES")
    print("-" * 40)
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"\nResult {i + 1}:")
        print(f"  Score: {score:.4f}")  # Lower = more similar for L2 distance
        print(f"  Content: {doc.page_content[:100]}...")

    # Method 3: MMR (Maximum Marginal Relevance)
    print("\n3. MMR SEARCH (Diverse Results)")
    print("-" * 40)
    print("MMR balances relevance with diversity to avoid redundant results")
    mmr_results = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=10,  # Fetch more candidates
        lambda_mult=0.5  # 0=max diversity, 1=max relevance
    )
    for i, doc in enumerate(mmr_results):
        print(f"\nResult {i + 1}:")
        print(f"  Content: {doc.page_content[:100]}...")


# =============================================================================
# FILTERING WITH METADATA
# =============================================================================

def demonstrate_metadata_filtering(vectorstore, query: str):
    """
    Show how to filter search results by metadata.
    """
    print("\nMETADATA FILTERING")
    print("=" * 50)
    print("""
    You can filter results based on metadata:

    # Filter by page number
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"page": 1}  # Only from page 1
    )

    # Filter by source file
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"source": "chapter1.pdf"}
    )

    # Complex filters (Chroma)
    results = vectorstore.similarity_search(
        query,
        k=3,
        filter={
            "$and": [
                {"page": {"$gte": 1}},
                {"page": {"$lte": 10}}
            ]
        }
    )
    """)


# =============================================================================
# COMPLETE EXAMPLE
# =============================================================================

def complete_vector_store_example():
    """
    Complete working example of creating and querying a vector store.
    """
    code = '''
    # Modern imports (recommended)
    import fitz  # PyMuPDF for better PDF extraction
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings  # Modern import
    from langchain_community.vectorstores import Chroma

    # 1. Load PDF with PyMuPDF (recommended for better extraction)
    doc = fitz.open("your_document.pdf")
    documents = [
        Document(page_content=page.get_text(), metadata={"page": i, "source": "your_document.pdf"})
        for i, page in enumerate(doc)
    ]
    doc.close()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Saves to disk
    )

    # 5. Query the store
    query = "What is machine learning?"
    results = vectorstore.similarity_search(query, k=3)

    for doc in results:
        print(doc.page_content)
        print("---")
    '''

    print("\n" + "=" * 60)
    print("COMPLETE VECTOR STORE EXAMPLE")
    print("=" * 60)
    print(code)


# =============================================================================
# CREATING A RETRIEVER
# =============================================================================

def create_retriever_example():
    """
    Show how to create a retriever from a vector store.

    A retriever is a higher-level interface that you'll use
    in the RAG chain.
    """
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    CREATING A RETRIEVER                      │
    └─────────────────────────────────────────────────────────────┘

    A retriever wraps the vector store and provides a clean interface
    for the RAG chain:

    # Basic retriever
    retriever = vectorstore.as_retriever()

    # With search parameters
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # or "mmr"
        search_kwargs={
            "k": 5,                # Number of results
            "score_threshold": 0.5 # Minimum similarity (optional)
        }
    )

    # With MMR for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    # Use the retriever
    docs = retriever.invoke("What is machine learning?")
    """)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_vector_stores():
    """Main demonstration."""
    print("=" * 60)
    print("LESSON 5: Vector Stores Demonstration")
    print("=" * 60)

    vector_store_comparison()

    # Try to create a simple demo
    try:
        from langchain_community.vectorstores import Chroma
        # Modern import (recommended)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        print("\n" + "-" * 40)
        print("LIVE DEMO: Creating and Querying a Vector Store")
        print("-" * 40)

        # Create sample documents
        documents = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                metadata={"source": "ml_intro.pdf", "page": 1}
            ),
            Document(
                page_content="Neural networks are computing systems inspired by biological neural networks in the brain.",
                metadata={"source": "ml_intro.pdf", "page": 2}
            ),
            Document(
                page_content="Deep learning uses multiple layers of neural networks to progressively extract features.",
                metadata={"source": "dl_chapter.pdf", "page": 1}
            ),
            Document(
                page_content="Python is a popular programming language used in data science and machine learning.",
                metadata={"source": "python_guide.pdf", "page": 1}
            ),
            Document(
                page_content="Pizza is a traditional Italian dish consisting of a flat bread base with toppings.",
                metadata={"source": "cooking.pdf", "page": 5}
            ),
        ]

        # Create embeddings
        print("\nCreating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings
        )

        # Demonstrate search
        demonstrate_similarity_search(vectorstore, "What is machine learning?")

    except ImportError as e:
        print(f"\nCouldn't run live demo: {e}")
        print("Install required packages: pip install chromadb sentence-transformers")

    complete_vector_store_example()
    create_retriever_example()


if __name__ == "__main__":
    demonstrate_vector_stores()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Vector stores efficiently store and search embeddings
    2. ChromaDB is perfect for learning (easy setup)
    3. FAISS is great for performance
    4. Use retrievers to interface with RAG chains

    Quick Code Summary (Modern Imports):
    ------------------------------------
    from langchain_community.vectorstores import Chroma
    # Modern import (recommended)
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    NEXT: Lesson 6 - Building the RAG Chain
    """)
