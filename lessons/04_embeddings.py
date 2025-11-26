"""
=============================================================================
LESSON 4: Embeddings - Converting Text to Vectors
=============================================================================

In this lesson, you'll learn:
- What embeddings are and why they matter
- Different embedding models (free vs paid)
- How to create embeddings for your chunks
- Comparing embedding quality

What are Embeddings?
--------------------
Embeddings are numerical representations (vectors) of text that capture
semantic meaning. Similar texts have similar embeddings.

    Text: "I love pizza"     →  [0.2, -0.5, 0.8, 0.1, ...]
    Text: "Pizza is great"   →  [0.3, -0.4, 0.7, 0.2, ...]  ← Similar!
    Text: "The sky is blue"  →  [-0.1, 0.9, -0.3, 0.5, ...] ← Different!

Why Embeddings for RAG?
-----------------------
1. Enable semantic search (find similar meaning, not just keywords)
2. Fast comparison using vector math
3. Works across languages and paraphrases

    ┌─────────────────┐         ┌─────────────────┐
    │  "How do I      │         │  "Instructions  │
    │   cook pasta?"  │  ─────► │   for making    │  ← Found by meaning!
    │                 │ Similar │   spaghetti"    │
    └─────────────────┘         └─────────────────┘
"""

from typing import List, Tuple
import numpy as np


import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# UNDERSTANDING EMBEDDINGS (No dependencies needed)
# =============================================================================

def explain_embeddings():
    """
    Visual explanation of how embeddings work.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    HOW EMBEDDINGS WORK                         ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  Step 1: Text → Numbers (Embedding Model)                     ║
    ║  ─────────────────────────────────────────                    ║
    ║  "Machine learning"  →  [0.23, -0.45, 0.67, 0.12, ...]       ║
    ║                          ↑                                    ║
    ║                    384-1536 dimensions                        ║
    ║                                                                ║
    ║  Step 2: Compare using Cosine Similarity                      ║
    ║  ────────────────────────────────────────                     ║
    ║                                                                ║
    ║      "ML is great"  ●─────────────● "AI is awesome"          ║
    ║                      \\  angle=15° /   (similar = high cos)    ║
    ║                       \\         /                             ║
    ║                        \\       /                              ║
    ║                         ●                                      ║
    ║                    "I like pizza"                             ║
    ║                    (angle=80° = different)                    ║
    ║                                                                ║
    ║  Cosine Similarity: 1.0 = identical, 0 = unrelated           ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    This is how we measure "how similar" two pieces of text are
    after converting them to embeddings.

    Returns:
        Float between -1 and 1 (1 = identical, 0 = unrelated)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# =============================================================================
# METHOD 1: Free Local Embeddings (Sentence Transformers)
# =============================================================================

def create_embeddings_local(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings using free, local models via Sentence Transformers.

    RECOMMENDED for:
    - Learning and development
    - Privacy-sensitive applications
    - No API costs

    Popular models:
    - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
    - all-mpnet-base-v2: Better quality, slower (768 dimensions)
    - multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
    """
    from sentence_transformers import SentenceTransformer

    # Load the model (downloads automatically first time)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(texts, convert_to_numpy=True)

    return embeddings.tolist()


def langchain_local_embeddings():
    """
    Use local embeddings with LangChain integration.

    NOTE: The modern import is from langchain_huggingface (recommended).
    The langchain_community import still works but is deprecated.

    Install: pip install langchain-huggingface
    """
    # Modern import (recommended) - same as production rag_system.py
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fallback to community (deprecated)
        from langchain_community.embeddings import HuggingFaceEmbeddings

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' for GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    return embeddings


# =============================================================================
# METHOD 2: OpenAI Embeddings (Paid, High Quality)
# =============================================================================

def create_embeddings_openai(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings using OpenAI's API.

    RECOMMENDED for:
    - Production applications
    - Highest quality embeddings
    - When cost is acceptable

    Models:
    - text-embedding-3-small: Cheaper, good quality (1536 dimensions)
    - text-embedding-3-large: Best quality (3072 dimensions)
    - text-embedding-ada-002: Legacy model

    Pricing (as of 2024):
    - text-embedding-3-small: $0.02 per 1M tokens
    - text-embedding-3-large: $0.13 per 1M tokens
    """
    from langchain_openai import OpenAIEmbeddings
    import os

    # Requires OPENAI_API_KEY environment variable
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        # api_key=os.getenv("OPENAI_API_KEY")  # Or set explicitly
    )

    vectors = embeddings.embed_documents(texts)
    return vectors


def langchain_openai_embeddings():
    """
    Use OpenAI embeddings with LangChain.
    """
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return embeddings


# =============================================================================
# METHOD 3: Other Embedding Options
# =============================================================================

def other_embedding_options():
    """
    Overview of other embedding options.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║              OTHER EMBEDDING OPTIONS                           ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  COHERE EMBEDDINGS                                            ║
    ║  ├─ High quality, multilingual                                ║
    ║  ├─ embed-english-v3.0 (1024 dims)                           ║
    ║  └─ from langchain_cohere import CohereEmbeddings            ║
    ║                                                                ║
    ║  GOOGLE VERTEX AI                                             ║
    ║  ├─ text-embedding-004                                        ║
    ║  └─ from langchain_google_vertexai import VertexAIEmbeddings ║
    ║                                                                ║
    ║  OLLAMA (Local, Free)                                         ║
    ║  ├─ Run models locally with Ollama                           ║
    ║  ├─ nomic-embed-text, mxbai-embed-large                      ║
    ║  └─ from langchain_ollama import OllamaEmbeddings            ║
    ║                                                                ║
    ║  AWS BEDROCK                                                   ║
    ║  ├─ Amazon Titan embeddings                                   ║
    ║  └─ from langchain_aws import BedrockEmbeddings              ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# COMPARING EMBEDDINGS
# =============================================================================

def compare_embeddings_demo():
    """
    Demonstrate how embeddings capture semantic similarity.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING SIMILARITY DEMO")
    print("=" * 60)

    texts = [
        "Machine learning is a type of artificial intelligence",
        "AI and ML are related technologies",
        "I love eating pizza on Fridays",
        "Deep learning uses neural networks",
    ]

    query = "What is machine learning?"

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Embed all texts and query
        text_embeddings = model.encode(texts)
        query_embedding = model.encode([query])[0]

        print(f"\nQuery: '{query}'")
        print("\nSimilarity scores:")
        print("-" * 50)

        # Calculate similarities
        similarities = []
        for i, (text, embedding) in enumerate(zip(texts, text_embeddings)):
            sim = cosine_similarity(query_embedding.tolist(), embedding.tolist())
            similarities.append((text, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        for text, sim in similarities:
            bar = "█" * int(sim * 30)
            print(f"{sim:.3f} {bar}")
            print(f"       '{text[:50]}...'")
            print()

    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")


# =============================================================================
# PUTTING IT ALL TOGETHER
# =============================================================================

def embedding_pipeline_example():
    """
    Complete example of embedding text chunks.
    """
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │              EMBEDDING PIPELINE FOR RAG                      │
    └─────────────────────────────────────────────────────────────┘

    # Modern imports (recommended)
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import fitz  # PyMuPDF for better PDF extraction
    from langchain_core.documents import Document

    # 1. Load documents with PyMuPDF (recommended)
    doc = fitz.open("your_document.pdf")
    documents = [
        Document(page_content=page.get_text(), metadata={"page": i})
        for i, page in enumerate(doc)
    ]
    doc.close()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Embed a single chunk (for testing)
    single_embedding = embeddings.embed_query("What is machine learning?")
    print(f"Embedding dimensions: {len(single_embedding)}")

    # 5. Embed all chunks (usually done by vector store)
    # chunk_embeddings = embeddings.embed_documents([c.page_content for c in chunks])
    """)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_embeddings():
    """Main demonstration."""
    print("=" * 60)
    print("LESSON 4: Embeddings Demonstration")
    print("=" * 60)

    explain_embeddings()

    # Demo with simple vectors (no dependencies)
    print("\n" + "-" * 40)
    print("COSINE SIMILARITY EXAMPLE (No dependencies)")
    print("-" * 40)

    # Fake embeddings for demonstration
    vec_a = [1.0, 0.0, 0.0]  # Points in x direction
    vec_b = [0.9, 0.1, 0.0]  # Almost same direction
    vec_c = [0.0, 1.0, 0.0]  # Points in y direction (orthogonal)

    print(f"Vector A: {vec_a}")
    print(f"Vector B: {vec_b}")
    print(f"Vector C: {vec_c}")
    print(f"\nSimilarity A-B: {cosine_similarity(vec_a, vec_b):.3f} (very similar)")
    print(f"Similarity A-C: {cosine_similarity(vec_a, vec_c):.3f} (orthogonal/different)")

    # Try real embeddings
    print("\n" + "-" * 40)
    print("REAL EMBEDDINGS DEMO")
    print("-" * 40)

    try:
        compare_embeddings_demo()
    except Exception as e:
        print(f"Could not run demo: {e}")
        print("Install: pip install sentence-transformers")

    other_embedding_options()
    embedding_pipeline_example()


if __name__ == "__main__":
    demonstrate_embeddings()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Embeddings convert text to numerical vectors
    2. Similar meaning = similar vectors (high cosine similarity)
    3. For learning: Use HuggingFaceEmbeddings (free, local)
    4. For production: Consider OpenAI embeddings

    Quick Code for Your RAG (Modern Imports):
    -----------------------------------------
    # Modern import (recommended)
    from langchain_huggingface import HuggingFaceEmbeddings

    # Or fallback to community (deprecated)
    # from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Embed a query
    query_vector = embeddings.embed_query("your question here")

    NEXT: Lesson 5 - Vector Stores
    """)
