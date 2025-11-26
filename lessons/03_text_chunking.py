"""
=============================================================================
LESSON 3: Text Chunking Strategies
=============================================================================

In this lesson, you'll learn:
- Why we need to chunk text
- Different chunking strategies
- How chunk size affects RAG performance
- Best practices for chunking

Why Chunk Text?
---------------
1. LLMs have token limits (context window)
2. Embeddings work better on smaller, focused text
3. Retrieval is more precise with smaller chunks
4. Reduces noise in the context

The Art of Chunking:
--------------------
- Too small: Loses context, fragments meaning
- Too large: Dilutes relevance, wastes tokens
- Just right: Captures complete thoughts, optimal retrieval

    ┌─────────────────────────────────────────────────────────┐
    │                    Original Document                     │
    │  "Machine learning is a subset of AI. It enables        │
    │   systems to learn from data. Neural networks are       │
    │   inspired by biological neurons..."                     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │   Chunk 1     │  │   Chunk 2     │  │   Chunk 3     │
    │ "Machine      │  │ "It enables   │  │ "Neural       │
    │  learning..." │  │  systems..."  │  │  networks..." │
    └───────────────┘  └───────────────┘  └───────────────┘
"""

from typing import List


import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# METHOD 1: Simple Character Splitting
# =============================================================================

def simple_character_split(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text by character count with overlap.

    This is the simplest method but NOT recommended because:
    - Can split words in the middle
    - Ignores sentence boundaries
    - May break semantic meaning

    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move back by overlap amount

    return chunks


# =============================================================================
# METHOD 2: Recursive Character Splitting (Recommended)
# =============================================================================

def recursive_character_split(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text recursively, trying to preserve natural boundaries.

    This method tries to split on (in order):
    1. Paragraphs (double newline)
    2. Sentences (period, exclamation, question mark)
    3. Words (space)
    4. Characters (last resort)

    This is LangChain's default and RECOMMENDED approach.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try each separator in order
    )

    chunks = splitter.split_text(text)
    return chunks


# =============================================================================
# METHOD 3: Sentence-Based Splitting
# =============================================================================

def sentence_based_split(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text by sentences, grouping a fixed number together.

    Good for:
    - Ensuring complete thoughts
    - Consistent semantic units

    Requires: nltk for sentence tokenization
    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)

    return chunks


# =============================================================================
# METHOD 4: Semantic Chunking (Advanced)
# =============================================================================

def semantic_chunking_example():
    """
    Semantic chunking splits based on meaning, not just characters.

    This is more advanced and uses embeddings to determine where
    to split based on semantic similarity.

    The idea:
    1. Split into sentences
    2. Create embeddings for each sentence
    3. Compare adjacent sentences
    4. Split where similarity drops significantly
    """
    print("""
    Semantic Chunking Process:
    --------------------------

    Sentence 1: "Machine learning is powerful." ──┐
    Sentence 2: "It uses data to learn."         ──┼── High similarity → Same chunk
    Sentence 3: "Models improve over time."      ──┘

    Sentence 4: "Now let's talk about cooking."  ──┐
    Sentence 5: "Pasta requires boiling water."  ──┼── Low similarity → New chunk
                                                   │   (topic changed!)

    This method requires:
    - An embedding model
    - Similarity threshold tuning
    - More computational resources
    """)


# =============================================================================
# METHOD 5: Document-Aware Splitting (Best for Structured Documents)
# =============================================================================

def document_aware_split():
    """
    Split based on document structure (headers, sections).

    Best for:
    - Technical documentation
    - Legal documents
    - Academic papers

    LangChain provides specialized splitters:
    - MarkdownHeaderTextSplitter
    - HTMLHeaderTextSplitter
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    markdown_text = """
# Introduction
This is the introduction section. It explains the basics.

## Background
Some background information here.

## Methodology
The methodology section describes our approach.

# Results
Here are the results of our experiment.

# Conclusion
The conclusion summarizes everything.
"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = splitter.split_text(markdown_text)

    return splits


# =============================================================================
# LANGCHAIN INTEGRATION (What you'll use in RAG)
# =============================================================================

def langchain_document_splitting(documents):
    """
    Split LangChain Document objects (from PDF loader).

    This is what you'll use in your RAG pipeline!
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Characters per chunk
        chunk_overlap=200,     # Overlap between chunks
        length_function=len,
        add_start_index=True,  # Track position in original doc
    )

    # Split documents while preserving metadata
    split_docs = splitter.split_documents(documents)

    return split_docs


# =============================================================================
# CHOOSING THE RIGHT CHUNK SIZE
# =============================================================================

def chunk_size_guide():
    """
    Guide for choosing chunk size based on use case.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    CHUNK SIZE GUIDE                            ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  SMALLER CHUNKS (100-500 chars)                               ║
    ║  ├─ ✓ More precise retrieval                                  ║
    ║  ├─ ✓ Good for Q&A with specific answers                      ║
    ║  ├─ ✗ May lose context                                        ║
    ║  └─ Best for: FAQs, definitions, quick lookups                ║
    ║                                                                ║
    ║  MEDIUM CHUNKS (500-1500 chars) ← RECOMMENDED START           ║
    ║  ├─ ✓ Good balance of context and precision                   ║
    ║  ├─ ✓ Works well for most use cases                           ║
    ║  └─ Best for: General RAG applications                        ║
    ║                                                                ║
    ║  LARGER CHUNKS (1500-3000 chars)                              ║
    ║  ├─ ✓ More context preserved                                  ║
    ║  ├─ ✓ Good for complex topics                                 ║
    ║  ├─ ✗ May include irrelevant information                      ║
    ║  └─ Best for: Summarization, complex analysis                 ║
    ║                                                                ║
    ║  OVERLAP (10-20% of chunk size)                               ║
    ║  ├─ Prevents information loss at boundaries                   ║
    ║  └─ Example: 1000 char chunks → 100-200 char overlap          ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_chunking():
    """Demonstrate different chunking strategies."""
    print("=" * 60)
    print("LESSON 3: Text Chunking Demonstration")
    print("=" * 60)

    sample_text = """
Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.
It focuses on developing computer programs that can access data and use it
to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction. The goal is to look for patterns in data
and make better decisions in the future based on the examples that we provide.

Neural networks are a set of algorithms, modeled loosely after the human brain,
that are designed to recognize patterns. They interpret sensory data through a
kind of machine perception, labeling or clustering raw input.

Deep learning is a subset of machine learning where artificial neural networks,
algorithms inspired by the human brain, learn from large amounts of data.
    """.strip()

    print(f"\nOriginal text length: {len(sample_text)} characters")
    print("\n" + "-" * 40)

    # Method 1: Simple split
    print("\n1. SIMPLE CHARACTER SPLIT (chunk_size=200, overlap=20)")
    chunks = simple_character_split(sample_text, chunk_size=200, overlap=20)
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   Chunk 1: '{chunks[0][:50]}...'")
    print(f"   ⚠ Notice: May split mid-word!")

    # Method 2: Recursive split (requires langchain)
    print("\n2. RECURSIVE CHARACTER SPLIT (chunk_size=200, overlap=20)")
    try:
        chunks = recursive_character_split(sample_text, chunk_size=200, overlap=20)
        print(f"   Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i + 1}: '{chunk[:50]}...'")
        print(f"   ✓ Respects sentence boundaries!")
    except ImportError:
        print("   LangChain not installed. Run: pip install langchain")

    # Chunk size guide
    print("\n3. CHUNK SIZE RECOMMENDATIONS")
    chunk_size_guide()


if __name__ == "__main__":
    demonstrate_chunking()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. Use RecursiveCharacterTextSplitter for most cases
    2. Start with chunk_size=1000, overlap=200
    3. Adjust based on your document type and query patterns
    4. Test different sizes with your actual queries

    Quick Code for Your RAG:
    ------------------------
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    NEXT: Lesson 4 - Embeddings
    """)
