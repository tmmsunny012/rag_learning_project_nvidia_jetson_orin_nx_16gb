"""
=============================================================================
LESSON 1: Introduction to RAG (Retrieval-Augmented Generation)
=============================================================================

What is RAG?
------------
RAG is a technique that enhances Large Language Models (LLMs) by providing them
with relevant context from your own documents. Instead of relying solely on
the model's training data, RAG retrieves relevant information from your
documents and includes it in the prompt.

The RAG Pipeline:
-----------------
1. LOAD      → Load your PDF documents
2. SPLIT    → Split documents into smaller chunks
3. EMBED    → Convert chunks into numerical vectors (embeddings)
4. STORE    → Store embeddings in a vector database
5. RETRIEVE → Find relevant chunks based on user query
6. GENERATE → Send query + relevant chunks to LLM for answer

Why RAG?
--------
- Use your private/custom data with LLMs
- Reduce hallucinations by grounding responses in real data
- Keep information up-to-date without retraining
- Cost-effective compared to fine-tuning

Let's start with a simple overview diagram:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Your      │     │   Text      │     │  Embedding  │
    │   PDFs      │ ──► │   Chunks    │ ──► │   Vectors   │
    └─────────────┘     └─────────────┘     └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │   Vector    │
                                            │   Database  │
                                            └─────────────┘
                                                   │
    ┌─────────────┐     ┌─────────────┐           │
    │   User      │     │  Relevant   │           │
    │   Query     │ ──► │  Chunks     │ ◄─────────┘
    └─────────────┘     └─────────────┘
                              │
                              ▼
                        ┌─────────────┐     ┌─────────────┐
                        │   LLM       │ ──► │   Answer    │
                        │   + Context │     │             │
                        └─────────────┘     └─────────────┘

"""

# Simple demonstration of the concept
def demonstrate_rag_concept():
    """
    This function demonstrates the RAG concept without any dependencies.
    It shows how context can improve LLM responses.
    """

    # Simulated document chunks (in real RAG, these come from your PDFs)
    document_chunks = [
        "The company was founded in 2020 by John Smith.",
        "Our main product is CloudSync, a file synchronization tool.",
        "The headquarters is located in San Francisco, California.",
        "We have 150 employees across 5 countries.",
        "Annual revenue in 2023 was $50 million."
    ]

    # Simulated user query
    user_query = "Where is the company headquartered?"

    # Step 1: Find relevant chunks (simplified - real RAG uses vector similarity)
    relevant_chunks = []
    for chunk in document_chunks:
        # Simple keyword matching (real RAG uses embeddings)
        if any(word in chunk.lower() for word in ["headquarters", "located", "office"]):
            relevant_chunks.append(chunk)

    print("=" * 60)
    print("RAG CONCEPT DEMONSTRATION")
    print("=" * 60)
    print(f"\nUser Query: {user_query}")
    print(f"\nRetrieved Context: {relevant_chunks}")

    # Step 2: Create augmented prompt
    context = "\n".join(relevant_chunks)
    augmented_prompt = f"""
    Based on the following context, answer the question.

    Context:
    {context}

    Question: {user_query}

    Answer:
    """

    print(f"\nAugmented Prompt (sent to LLM):")
    print("-" * 40)
    print(augmented_prompt)
    print("-" * 40)

    # In real RAG, this prompt would be sent to an LLM
    # The LLM would respond: "The company is headquartered in San Francisco, California."

    return augmented_prompt


if __name__ == "__main__":
    demonstrate_rag_concept()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
    Now that you understand the concept, let's build a real RAG system!

    Run the lessons in order:
    1. 01_introduction.py      ← You are here
    2. 02_pdf_loading.py       → Load PDF documents
    3. 03_text_chunking.py     → Split text into chunks
    4. 04_embeddings.py        → Create vector embeddings
    5. 05_vector_store.py      → Store and retrieve vectors
    6. 06_rag_chain.py         → Build the complete RAG chain
    7. 07_query_interface.py   → Interactive query system
    """)
