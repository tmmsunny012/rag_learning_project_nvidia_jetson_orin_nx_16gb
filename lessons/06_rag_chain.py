"""
=============================================================================
LESSON 6: Building the RAG Chain - Bringing It All Together
=============================================================================

In this lesson, you'll learn:
- How to connect all components into a RAG chain
- Different chain architectures
- Prompt templates for RAG
- Using different LLMs (free and paid)

The RAG Chain:
--------------
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   User      │     │  Retriever  │     │  Retrieved  │
    │   Query     │ ──► │  (Vector    │ ──► │  Documents  │
    └─────────────┘     │   Store)    │     └─────────────┘
                        └─────────────┘            │
                                                   ▼
                        ┌─────────────┐     ┌─────────────┐
                        │   Final     │     │   Prompt    │
                        │   Answer    │ ◄── │   Template  │
                        └─────────────┘     │   + LLM     │
                                            └─────────────┘
"""

from typing import List, Optional


import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# LLM OPTIONS
# =============================================================================

def llm_options():
    """
    Overview of LLM options for RAG.
    """
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                      LLM OPTIONS FOR RAG                       ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  FREE / LOCAL OPTIONS                                         ║
    ║  ─────────────────────                                        ║
    ║  OLLAMA (Recommended for learning)                            ║
    ║  ├─ Run models locally: llama3.2, mistral, phi3              ║
    ║  ├─ from langchain_ollama import ChatOllama                  ║
    ║  ├─ Install: https://ollama.ai                               ║
    ║  └─ Supports REMOTE servers (e.g., Jetson Orin NX)           ║
    ║                                                                ║
    ║  HUGGING FACE                                                  ║
    ║  ├─ Free tier available                                       ║
    ║  ├─ Many open models                                          ║
    ║  └─ from langchain_huggingface import HuggingFacePipeline    ║
    ║                                                                ║
    ║  PAID OPTIONS                                                  ║
    ║  ────────────────                                             ║
    ║  OPENAI                                                        ║
    ║  ├─ GPT-4o, GPT-4o-mini                                       ║
    ║  ├─ from langchain_openai import ChatOpenAI                  ║
    ║  └─ Best quality, ~$0.01-0.03 per 1K tokens                  ║
    ║                                                                ║
    ║  ANTHROPIC (Claude)                                           ║
    ║  ├─ Claude 3.5 Sonnet, Claude 3 Opus                         ║
    ║  ├─ from langchain_anthropic import ChatAnthropic            ║
    ║  └─ Great for complex reasoning                               ║
    ║                                                                ║
    ║  GOOGLE                                                        ║
    ║  ├─ Gemini Pro, Gemini Flash                                  ║
    ║  └─ from langchain_google_genai import ChatGoogleGenerativeAI║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# CREATING LLMS
# =============================================================================

def create_ollama_llm(model_name: str = "llama3.2", base_url: str = None):
    """
    Create an Ollama LLM (free, runs locally or on remote server).

    Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Pull a model: ollama pull llama3.2
    3. Ollama server runs automatically

    Remote Ollama Setup (e.g., NVIDIA Jetson Orin NX):
    --------------------------------------------------
    You can run Ollama on a remote server with GPU and connect to it:

    1. On your server (e.g., Jetson Orin NX at 192.168.178.124):
       - Install Ollama
       - Set environment variable: OLLAMA_HOST=0.0.0.0:11434
       - Restart Ollama service

    2. Pass the base_url parameter:
       llm = create_ollama_llm("llama3.2", base_url="http://192.168.178.124:11434")

    Args:
        model_name: Name of the Ollama model (e.g., "llama3.2", "mistral")
        base_url: Custom Ollama server URL (e.g., "http://192.168.178.124:11434")
                  If None, uses localhost or OLLAMA_HOST environment variable

    Returns:
        ChatOllama instance configured for the specified server
    """
    import os
    from langchain_ollama import ChatOllama

    # Use custom base_url if provided, otherwise check environment variable
    ollama_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        temperature=0,  # 0 = deterministic, 1 = creative
        base_url=ollama_url
    )

    print(f"Ollama configured: {ollama_url} (model: {model_name})")
    return llm


def create_openai_llm(model_name: str = "gpt-4o-mini"):
    """
    Create an OpenAI LLM.

    Prerequisites:
    1. Set OPENAI_API_KEY environment variable
    2. pip install langchain-openai
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
    )

    return llm


def create_anthropic_llm(model_name: str = "claude-3-5-sonnet-20241022"):
    """
    Create an Anthropic Claude LLM.

    Prerequisites:
    1. Set ANTHROPIC_API_KEY environment variable
    2. pip install langchain-anthropic
    """
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    return llm


# =============================================================================
# RAG PROMPT TEMPLATES
# =============================================================================

def get_rag_prompt():
    """
    Create a prompt template for RAG.

    The prompt template has two key variables:
    - {context}: The retrieved documents
    - {question}: The user's question
    """
    from langchain_core.prompts import PromptTemplate

    template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know,
don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


def get_custom_rag_prompt(system_instruction: str):
    """
    Create a custom RAG prompt with specific instructions.

    Examples of system instructions:
    - "You are a helpful assistant for our company documentation."
    - "Answer in bullet points only."
    - "If unsure, ask for clarification."
    """
    from langchain_core.prompts import PromptTemplate

    template = f"""{system_instruction}

Use the following context to answer the question.
If the answer is not in the context, say "I don't have information about that."

Context:
{{context}}

Question: {{question}}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    return prompt


# =============================================================================
# METHOD 1: Simple RAG Chain (LangChain Expression Language)
# =============================================================================

def create_simple_rag_chain(retriever, llm):
    """
    Create a simple RAG chain using LangChain Expression Language (LCEL).

    This is the MODERN and RECOMMENDED approach.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # Create prompt
    prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:""")

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the chain using LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =============================================================================
# METHOD 2: RetrievalQA Chain (Classic Approach)
# =============================================================================

def create_retrieval_qa_chain(retriever, llm):
    """
    Create a RetrievalQA chain (older but still valid approach).

    This provides more built-in functionality like:
    - Source document tracking
    - Built-in prompts
    """
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all docs in context
        retriever=retriever,
        return_source_documents=True,  # Include sources in response
    )

    return qa_chain


# =============================================================================
# METHOD 3: Conversational RAG (With Memory)
# =============================================================================

def create_conversational_rag_chain(retriever, llm):
    """
    Create a RAG chain that remembers conversation history.

    Great for:
    - Follow-up questions
    - Chatbot interfaces
    - Complex multi-turn conversations
    """
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    # Create memory to store conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    return chain


# =============================================================================
# COMPLETE RAG PIPELINE
# =============================================================================

def complete_rag_pipeline():
    """
    Show the complete RAG pipeline code.
    """
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │              COMPLETE RAG PIPELINE CODE                      │
    └─────────────────────────────────────────────────────────────┘
    """)

    code = '''
# Modern imports (recommended)
import fitz  # PyMuPDF for better PDF extraction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Modern import
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama  # or ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================
# STEP 1: Load and Process Documents (PyMuPDF)
# ============================================
doc = fitz.open("your_document.pdf")
documents = [
    Document(page_content=page.get_text(), metadata={"page": i, "source": "your_document.pdf"})
    for i, page in enumerate(doc)
]
doc.close()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ============================================
# STEP 2: Create Vector Store
# ============================================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================
# STEP 3: Create LLM
# ============================================
# Option A: Local Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

# Option B: Remote Ollama (e.g., Jetson Orin NX)
# llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://192.168.178.124:11434")

# ============================================
# STEP 4: Create RAG Chain
# ============================================
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
If you can't find the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================
# STEP 5: Query the System
# ============================================
question = "What is machine learning?"
answer = rag_chain.invoke(question)
print(answer)
'''

    print(code)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_rag_chain():
    """Main demonstration."""
    print("=" * 60)
    print("LESSON 6: RAG Chain Demonstration")
    print("=" * 60)

    llm_options()

    # Try to create a working demo
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        print("\n" + "-" * 40)
        print("LIVE DEMO: Creating a RAG Chain")
        print("-" * 40)

        # Create sample documents
        documents = [
            Document(
                page_content="Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
                metadata={"source": "ml_intro.pdf", "page": 1}
            ),
            Document(
                page_content="Supervised learning uses labeled data to train models. Examples include classification and regression.",
                metadata={"source": "ml_intro.pdf", "page": 2}
            ),
            Document(
                page_content="Neural networks are inspired by the human brain and consist of layers of interconnected nodes.",
                metadata={"source": "dl_chapter.pdf", "page": 1}
            ),
        ]

        # Create embeddings and vector store
        print("\nCreating embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # Test retrieval
        print("\nTesting retrieval...")
        query = "What is machine learning?"
        docs = retriever.invoke(query)
        print(f"Query: '{query}'")
        print(f"Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs):
            print(f"  {i + 1}. {doc.page_content[:80]}...")

        # Try with Ollama
        try:
            from langchain_ollama import ChatOllama

            print("\n" + "-" * 40)
            print("Testing with Ollama LLM...")
            print("-" * 40)

            llm = ChatOllama(model="llama3.2", temperature=0)

            prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:""")

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(query)
            print(f"\nQuestion: {query}")
            print(f"Answer: {answer}")

        except Exception as e:
            print(f"\nOllama not available: {e}")
            print("To use Ollama:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Run: ollama pull llama3.2")
            print("3. pip install langchain-ollama")

    except ImportError as e:
        print(f"\nCouldn't run demo: {e}")
        print("Install: pip install chromadb sentence-transformers langchain")

    complete_rag_pipeline()


if __name__ == "__main__":
    demonstrate_rag_chain()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. RAG Chain = Retriever + Prompt + LLM
    2. Use LCEL (LangChain Expression Language) for modern chains
    3. Ollama provides free local LLMs
    4. OpenAI/Anthropic for production quality

    Chain Types:
    - Simple RAG: One-shot Q&A
    - RetrievalQA: With source tracking
    - Conversational: With memory

    NEXT: Lesson 7 - Interactive Query Interface
    """)
