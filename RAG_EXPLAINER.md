# RAG (Retrieval-Augmented Generation) - Complete Guide

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Why Use RAG?](#why-use-rag)
3. [RAG vs Other Approaches](#rag-vs-other-approaches)
4. [The RAG Pipeline](#the-rag-pipeline)
5. [Component Deep Dive](#component-deep-dive)
6. [Key Concepts](#key-concepts)
7. [Code Examples](#code-examples)
8. [Best Practices](#best-practices)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [Advanced Topics](#advanced-topics)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances Large Language Models (LLMs) by giving them access to external knowledge sources. Instead of relying solely on what the model learned during training, RAG retrieves relevant information from your documents and includes it in the prompt.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRADITIONAL LLM                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User Question ──────────────────────────────► LLM ────► Answer    │
│                                                  │                   │
│                                         (Uses only training         │
│                                          data, may hallucinate)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                           RAG SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User Question ────┬────────────────────────────────────────┐      │
│                     │                                        │      │
│                     ▼                                        ▼      │
│              ┌─────────────┐                          ┌──────────┐  │
│              │  Retrieve   │                          │   LLM    │  │
│              │  Relevant   │─── Context + Question ──►│ Generate │  │
│              │  Documents  │                          │  Answer  │  │
│              └─────────────┘                          └──────────┘  │
│                     │                                        │      │
│                     ▼                                        ▼      │
│              Your PDF/Docs                              Answer      │
│              (Knowledge Base)                      (Grounded in     │
│                                                     your data)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Simple Analogy

Think of RAG like an **open-book exam**:
- **Without RAG**: The student (LLM) answers from memory only
- **With RAG**: The student can look up information in their textbook (your documents) before answering

---

## Why Use RAG?

### Problems RAG Solves

| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| **Outdated Information** | Model only knows training data (cutoff date) | Can access current documents |
| **Hallucinations** | Model may make up facts | Answers grounded in real documents |
| **Domain Knowledge** | Generic knowledge only | Access to your specific data |
| **Privacy** | Can't use private data | Your data stays local |
| **Cost** | Fine-tuning is expensive | No retraining needed |
| **Transparency** | "Black box" answers | Can show source documents |

### When to Use RAG

**Good Use Cases:**
- Question answering over company documents
- Customer support chatbots
- Legal/medical document analysis
- Technical documentation search
- Research paper analysis
- Personal knowledge management

**Not Ideal For:**
- General conversation (no documents needed)
- Creative writing tasks
- Simple calculations
- Real-time data (use APIs instead)

---

## RAG vs Other Approaches

```
┌────────────────────────────────────────────────────────────────────────┐
│                    APPROACHES TO CUSTOMIZE LLMS                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  APPROACH          COST      TIME       BEST FOR                       │
│  ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│  Prompting         Free      Minutes    Simple customization           │
│  (Few-shot)                             Adding examples to prompt       │
│                                                                         │
│  RAG               Low       Hours      Adding knowledge/documents     │
│  (This project!)                        Dynamic, updatable content     │
│                                                                         │
│  Fine-tuning       Medium    Days       Teaching new behaviors         │
│                                         Consistent style/format        │
│                                                                         │
│  Pre-training      Very High Weeks+     Creating specialized model     │
│                                         (Rare, requires huge data)     │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Cost** | Low (just inference) | High (GPU training) |
| **Update Data** | Easy (add new docs) | Requires retraining |
| **Transparency** | Can cite sources | Black box |
| **Data Privacy** | Data stays local | Data used in training |
| **Best For** | Knowledge retrieval | Behavior/style change |

---

## The RAG Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE STAGES                               │
└─────────────────────────────────────────────────────────────────────────┘

 INDEXING PHASE (Done once, when you add documents)
 ════════════════════════════════════════════════════

   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  1.LOAD  │────►│ 2.SPLIT  │────►│ 3.EMBED  │────►│ 4.STORE  │
   │   PDFs   │     │  Chunks  │     │ Vectors  │     │ Database │
   └──────────┘     └──────────┘     └──────────┘     └──────────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
    Raw PDF         Text chunks      Numerical        ChromaDB/
    documents       (1000 chars)     vectors          FAISS
                                     [0.2, -0.5...]


 QUERY PHASE (Every time user asks a question)
 ════════════════════════════════════════════════════

   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ 5.QUERY  │────►│6.RETRIEVE│────►│ 7.AUGMENT│────►│8.GENERATE│
   │  Input   │     │  Search  │     │  Prompt  │     │  Answer  │
   └──────────┘     └──────────┘     └──────────┘     └──────────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
   "What is ML?"    Find similar     Add context       LLM creates
                    chunks           to prompt         final answer
```

### Step-by-Step Breakdown

#### Stage 1: LOAD (Document Loading)
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Result: List of Document objects
# [Document(page_content="...", metadata={"source": "...", "page": 0})]
```

**What happens:**
- PDF is read page by page
- Text is extracted from each page
- Metadata (filename, page number) is preserved

#### Stage 2: SPLIT (Text Chunking)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
chunks = splitter.split_documents(documents)
```

**Why chunk?**
- LLMs have token limits
- Smaller chunks = more precise retrieval
- Overlap prevents losing context at boundaries

```
Original Document:
┌────────────────────────────────────────────────────────────────┐
│ Machine learning is a subset of AI. It enables systems to     │
│ learn from data without explicit programming. Neural networks │
│ are inspired by the brain. They consist of layers...          │
└────────────────────────────────────────────────────────────────┘

After Chunking (with overlap):
┌─────────────────────┐
│ Chunk 1:            │
│ Machine learning is │
│ a subset of AI. It  │
│ enables systems to  │
│ learn from data...  │
└─────────────────────┘
         ┌─────────────────────┐
         │ Chunk 2:            │  ◄── Overlaps with Chunk 1
         │ ...systems to learn │
         │ from data without   │
         │ explicit programming│
         │ Neural networks...  │
         └─────────────────────┘
```

#### Stage 3: EMBED (Create Embeddings)
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Convert text to vector
vector = embeddings.embed_query("What is machine learning?")
# Result: [0.023, -0.456, 0.789, ...] (384 dimensions)
```

**What are embeddings?**
- Numerical representation of text meaning
- Similar meanings = similar vectors
- Enables semantic search (find by meaning, not just keywords)

```
Text Embeddings Visualization (simplified to 2D):

                    "AI and ML"
                         ●
                        /
                       /
    "Machine         /
     Learning" ●────●
                    \
                     \
                      ● "Neural Networks"


                                        ● "Pizza recipes"
                                          (far away = different topic)
```

#### Stage 4: STORE (Vector Database)
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

**What's stored:**
```
┌────────────────────────────────────────────────────────────────┐
│                      VECTOR DATABASE                            │
├──────┬─────────────────────────────┬───────────────────────────┤
│  ID  │  Embedding Vector           │  Metadata + Text          │
├──────┼─────────────────────────────┼───────────────────────────┤
│  1   │  [0.12, -0.45, 0.78, ...]  │  {source: "doc.pdf",      │
│      │                             │   page: 1,                 │
│      │                             │   text: "Machine..."}      │
├──────┼─────────────────────────────┼───────────────────────────┤
│  2   │  [0.34, -0.23, 0.56, ...]  │  {source: "doc.pdf",      │
│      │                             │   page: 1,                 │
│      │                             │   text: "Neural..."}       │
└──────┴─────────────────────────────┴───────────────────────────┘
```

#### Stage 5-6: QUERY & RETRIEVE
```python
query = "What is machine learning?"

# Convert query to vector
query_vector = embeddings.embed_query(query)

# Find similar vectors in database
results = vectorstore.similarity_search(query, k=3)
# Returns top 3 most similar chunks
```

**How similarity search works:**
```
Query: "What is machine learning?"
Query Vector: [0.15, -0.42, 0.81, ...]

Compare with all stored vectors using cosine similarity:

Chunk 1: similarity = 0.92  ◄── Most similar (returned)
Chunk 2: similarity = 0.87  ◄── Second most similar (returned)
Chunk 3: similarity = 0.85  ◄── Third most similar (returned)
Chunk 4: similarity = 0.34      (not relevant enough)
Chunk 5: similarity = 0.12      (about different topic)
```

#### Stage 7: AUGMENT (Build Prompt)
```python
context = "\n".join([doc.page_content for doc in results])

prompt = f"""Answer based on this context:

Context:
{context}

Question: {query}

Answer:"""
```

**The augmented prompt:**
```
┌────────────────────────────────────────────────────────────────┐
│ Answer based on this context:                                   │
│                                                                 │
│ Context:                                                        │
│ Machine learning is a subset of artificial intelligence that   │
│ enables systems to learn from data without explicit            │
│ programming. It uses algorithms to identify patterns...        │
│                                                                 │
│ Neural networks are computing systems inspired by biological   │
│ neural networks. They consist of layers of interconnected...   │
│                                                                 │
│ Question: What is machine learning?                            │
│                                                                 │
│ Answer:                                                         │
└────────────────────────────────────────────────────────────────┘
```

#### Stage 8: GENERATE (LLM Response)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
answer = llm.invoke(prompt)
```

**Final answer:** Based on the retrieved context, the LLM generates an accurate, grounded response.

---

## Component Deep Dive

### Document Loaders

| Loader | Use Case | Example |
|--------|----------|---------|
| `PyPDFLoader` | PDF files | `PyPDFLoader("doc.pdf")` |
| `TextLoader` | Plain text | `TextLoader("doc.txt")` |
| `CSVLoader` | CSV data | `CSVLoader("data.csv")` |
| `DirectoryLoader` | Multiple files | `DirectoryLoader("./docs")` |
| `WebBaseLoader` | Web pages | `WebBaseLoader("https://...")` |

### Text Splitters

| Splitter | Strategy | Best For |
|----------|----------|----------|
| `RecursiveCharacterTextSplitter` | Split by paragraphs, then sentences, then words | Most documents |
| `CharacterTextSplitter` | Split by character count | Simple text |
| `TokenTextSplitter` | Split by token count | Token-aware splitting |
| `MarkdownHeaderTextSplitter` | Split by headers | Markdown docs |
| `HTMLHeaderTextSplitter` | Split by HTML tags | Web content |

### Embedding Models

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Free |
| `all-mpnet-base-v2` | 768 | Medium | Better | Free |
| `text-embedding-3-small` | 1536 | Fast | Great | $0.02/1M tokens |
| `text-embedding-3-large` | 3072 | Medium | Best | $0.13/1M tokens |

### Vector Stores

| Store | Type | Best For | Persistence |
|-------|------|----------|-------------|
| **ChromaDB** | Local | Learning, small projects | Yes |
| **FAISS** | Local | Performance, larger data | Manual |
| **Pinecone** | Cloud | Production, scaling | Yes |
| **Weaviate** | Both | Hybrid search | Yes |
| **Qdrant** | Both | Filtering, performance | Yes |

### LLM Options

| Provider | Models | Cost | Quality |
|----------|--------|------|---------|
| **Ollama** (Local) | llama3.2, mistral, phi3 | Free | Good |
| **OpenAI** | gpt-4o, gpt-4o-mini | ~$0.01-0.03/1K | Excellent |
| **Anthropic** | claude-3.5-sonnet | ~$0.003-0.015/1K | Excellent |
| **Google** | gemini-pro | ~$0.001/1K | Very Good |

---

## Key Concepts

### Embeddings & Similarity

**Cosine Similarity:** Measures how similar two vectors are (0 to 1).

```
Vector A: [1, 0, 0]     Vector B: [0.9, 0.1, 0]

Cosine Similarity = (A · B) / (|A| × |B|)
                  = 0.9 / (1 × 0.906)
                  = 0.99 (very similar)
```

### Chunk Size Trade-offs

```
SMALL CHUNKS (100-500 chars)
├── Pros: Precise retrieval, less noise
├── Cons: May lose context
└── Best for: FAQs, definitions

MEDIUM CHUNKS (500-1500 chars)  ◄── RECOMMENDED
├── Pros: Good balance
├── Cons: None significant
└── Best for: Most use cases

LARGE CHUNKS (1500-3000 chars)
├── Pros: More context preserved
├── Cons: May include irrelevant info
└── Best for: Complex topics, summaries
```

### Retrieval Strategies

**Similarity Search:** Find most similar chunks by vector distance.

**MMR (Maximum Marginal Relevance):** Balance relevance with diversity.
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
)
```

**Hybrid Search:** Combine vector search with keyword search.

---

## Code Examples

### Minimal RAG System (50 lines)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load PDF
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Create LLM
llm = ChatOllama(model="llama3.2")

# 5. Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer based on this context:
{context}

Question: {question}
""")

# 6. Create chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Query
answer = chain.invoke("What is machine learning?")
print(answer)
```

### With Conversation Memory

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# First question
result = chain({"question": "What is machine learning?"})

# Follow-up (remembers context)
result = chain({"question": "How does it differ from deep learning?"})
```

---

## Best Practices

### 1. Chunk Size Optimization
```python
# Start with these defaults
chunk_size = 1000
chunk_overlap = 200  # 20% of chunk_size

# Adjust based on:
# - Document type (technical = smaller, narrative = larger)
# - Query type (specific = smaller, conceptual = larger)
```

### 2. Retrieval Tuning
```python
# Get more candidates, then filter
retriever = vectorstore.as_retriever(
    search_type="mmr",           # Diverse results
    search_kwargs={
        "k": 5,                  # Return 5 results
        "fetch_k": 20,           # Consider 20 candidates
        "lambda_mult": 0.7       # Balance relevance/diversity
    }
)
```

### 3. Prompt Engineering
```python
prompt = """You are a helpful assistant answering questions about {topic}.

Rules:
1. Only use information from the provided context
2. If the answer isn't in the context, say "I don't have that information"
3. Cite which document/page your answer came from
4. Be concise but complete

Context:
{context}

Question: {question}

Answer:"""
```

### 4. Evaluation Metrics

| Metric | What it Measures |
|--------|------------------|
| **Retrieval Precision** | % of retrieved docs that are relevant |
| **Retrieval Recall** | % of relevant docs that were retrieved |
| **Answer Faithfulness** | Is answer supported by context? |
| **Answer Relevance** | Does answer address the question? |

---

## Common Issues & Solutions

### Issue 1: Poor Retrieval Quality

**Symptoms:** Wrong documents retrieved, missing relevant info

**Solutions:**
```python
# 1. Adjust chunk size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # Try smaller chunks
    chunk_overlap=100
)

# 2. Use better embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 3. Retrieve more documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

### Issue 2: Hallucinations

**Symptoms:** LLM makes up information not in documents

**Solutions:**
```python
# 1. Stricter prompt
prompt = """Answer ONLY using the context below.
If the answer is not explicitly stated, say "Not found in documents."

Context: {context}
Question: {question}"""

# 2. Lower temperature
llm = ChatOllama(model="llama3.2", temperature=0)  # More deterministic

# 3. Show sources for verification
results = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score}, Source: {doc.metadata['source']}")
```

### Issue 3: Slow Performance

**Solutions:**
```python
# 1. Use FAISS instead of Chroma for large datasets
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 2. Reduce embedding dimensions
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # 384 dims vs 768+
)

# 3. Limit retrieved documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### Issue 4: Context Window Exceeded

**Symptoms:** Error about too many tokens

**Solutions:**
```python
# 1. Smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500)

# 2. Retrieve fewer documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Summarize retrieved content
from langchain.chains.summarize import load_summarize_chain
```

---

## Advanced Topics

### 1. Hybrid Search
Combine vector similarity with keyword matching:
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Keyword-based retriever
bm25_retriever = BM25Retriever.from_documents(chunks)

# Vector retriever
vector_retriever = vectorstore.as_retriever()

# Combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

### 2. Query Transformation
Improve retrieval by transforming the query:
```python
# Multi-query: Generate variations
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
# Generates: "What is ML?", "Define machine learning", "Explain ML"
```

### 3. Re-ranking
Re-order results for better relevance:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 4. Parent Document Retriever
Retrieve small chunks but return larger context:
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
)
```

---

## Quick Reference

### Minimal Setup
```bash
pip install langchain langchain-community chromadb sentence-transformers langchain-ollama
ollama pull llama3.2
```

### Essential Imports
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

### Default Parameters
```python
chunk_size = 1000
chunk_overlap = 200
embedding_model = "all-MiniLM-L6-v2"
llm_model = "llama3.2"
retrieval_k = 3
```

---

## Summary

RAG is powerful because it:
1. **Grounds** LLM responses in your actual documents
2. **Updates** easily by adding new documents
3. **Scales** from personal projects to enterprise
4. **Costs** less than fine-tuning
5. **Provides** transparency through source citations

The key to successful RAG is:
- Good chunking strategy
- Quality embeddings
- Proper retrieval tuning
- Clear prompts

Now go build something awesome!
