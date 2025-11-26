"""
=============================================================================
RAG System - Complete Implementation
=============================================================================

This is your production-ready RAG system that you can use with your PDF data.
Just point it to your PDFs and start querying!

Usage:
    python src/rag_system.py --pdf-dir ./data --query "Your question here"

    # Or interactive mode:
    python src/rag_system.py --pdf-dir ./data --interactive
"""

import argparse
import re
import sys
import io
from pathlib import Path
from typing import List, Optional
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



def normalize_spaced_text(text: str) -> str:
    """
    Fix text extracted from PDFs with character-level spacing.
    Converts 'T M  M O N I R U Z Z A M A N' to 'TM MONIRUZZAMAN'
    """
    lines = text.split('\n')
    normalized_lines = []

    for line in lines:
        words = line.split()
        if words:
            single_char_count = sum(1 for w in words if len(w) == 1)
            ratio = single_char_count / len(words)

            # If more than 60% are single chars, it's likely spaced-out text
            if ratio > 0.6 and len(words) > 3:
                # Remove single spaces between single chars
                normalized = re.sub(r'(?<=[A-Za-z0-9]) (?=[A-Za-z0-9](?:\s|$))', '', line)
                normalized = re.sub(r'\s{2,}', ' ', normalized)
                normalized_lines.append(normalized.strip())
            else:
                normalized_lines.append(line)
        else:
            normalized_lines.append(line)

    return '\n'.join(normalized_lines)


class RAGSystem:
    """
    Production-ready RAG system for querying PDF documents.
    """

    def __init__(
        self,
        pdf_directory: str,
        persist_directory: str = "./vector_db_v2",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        normalize_text: bool = True,
    ):
        self.pdf_directory = Path(pdf_directory)
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize_text = normalize_text

        self._vectorstore = None
        self._retriever = None
        self._llm = None
        self._chain = None

    def _load_documents(self) -> List:
        """Load PDF documents using PyMuPDF for better text extraction."""
        import fitz  # PyMuPDF - much better extraction than PyPDF
        from langchain_core.documents import Document

        if not self.pdf_directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.pdf_directory}")

        documents = []
        pdf_files = list(self.pdf_directory.glob("**/*.pdf"))

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
            except Exception as e:
                print(f"Error loading {pdf_path}: {e}")

        return documents

    def _split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        return splitter.split_documents(documents)

    def _get_embeddings(self):
        """Get embedding model."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _create_vectorstore(self, chunks: List):
        """Create vector store from chunks."""
        from langchain_community.vectorstores import Chroma

        embeddings = self._get_embeddings()

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(self.persist_directory)
        )

        return self._vectorstore

    def _load_vectorstore(self):
        """Load existing vector store."""
        from langchain_community.vectorstores import Chroma

        embeddings = self._get_embeddings()

        self._vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=embeddings
        )

        return self._vectorstore

    def build_index(self, force_rebuild: bool = False):
        """
        Build or load the vector index.

        Args:
            force_rebuild: If True, rebuild even if index exists
        """
        print("=" * 50)
        print("Building RAG Index")
        print("=" * 50)

        # Check if we can load existing index
        if self.persist_directory.exists() and not force_rebuild:
            print(f"Loading existing index from {self.persist_directory}")
            self._load_vectorstore()
            print("Index loaded successfully!")
            return

        # Build new index
        print(f"Loading PDFs from {self.pdf_directory}...")
        documents = self._load_documents()
        print(f"Loaded {len(documents)} pages")

        print("Splitting into chunks...")
        chunks = self._split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Creating embeddings and vector store...")
        self._create_vectorstore(chunks)
        print(f"Index saved to {self.persist_directory}")

        print("Index built successfully!")

    def _get_retriever(self, k: int = 6):
        """Get retriever from vector store with query expansion."""
        if self._vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_index() first.")

        # Use similarity search with expanded queries for better recall
        base_retriever = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 30}
        )

        # Wrap with query expansion for temporal queries
        class ExpandedRetriever:
            def __init__(self, retriever, vectorstore, k):
                self.retriever = retriever
                self.vectorstore = vectorstore
                self.k = k

            def invoke(self, query):
                # Expand temporal queries
                expanded_terms = []
                query_lower = query.lower()

                if any(term in query_lower for term in ['last 5 years', 'past 5 years', 'recent', '5 years']):
                    expanded_terms.extend(['work experience', 'employment', '2020', '2021', '2022', '2023', '2024', '2025'])
                if any(term in query_lower for term in ['work', 'job', 'career', 'did', 'do']):
                    expanded_terms.extend(['work experience', 'employment', 'company', 'role', 'position'])

                # Get results from original query
                results = list(self.retriever.invoke(query))

                # If we have expanded terms, also search for those
                if expanded_terms:
                    expanded_query = query + " " + " ".join(expanded_terms)
                    extra_results = self.vectorstore.similarity_search(expanded_query, k=self.k)
                    # Add unique results
                    seen = {doc.page_content[:100] for doc in results}
                    for doc in extra_results:
                        if doc.page_content[:100] not in seen:
                            results.append(doc)
                            seen.add(doc.page_content[:100])

                return results[:self.k * 2]  # Return more context

        self._retriever = ExpandedRetriever(base_retriever, self._vectorstore, k)
        return self._retriever

    def _get_llm(self, llm_type: str = "ollama", model: str = "llama3.2:3b", base_url: str = None):
        """Get LLM.

        Args:
            llm_type: Type of LLM ("ollama", "openai", "anthropic")
            model: Model name
            base_url: Custom Ollama server URL (e.g., "http://192.168.178.124:11434")
        """
        if llm_type == "ollama":
            from langchain_ollama import ChatOllama

            # Use custom base_url if provided, otherwise default to localhost
            ollama_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self._llm = ChatOllama(
                model=model,
                temperature=0,
                base_url=ollama_url
            )
            print(f"Using Ollama at: {ollama_url}")

        elif llm_type == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model=model, temperature=0)

        elif llm_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(model=model, temperature=0)

        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

        return self._llm

    def _create_chain(self):
        """Create the RAG chain."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template("""You are a helpful assistant answering questions about a person's resume/CV.
Use ONLY the information from the context below to answer. Be specific and include dates, company names, and details.

Context from resume:
{context}

Question: {question}

Instructions:
- Extract specific information from the context (company names, dates, job titles, achievements)
- If asking about recent years or time periods, look for date ranges in the work experience
- Include relevant details like technologies used, responsibilities, and accomplishments
- If the information is not in the context, say "I don't have that specific information in the resume."

Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def retrieve_and_format(question):
            docs = self._retriever.invoke(question)
            return format_docs(docs)

        from langchain_core.runnables import RunnableLambda
        self._chain = (
            {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

        return self._chain

    def setup(self, llm_type: str = "ollama", model: str = "llama3.2:3b", k: int = 4, ollama_host: str = None):
        """
        Set up the complete RAG system.

        Args:
            llm_type: Type of LLM ("ollama", "openai", "anthropic")
            model: Model name
            k: Number of documents to retrieve
            ollama_host: Custom Ollama server URL (e.g., "http://192.168.178.124:11434")
        """
        print(f"Setting up RAG with {llm_type}/{model}...")

        self._get_retriever(k=k)
        self._get_llm(llm_type=llm_type, model=model, base_url=ollama_host)
        self._create_chain()

        print("RAG system ready!")

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: The question to ask

        Returns:
            The answer from the system
        """
        if self._chain is None:
            raise ValueError("Chain not initialized. Call setup() first.")

        return self._chain.invoke(question)

    def query_with_sources(self, question: str) -> dict:
        """
        Query and return answer with source documents.

        Args:
            question: The question to ask

        Returns:
            Dict with 'answer' and 'sources'
        """
        answer = self.query(question)
        sources = self._retriever.invoke(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc in sources
            ]
        }

    def interactive(self):
        """Start interactive query session."""
        print("\n" + "=" * 50)
        print("Interactive RAG Session")
        print("Type 'quit' to exit, 'sources' to toggle source display")
        print("=" * 50)

        show_sources = False

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if question.lower() == 'sources':
                    show_sources = not show_sources
                    print(f"Sources: {'ON' if show_sources else 'OFF'}")
                    continue

                print("\nSearching...")

                if show_sources:
                    result = self.query_with_sources(question)
                    print(f"\nAnswer: {result['answer']}")
                    print("\nSources:")
                    for i, src in enumerate(result['sources'], 1):
                        print(f"  {i}. {Path(src['source']).name} (p.{src['page']})")
                else:
                    answer = self.query(question)
                    print(f"\nAnswer: {answer}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG System for PDF Q&A")

    parser.add_argument(
        "--pdf-dir",
        type=str,
        required=True,
        help="Directory containing PDF files"
    )

    parser.add_argument(
        "--db-dir",
        type=str,
        default="./vector_db_v2",
        help="Directory to store vector database"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask (single query mode)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild the index"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="LLM provider"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Model name"
    )

    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server URL (e.g., http://192.168.178.124:11434)"
    )

    args = parser.parse_args()

    # Create RAG system
    rag = RAGSystem(
        pdf_directory=args.pdf_dir,
        persist_directory=args.db_dir
    )

    # Build index
    rag.build_index(force_rebuild=args.rebuild)

    # Set up the chain
    rag.setup(llm_type=args.llm, model=args.model, ollama_host=args.ollama_host)

    # Run query or interactive mode
    if args.interactive:
        rag.interactive()
    elif args.query:
        result = rag.query_with_sources(args.query)
        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {Path(src['source']).name} (page {src['page']})")
    else:
        print("\nUse --query 'your question' or --interactive")
        print("Example: python rag_system.py --pdf-dir ./data --query 'What is ML?'")


if __name__ == "__main__":
    main()
