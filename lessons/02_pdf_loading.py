"""
=============================================================================
LESSON 2: PDF Loading and Text Extraction
=============================================================================

In this lesson, you'll learn:
- How to load PDF files using different libraries
- Extract text from PDFs
- Handle different PDF formats
- Preserve metadata

We'll cover four approaches:
1. PyMuPDF (RECOMMENDED - best text extraction quality)
2. PyPDF (simple, fast)
3. PDFPlumber (better for complex layouts with tables)
4. LangChain loaders (for integration with RAG pipeline)

Why PyMuPDF is Recommended:
---------------------------
PyMuPDF (fitz) provides superior text extraction compared to other libraries:
- Better handling of complex PDF layouts
- More accurate text spacing and formatting
- Faster processing speed
- Better Unicode support
- Direct access to page-level metadata
"""

from pathlib import Path
import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# METHOD 1: Using PyMuPDF (RECOMMENDED - Best Quality)
# =============================================================================

def load_pdf_with_pymupdf(pdf_path: str) -> dict:
    """
    Load a PDF using PyMuPDF (fitz) library.

    PyMuPDF is RECOMMENDED because:
    - Superior text extraction quality
    - Better handling of complex layouts
    - More accurate spacing and formatting
    - Faster processing speed
    - Better Unicode support

    This is the same method used in the production rag_system.py!

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with text and metadata
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)

    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": page_num + 1,
            "text": text
        })

    metadata = {
        "total_pages": len(doc),
        "title": doc.metadata.get("title"),
        "author": doc.metadata.get("author"),
        "source": pdf_path
    }

    doc.close()

    return {
        "pages": pages,
        "metadata": metadata
    }


def load_pdf_with_pymupdf_langchain(pdf_path: str):
    """
    Load a PDF using PyMuPDF and return LangChain Document objects.

    This combines PyMuPDF's superior extraction with LangChain's
    Document format for seamless integration with RAG pipelines.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of LangChain Document objects
    """
    import fitz  # PyMuPDF
    from langchain_core.documents import Document

    doc = fitz.open(pdf_path)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # Only add non-empty pages
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_num,
                    "total_pages": len(doc)
                }
            ))

    doc.close()
    return documents


# =============================================================================
# METHOD 2: Using PyPDF (Simple approach)
# =============================================================================

def load_pdf_with_pypdf(pdf_path: str) -> dict:
    """
    Load a PDF using PyPDF library.

    PyPDF is great for:
    - Simple text extraction
    - Fast processing
    - Getting metadata

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with text and metadata
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)

    # Extract text from all pages
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        pages.append({
            "page_number": i + 1,
            "text": text
        })

    # Get metadata
    metadata = {
        "total_pages": len(reader.pages),
        "title": reader.metadata.title if reader.metadata else None,
        "author": reader.metadata.author if reader.metadata else None,
        "source": pdf_path
    }

    return {
        "pages": pages,
        "metadata": metadata
    }


# =============================================================================
# METHOD 3: Using PDFPlumber (Better for complex layouts with tables)
# =============================================================================

def load_pdf_with_pdfplumber(pdf_path: str) -> dict:
    """
    Load a PDF using PDFPlumber library.

    PDFPlumber is great for:
    - Complex layouts with tables
    - Extracting tables as structured data
    - Better handling of multi-column text

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with text, tables, and metadata
    """
    import pdfplumber

    pages = []
    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()

            # Extract tables (if any)
            tables = page.extract_tables()
            if tables:
                tables_found.extend([{
                    "page": i + 1,
                    "table": table
                } for table in tables])

            pages.append({
                "page_number": i + 1,
                "text": text,
                "has_tables": len(tables) > 0
            })

        metadata = {
            "total_pages": len(pdf.pages),
            "source": pdf_path
        }

    return {
        "pages": pages,
        "tables": tables_found,
        "metadata": metadata
    }


# =============================================================================
# METHOD 4: Using LangChain's Document Loaders
# =============================================================================

def load_pdf_with_langchain(pdf_path: str):
    """
    Load a PDF using LangChain's PyPDFLoader.

    NOTE: For better text extraction quality, consider using
    load_pdf_with_pymupdf_langchain() instead!

    LangChain's PyPDFLoader is convenient because:
    - Returns Document objects ready for processing
    - Includes metadata automatically
    - Integrates seamlessly with LangChain pipeline

    However, it uses PyPDF under the hood, which may not extract
    text as accurately as PyMuPDF for complex PDFs.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of LangChain Document objects
    """
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Each document has:
    # - page_content: The text content
    # - metadata: Dict with source, page number, etc.

    return documents


def load_multiple_pdfs(pdf_directory: str):
    """
    Load all PDFs from a directory.

    Args:
        pdf_directory: Path to directory containing PDFs

    Returns:
        List of all Document objects from all PDFs
    """
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",  # Match all PDF files
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    return documents


# =============================================================================
# DEMONSTRATION
# =============================================================================

def create_sample_pdf():
    """Create a sample PDF for testing (if none exists)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    sample_path = Path(__file__).parent.parent / "data" / "sample.pdf"
    sample_path.parent.mkdir(exist_ok=True)

    if sample_path.exists():
        return str(sample_path)

    c = canvas.Canvas(str(sample_path), pagesize=letter)

    # Page 1
    c.drawString(100, 750, "RAG Learning Project - Sample Document")
    c.drawString(100, 700, "Chapter 1: Introduction to Machine Learning")
    c.drawString(100, 650, "Machine learning is a subset of artificial intelligence")
    c.drawString(100, 620, "that enables systems to learn from data.")
    c.drawString(100, 570, "Key concepts include:")
    c.drawString(120, 540, "- Supervised Learning")
    c.drawString(120, 510, "- Unsupervised Learning")
    c.drawString(120, 480, "- Reinforcement Learning")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Chapter 2: Neural Networks")
    c.drawString(100, 700, "Neural networks are computing systems inspired by")
    c.drawString(100, 670, "biological neural networks in the brain.")
    c.drawString(100, 620, "A typical neural network consists of:")
    c.drawString(120, 590, "- Input Layer: Receives the input data")
    c.drawString(120, 560, "- Hidden Layers: Process the data")
    c.drawString(120, 530, "- Output Layer: Produces the final result")
    c.showPage()

    c.save()
    return str(sample_path)


def demonstrate_pdf_loading():
    """Demonstrate different PDF loading methods."""
    print("=" * 60)
    print("LESSON 2: PDF Loading Demonstration")
    print("=" * 60)

    # First check for existing PDFs in data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    existing_pdfs = list(data_dir.glob("*.pdf"))

    if existing_pdfs:
        sample_path = str(existing_pdfs[0])
        print(f"\n[OK] Found existing PDF: {existing_pdfs[0].name}")
    else:
        # Try to create sample PDF (requires reportlab)
        try:
            sample_path = create_sample_pdf()
            print(f"\n[OK] Sample PDF created at: {sample_path}")
        except ImportError:
            print("\n[!] reportlab not installed. No PDFs available.")
            print("    Either add a PDF to the data/ folder")
            print("    Or install reportlab: pip install reportlab")
            return

    # Check if we have a real PDF to test with
    if not Path(sample_path).exists():
        print(f"\n[!] No PDF found at {sample_path}")
        print("    Please add a PDF file to the data/ directory")
        return

    print("\n" + "-" * 40)
    print("Method 1: PyMuPDF (RECOMMENDED)")
    print("-" * 40)

    try:
        result = load_pdf_with_pymupdf(sample_path)
        print(f"Pages loaded: {result['metadata']['total_pages']}")
        print(f"First page preview: {result['pages'][0]['text'][:200]}...")
        print("\n[*] This is the RECOMMENDED method for best text quality!")
    except ImportError:
        print("PyMuPDF not installed. Run: pip install pymupdf")

    print("\n" + "-" * 40)
    print("Method 1b: PyMuPDF + LangChain Documents")
    print("-" * 40)

    try:
        documents = load_pdf_with_pymupdf_langchain(sample_path)
        print(f"Documents loaded: {len(documents)}")
        if documents:
            print(f"First page preview: {documents[0].page_content[:200]}...")
        print("\n[*] Best of both worlds: PyMuPDF quality + LangChain format!")
    except ImportError as e:
        print(f"Missing dependencies: {e}")

    print("\n" + "-" * 40)
    print("Method 2: PyPDF")
    print("-" * 40)

    try:
        result = load_pdf_with_pypdf(sample_path)
        print(f"Pages loaded: {result['metadata']['total_pages']}")
        print(f"First page preview: {result['pages'][0]['text'][:200]}...")
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")

    print("\n" + "-" * 40)
    print("Method 3: PDFPlumber")
    print("-" * 40)

    try:
        result = load_pdf_with_pdfplumber(sample_path)
        print(f"Pages loaded: {result['metadata']['total_pages']}")
        print(f"Tables found: {len(result['tables'])}")
    except ImportError:
        print("pdfplumber not installed. Run: pip install pdfplumber")

    print("\n" + "-" * 40)
    print("Method 4: LangChain PyPDFLoader")
    print("-" * 40)

    try:
        documents = load_pdf_with_langchain(sample_path)
        print(f"Documents loaded: {len(documents)}")
        for doc in documents[:2]:  # Show first 2 pages
            print(f"\nPage {doc.metadata.get('page', 'N/A')}:")
            print(f"  Content preview: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
    except ImportError as e:
        print(f"LangChain not installed. Run: pip install langchain langchain-community")
        print(f"Error: {e}")


if __name__ == "__main__":
    demonstrate_pdf_loading()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. PyMuPDF (RECOMMENDED): Best text extraction quality
       - Used in the production rag_system.py
       - pip install pymupdf

    2. PyPDF: Simple and fast, but may have spacing issues

    3. PDFPlumber: Best for documents with tables

    4. LangChain Loaders: Convenient but uses PyPDF internally

    For RAG projects, we RECOMMEND:
    - Use PyMuPDF + LangChain Document format (load_pdf_with_pymupdf_langchain)
    - This gives you the best extraction quality + pipeline compatibility

    Quick Code for Your RAG:
    ------------------------
    import fitz  # PyMuPDF
    from langchain_core.documents import Document

    doc = fitz.open("your.pdf")
    documents = [
        Document(page_content=page.get_text(), metadata={"page": i})
        for i, page in enumerate(doc)
    ]
    doc.close()

    NEXT: Lesson 3 - Text Chunking
    """)
