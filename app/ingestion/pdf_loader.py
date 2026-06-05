"""
Purpose:
    PDF document loader for the RAG Knowledge Engine Platform.

Why this file exists:
    The ingestion layer is responsible for loading documents
    from external sources and converting them into a format
    that can be processed by the rest of the RAG pipeline.

How it fits into the architecture:
    PDF File
        ↓
    PDFLoader
        ↓
    LangChain Document Objects
        ↓
    Chunking Layer

Responsibilities:
    - Load PDF files
    - Return LangChain Document objects
    - Preserve document metadata

Non-responsibilities:
    - Chunking
    - Embeddings
    - Vector storage
    - Retrieval
    - LLM generation
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Loads PDF files and returns LangChain Document objects.
    """

    def load(self, pdf_path: str) -> list[Document]:
        """
        Load a PDF file.

        Args:
            pdf_path:
                Path to the PDF file.

        Returns:
            List of LangChain Document objects.

        Raises:
            FileNotFoundError:
                If the PDF file does not exist.
        """

        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_file}"
            )

        loader = PyPDFLoader(str(pdf_file))

        documents = loader.load()

        return documents