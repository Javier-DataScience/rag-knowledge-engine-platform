"""
Purpose:
    Text chunking module for the RAG Knowledge Engine Platform.

Why this file exists:
    Large documents are difficult to search effectively.
    Chunking breaks documents into smaller pieces that
    can later be converted into embeddings and stored
    in a vector database.

How it fits into the architecture:

    PDF/TXT
        ↓
    Ingestion Layer
        ↓
    Document Objects
        ↓
    TextSplitter
        ↓
    Chunks
        ↓
    Embeddings Layer

Chunking Strategy:
    - Recursive Character Splitting
    - Chunk Size: configured in settings.py
    - Chunk Overlap: configured in settings.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class TextSplitter:
    """
    Splits documents into smaller overlapping chunks.
    """

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents:
                List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """

        chunks = self.splitter.split_documents(documents)

        return chunks