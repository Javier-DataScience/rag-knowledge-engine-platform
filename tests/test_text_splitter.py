"""
Purpose:
    Validate the chunking layer.

Why this file exists:
    We want to confirm that documents loaded from a PDF
    can be successfully split into smaller chunks.

How it fits into the architecture:

    PDF
        ↓
    PDFLoader
        ↓
    Document Objects
        ↓
    TextSplitter
        ↓
    Chunks
"""

from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_splitter import TextSplitter


def main():
    pdf_path = "data/raw/Supervised-Learning.pdf"

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)

    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:\n")
        print(chunks[0].page_content[:500])

        print("\nChunk metadata:\n")
        print(chunks[0].metadata)


if __name__ == "__main__":
    main()