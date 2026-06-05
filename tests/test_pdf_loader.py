"""
Purpose:
    Validate that the PDFLoader can successfully load
    a PDF document and return LangChain Document objects.

Why this file exists:
    Before building chunking, embeddings, and retrieval,
    we must confirm that document ingestion works correctly.

How it fits into the architecture:

    PDF File
        ↓
    PDFLoader
        ↓
    Document Objects
        ↓
    Future Chunking Layer
"""

from app.ingestion.pdf_loader import PDFLoader


def main():
    pdf_path = "data/raw/Supervised-Learning.pdf"

    loader = PDFLoader()

    documents = loader.load(pdf_path)

    print(f"Number of documents loaded: {len(documents)}")

    if documents:
        print("\nFirst document preview:\n")
        print(documents[0].page_content[:500])


if __name__ == "__main__":
    main()