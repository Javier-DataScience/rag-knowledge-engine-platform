"""
Purpose:
    Validate FAISS storage and retrieval.

Architecture Tested:

    PDF
        ↓
    Loader
        ↓
    Documents
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    FAISS
        ↓
    Search
"""

from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_splitter import TextSplitter
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstores.faiss_store import FAISSStore


def main():
    pdf_path = "data/raw/Supervised-Learning.pdf"

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)

    model = EmbeddingModel()

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = model.embed_documents(
        chunk_texts
    )

    store = FAISSStore()

    store.add_documents(
        chunks,
        embeddings
    )

    query = "What is supervised learning?"

    query_embedding = model.embed_query(
        query
    )

    results = store.search(
        query_embedding,
        k=3
    )

    print("\nTop Results:\n")

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(result.page_content[:300])
        print("-" * 50)


if __name__ == "__main__":
    main()