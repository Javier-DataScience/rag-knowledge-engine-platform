"""
Purpose:
    Validate the retrieval layer.

Architecture Tested:

    PDF
        ↓
    Loader
        ↓
    Chunking
        ↓
    Embeddings
        ↓
    FAISS
        ↓
    Retriever
"""

from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_splitter import TextSplitter
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstores.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever


def main():
    pdf_path = "data/raw/Supervised-Learning.pdf"

    loader = PDFLoader()
    documents = loader.load(pdf_path)

    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)

    embedding_model = EmbeddingModel()

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(
        chunk_texts
    )

    store = FAISSStore()

    store.add_documents(
        chunks,
        embeddings
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=store,
    )

    results = retriever.retrieve(
        "What is supervised learning?"
    )

    print("\nRetrieved Chunks:\n")

    for i, chunk in enumerate(results, start=1):
        print(f"Chunk {i}")
        print(chunk.page_content[:300])
        print("-" * 50)


if __name__ == "__main__":
    main()