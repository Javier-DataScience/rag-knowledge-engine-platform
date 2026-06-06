"""
Purpose:
    Validate the complete RAG workflow.

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
        ↓
    RAG Pipeline
        ↓
    Llama 3
"""

from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_splitter import TextSplitter
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstores.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM
from app.rag.rag_pipeline import RAGPipeline


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

    llm = OllamaLLM()

    rag = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    result = rag.ask(
        "What is supervised machine learning?"
    )

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()