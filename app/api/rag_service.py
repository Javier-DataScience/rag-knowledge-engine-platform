"""
Purpose:
    Centralized RAG service for FastAPI.

Why this file exists:
    Creates and stores the RAG pipeline
    so it can be reused across requests.

Architecture:

    FastAPI
        ↓
    RAGService
        ↓
    RAGPipeline
        ↓
    Retriever
        ↓
    FAISS
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


class RAGService:
    """
    Creates and manages the RAG pipeline.
    """

    def __init__(
        self,
        pdf_path: str,
    ):
        self.pdf_path = pdf_path

        self.rag_pipeline = self._build_pipeline()

    def _build_pipeline(
        self,
    ) -> RAGPipeline:
        """
        Build the complete RAG pipeline.
        """

        loader = PDFLoader()
        documents = loader.load(
            self.pdf_path
        )

        splitter = TextSplitter()

        chunks = splitter.split_documents(
            documents
        )

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
            embeddings,
        )

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=store,
        )

        llm = OllamaLLM()

        return RAGPipeline(
            retriever=retriever,
            llm=llm,
        )

    def ask(
        self,
        question: str,
    ) -> dict:
        """
        Ask a question using the
        initialized RAG pipeline.
        """

        return self.rag_pipeline.ask(
            question
        )