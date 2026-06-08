"""
Module: rag_service.py

Purpose:
    This module defines the RAGService class, which acts as the
    high-level orchestrator of the Retrieval-Augmented Generation (RAG) system.

    It is responsible for:
    - Loading and processing documents
    - Chunking text
    - Generating embeddings
    - Building and querying the FAISS vector store
    - Performing retrieval
    - Calling the LLM (Ollama)
    - Returning final answers with source metadata

    Additionally, this version includes a production-quality improvement:
    - Deduplication of retrieved source documents (by source + page)
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
    High-level service that builds and manages the full RAG pipeline.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize the RAG service.

        Args:
            pdf_path (str): Path to the PDF document used as knowledge source.
        """
        self.pdf_path = pdf_path
        self.rag_pipeline = self._build_pipeline()

    def _build_pipeline(self) -> RAGPipeline:
        """
        Build the full RAG pipeline step by step.
        """

        # Load documents
        loader = PDFLoader()
        documents = loader.load(self.pdf_path)

        # Split into chunks
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        # Create embeddings
        embedding_model = EmbeddingModel()
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.embed_documents(chunk_texts)

        # Build vector store
        store = FAISSStore()
        store.add_documents(chunks, embeddings)

        # Retriever
        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=store,
        )

        # LLM
        llm = OllamaLLM()

        # RAG pipeline
        return RAGPipeline(
            retriever=retriever,
            llm=llm,
        )

    def ask(self, question: str) -> dict:
        """
        Ask a question to the RAG system and return a cleaned response.

        This version includes:
        - Deduplication of sources (post-processing layer)
        """

        result = self.rag_pipeline.ask(question)

        # -----------------------------
        # Deduplicate sources
        # -----------------------------
        seen = set()
        unique_sources = []

        for src in result["sources"]:
            key = (src["source"], src["page"])

            if key not in seen:
                seen.add(key)
                unique_sources.append(src)

        result["sources"] = unique_sources

        return result