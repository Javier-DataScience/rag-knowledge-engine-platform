"""
Module: rag_service.py

Purpose:
    Core RAG orchestration layer.

Now includes:
- structured logging for pipeline steps
"""

from app.ingestion.pdf_loader import PDFLoader
from app.chunking.text_splitter import TextSplitter
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstores.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM
from app.rag.rag_pipeline import RAGPipeline
from app.core.logger import get_logger


class RAGService:

    def __init__(self, pdf_path: str):

        self.logger = get_logger("rag_service")
        self.pdf_path = pdf_path

        self.logger.info(f"Initializing RAG pipeline with: {pdf_path}")

        self.rag_pipeline = self._build_pipeline()

    def _build_pipeline(self):

        self.logger.info("Loading PDF documents")
        loader = PDFLoader()
        documents = loader.load(self.pdf_path)

        self.logger.info(f"Documents loaded: {len(documents)}")

        self.logger.info("Splitting documents into chunks")
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        self.logger.info(f"Chunks created: {len(chunks)}")

        self.logger.info("Generating embeddings")
        embedding_model = EmbeddingModel()

        chunk_texts = [c.page_content for c in chunks]
        embeddings = embedding_model.embed_documents(chunk_texts)

        self.logger.info("Embeddings generated")

        self.logger.info("Building FAISS index")
        store = FAISSStore()
        store.add_documents(chunks, embeddings)

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=store,
        )

        llm = OllamaLLM()

        self.logger.info("RAG pipeline ready")

        return RAGPipeline(
            retriever=retriever,
            llm=llm,
        )

    def ask(self, question: str):

        self.logger.info(f"Retrieving answer for question: {question}")

        result = self.rag_pipeline.ask(question)

        self.logger.info("Answer generated")

        return result