"""
Purpose:
    Retrieval layer for the RAG Knowledge Engine Platform.

Why this file exists:
    The retriever is responsible for finding the most
    relevant document chunks for a user question.

How it fits into the architecture:

    User Question
            ↓
      Retriever
            ↓
    Query Embedding
            ↓
         FAISS
            ↓
    Relevant Chunks

Responsibilities:
    - Receive a user question
    - Generate a query embedding
    - Search the vector store
    - Return relevant chunks

Non-responsibilities:
    - Prompt construction
    - LLM generation
    - API endpoints
"""

from langchain_core.documents import Document

from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstores.faiss_store import FAISSStore
from app.config.settings import TOP_K_RESULTS


class Retriever:
    """
    Retrieves relevant chunks for a query.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: FAISSStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> list[Document]:
        """
        Retrieve top-k relevant chunks.
        """

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
        )

        return results