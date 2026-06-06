"""
Purpose:
    Embedding model wrapper for the RAG Knowledge Engine Platform.

Why this file exists:
    Embeddings convert text into numerical vectors that
    can be stored and searched using a vector database.

How it fits into the architecture:

    Text Chunks
        ↓
    EmbeddingModel
        ↓
    Vectors
        ↓
    FAISS (next phase)

Responsibilities:
    - Load embedding model
    - Convert documents into vectors
    - Convert queries into vectors

Non-responsibilities:
    - Vector storage
    - Retrieval
    - LLM generation
"""

from sentence_transformers import SentenceTransformer

from app.config.settings import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Wrapper around a Hugging Face embedding model.
    """

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    def embed_documents(
        self,
        texts: list[str]
    ):
        """
        Generate embeddings for document chunks.
        """

        return self.model.encode(
            texts,
            convert_to_numpy=True
        )

    def embed_query(
        self,
        query: str
    ):
        """
        Generate embedding for a user query.
        """

        return self.model.encode(
            query,
            convert_to_numpy=True
        )