"""
Purpose:
    FAISS vector store wrapper for the RAG Knowledge Engine Platform.

Why this file exists:
    FAISS stores embeddings and performs similarity search.

How it fits into the architecture:

    Chunks
        ↓
    EmbeddingModel
        ↓
    Vectors
        ↓
    FAISSStore
        ↓
    Similarity Search

Responsibilities:
    - Store document embeddings
    - Perform similarity search
    - Return matching documents

Non-responsibilities:
    - PDF loading
    - Chunking
    - LLM generation
"""

import faiss
import numpy as np

from langchain_core.documents import Document


class FAISSStore:
    """
    Simple FAISS vector store wrapper.
    """

    def __init__(self):
        self.index = None
        self.documents = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and embeddings to FAISS.
        """

        embedding_dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            embedding_dimension
        )

        self.index.add(
            embeddings.astype("float32")
        )

        self.documents = documents

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> list[Document]:
        """
        Return top-k most similar documents.
        """

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:
            results.append(
                self.documents[idx]
            )

        return results