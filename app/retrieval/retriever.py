"""
Retriever - Phase A.4 (Source Citations / Page Tracking)

Adds:
- page-level metadata
- chunk provenance tracking
- citation-ready retrieval output
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


class Retriever:
    def __init__(self, pdf_path=None):

        self.model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "BAAI/bge-small-en-v1.5"
        )

        self.model = SentenceTransformer(self.model_name)

        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)

        # store structured chunks with metadata
        self.documents = []

        self.pdf_path = pdf_path or "data/raw/Supervised-Learning.pdf"

        self._build_index()

    # =========================================================
    # PDF LOADER WITH PAGE TRACKING
    # =========================================================
    def _load_pdf(self):

        reader = PdfReader(self.pdf_path)

        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()

            if text:
                pages.append({
                    "page": i + 1,
                    "text": text
                })

        return pages

    # =========================================================
    # TOKEN-AWARE CHUNKING + PAGE TRACKING
    # =========================================================
    def _chunk_text(self, pages, max_tokens=256, overlap_sentences=2):

        tokenizer = self.model.tokenizer

        chunks = []

        for page in pages:

            sentences = page["text"].split(". ")

            current_chunk = []
            current_tokens = 0

            for sentence in sentences:

                sentence_tokens = tokenizer.encode(
                    sentence,
                    add_special_tokens=False
                )

                sentence_len = len(sentence_tokens)

                if current_tokens + sentence_len > max_tokens:

                    if current_chunk:

                        chunks.append({
                            "text": " ".join(current_chunk),
                            "page": page["page"]
                        })

                    current_chunk = current_chunk[-overlap_sentences:]
                    current_tokens = sum(
                        len(tokenizer.encode(s, add_special_tokens=False))
                        for s in current_chunk
                    )

                current_chunk.append(sentence)
                current_tokens += sentence_len

            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "page": page["page"]
                })

        return chunks

    # =========================================================
    # BUILD INDEX
    # =========================================================
    def _build_index(self):

        print("Loading PDF...")

        pages = self._load_pdf()

        print("Chunking with page tracking...")

        self.documents = self._chunk_text(pages)

        texts = [d["text"] for d in self.documents]

        print(f"Chunks created: {len(texts)}")

        print("Generating embeddings...")

        embeddings = self.model.encode(texts)

        embeddings = np.array(embeddings).astype("float32")

        print("Building FAISS index...")

        self.index.add(embeddings)

        print("Retriever ready.")

    # =========================================================
    # SEARCH WITH CITATIONS
    # =========================================================
    def search(self, query: str, top_k: int = 3):

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i in indices[0]:
            doc = self.documents[i]

            results.append({
                "text": doc["text"],
                "page": doc["page"]
            })

        return results