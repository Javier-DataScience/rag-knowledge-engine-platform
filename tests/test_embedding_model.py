"""
Purpose:
    Validate embedding generation.

Why this file exists:
    Confirm that the embedding model loads correctly
    and produces vectors.
"""

from app.embeddings.embedding_model import EmbeddingModel


def main():
    model = EmbeddingModel()

    text = "What is supervised machine learning?"

    embedding = model.embed_query(text)

    print("Embedding shape:")
    print(embedding.shape)

    print("\nFirst 10 values:")
    print(embedding[:10])


if __name__ == "__main__":
    main()