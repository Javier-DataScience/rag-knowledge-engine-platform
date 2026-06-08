"""
Purpose:
    Validate the RAG service layer.

Architecture Tested:

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

from app.api.rag_service import RAGService


def main():
    service = RAGService(
        pdf_path="data/raw/Supervised-Learning.pdf"
    )

    result = service.ask(
        "What is supervised machine learning?"
    )

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()