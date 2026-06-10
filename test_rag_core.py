"""
TEST SCRIPT - RAG PIPELINE (PHASE A.4)

PURPOSE:
--------
This script tests the full RAG system end-to-end:

1. Loads the PDF-based retriever (FAISS + embeddings)
2. Loads the LLM wrapper (Ollama)
3. Builds the RAGPipeline with dependency injection
4. Sends a test question
5. Prints the final grounded answer

WHY THIS EXISTS:
----------------
This is a local sanity-check tool BEFORE:
- FastAPI integration
- Gradio UI
- Docker packaging

It ensures the core RAG system works independently of deployment layers.
"""

from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM
from app.rag.rag_pipeline import RAGPipeline


def main():

    # ---------------------------------------------------------
    # STEP 1: Initialize components separately (clean design)
    # ---------------------------------------------------------
    print("Initializing RAG pipeline...")

    retriever = Retriever()
    llm = OllamaLLM()

    # ---------------------------------------------------------
    # STEP 2: Inject dependencies into pipeline
    # ---------------------------------------------------------
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm
    )

    # ---------------------------------------------------------
    # STEP 3: Test query
    # ---------------------------------------------------------
    question = "What is supervised learning?"

    print("\nQuestion:", question)

    # ---------------------------------------------------------
    # STEP 4: Get answer from RAG system
    # ---------------------------------------------------------
    answer = rag.ask(question)

    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()