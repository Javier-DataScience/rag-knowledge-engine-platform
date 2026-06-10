"""
RAG CONTRACT TEST (FREEZE LAYER)

PURPOSE:
--------
This test ensures that the RAG system remains stable before:
- FastAPI integration
- Gradio UI
- Docker packaging

If this test passes → core system is SAFE to deploy.

If this test fails → DO NOT proceed to deployment.
"""

from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM
from app.rag.rag_pipeline import RAGPipeline


def test_basic_rag_flow():

    print("\n[CONTRACT TEST] Initializing components...")

    retriever = Retriever()
    llm = OllamaLLM()

    rag = RAGPipeline(retriever=retriever, llm=llm)

    question = "What is supervised learning?"

    print("\n[CONTRACT TEST] Running query...")

    answer = rag.ask(question)

    print("\n[CONTRACT TEST] Answer received:\n")
    print(answer)

    # -----------------------------
    # VALIDATION RULES (IMPORTANT)
    # -----------------------------
    assert answer is not None, "Answer is None"
    assert len(answer) > 20, "Answer too short"

    print("\n[CONTRACT TEST] PASSED ✔ SYSTEM IS STABLE")


if __name__ == "__main__":
    test_basic_rag_flow()