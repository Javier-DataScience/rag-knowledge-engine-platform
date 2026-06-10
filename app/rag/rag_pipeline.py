"""
Module: rag_pipeline.py

Purpose:
--------
Core RAG orchestration layer.

Responsibilities:
- Retrieve relevant chunks from retriever
- Build context for the LLM
- Generate final answer
- Preserve retrieval metadata for citations

Outputs:
--------
Returns:
{
    "answer": str,
    "sources": list
}

This allows:
- FastAPI to expose structured citations
- Streamlit to display sources
- Gradio to reuse the same API contract

Phase:
------
Phase A.4 - Citation-ready RAG pipeline
"""

from typing import Dict, List


class RAGPipeline:

    def __init__(self, retriever, llm):
        """
        Parameters
        ----------
        retriever:
            FAISS retriever returning structured results.

        llm:
            LLM wrapper (Ollama or future models).
        """

        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str) -> Dict:

        # --------------------------------------------------
        # STEP 1: Retrieve documents
        # --------------------------------------------------

        results = self.retriever.search(question)

        # --------------------------------------------------
        # STEP 2: Build context
        # --------------------------------------------------

        context_texts: List[str] = []

        for item in results:

            if isinstance(item, dict) and "text" in item:
                context_texts.append(item["text"])

            elif isinstance(item, str):
                context_texts.append(item)

        context = "\n".join(context_texts)

        # --------------------------------------------------
        # STEP 3: Build prompt
        # --------------------------------------------------

        prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Instructions:
- Answer only using the provided context.
- Be clear and concise.
- If context is insufficient, say so.

Answer:
"""

        # --------------------------------------------------
        # STEP 4: Generate answer
        # --------------------------------------------------

        answer = self.llm.generate(prompt)

        # --------------------------------------------------
        # STEP 5: Build citations
        # --------------------------------------------------

        sources = []

        seen_pages = set()

        for item in results:

            if not isinstance(item, dict):
                continue

            page = item.get("page")

            if page in seen_pages:
                continue

            seen_pages.add(page)

            sources.append({
                "source": self.retriever.pdf_path.split("\\")[-1],
                "page": page
            })

        # --------------------------------------------------
        # STEP 6: Return structured response
        # --------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }