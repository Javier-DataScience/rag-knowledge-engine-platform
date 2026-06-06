"""
Purpose:
    End-to-end Retrieval-Augmented Generation pipeline.

Why this file exists:
    Connects retrieval and generation into a single workflow.

How it fits into the architecture:

    Question
        ↓
    Retriever
        ↓
    Relevant Chunks
        ↓
    Context Construction
        ↓
    Prompt Construction
        ↓
    Llama 3
        ↓
    Grounded Answer + Sources

Responsibilities:
    - Retrieve relevant chunks
    - Build context
    - Build grounded prompt
    - Generate final answer
    - Return source information
"""

from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM


class RAGPipeline:
    """
    End-to-end RAG workflow.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def ask(
        self,
        question: str,
    ) -> dict:
        """
        Generate a grounded answer and return sources.
        """

        documents = self.retriever.retrieve(
            question
        )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        prompt = f"""
You are a helpful AI assistant.

Use ONLY the information contained in the context below.

If the answer cannot be found in the context,
respond with:

"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm.generate(
            prompt
        )

        sources = []

        for doc in documents:
            sources.append(
                {
                    "source": doc.metadata.get(
                        "source",
                        "Unknown"
                    ),
                    "page": doc.metadata.get(
                        "page",
                        "Unknown"
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }