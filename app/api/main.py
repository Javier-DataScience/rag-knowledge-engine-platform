"""
Purpose:
    Entry point for the FastAPI application.

Why this file exists:
    Creates and exposes the FastAPI app instance.

Current Scope:
    - Health endpoint
    - RAG question-answering endpoint
"""

from fastapi import FastAPI

from app.api.schemas import (
    AskRequest,
    AskResponse,
)
from app.api.rag_service import RAGService

app = FastAPI(
    title="RAG Knowledge Engine Platform",
    version="1.0.0",
)

# Build the RAG system once when the server starts
rag_service = RAGService(
    pdf_path="data/raw/Supervised-Learning.pdf"
)


@app.get("/")
def root():
    """
    Health endpoint.
    """

    return {
        "message": "RAG Knowledge Engine Platform API is running"
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
):
    """
    Ask a question to the RAG system.
    """

    result = rag_service.ask(
        request.question
    )

    return result