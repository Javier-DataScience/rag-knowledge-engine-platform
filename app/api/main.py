"""
Module: main.py (FastAPI API Layer)

Purpose:
--------
This module exposes the RAG system as a REST API.

It acts as a thin orchestration layer between:
    - Streamlit UI
    - Gradio UI (future)
    - RAGPipeline (core logic)
    - Retriever + LLM backend

Endpoints:
----------
1. POST /ask
    - Receives a question
    - Returns answer + structured citations

2. POST /upload
    - Receives a PDF file
    - Stores it locally
    - Rebuilds RAG pipeline with new document

IMPORTANT DESIGN RULES:
-----------------------
- NO embedding logic here
- NO chunking logic here
- NO FAISS logic here
- NO LLM logic here

This file ONLY orchestrates existing components.

Phase A.4:
----------
Adds structured citation support.

API response format:

{
    "question": "...",
    "answer": "...",
    "sources": [
        {
            "source": "document.pdf",
            "page": 5
        }
    ]
}
"""

from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.retrieval.retriever import Retriever
from app.llms.ollama_llm import OllamaLLM
from app.rag.rag_pipeline import RAGPipeline


# --------------------------------------------------
# App initialization
# --------------------------------------------------

app = FastAPI(
    title="RAG Knowledge Engine API",
    version="1.0"
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Initializing RAG system...")

llm = OllamaLLM()

# Default pipeline
rag = RAGPipeline(
    retriever=Retriever(),
    llm=llm
)


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "RAG API is running"
    }


# --------------------------------------------------
# Question Answering Endpoint
# --------------------------------------------------

@app.post("/ask")
def ask_question(payload: dict):

    question = payload.get("question", "")

    result = rag.ask(question)

    return {
        "question": question,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])
    }


# --------------------------------------------------
# PDF Upload + Reindexing Endpoint
# --------------------------------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    # Save uploaded PDF

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    print("PDF received:", file.filename)
    print("Rebuilding RAG pipeline...")

    global rag

    rag = RAGPipeline(
        retriever=Retriever(pdf_path=file_path),
        llm=llm
    )

    return {
        "status": "ok",
        "message": "PDF indexed successfully",
        "file": file.filename
    }