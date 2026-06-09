"""
Module: main.py

Purpose:
    FastAPI entry point for the RAG Knowledge Engine Platform.

    Adds:
    - /ask endpoint (RAG Q&A)
    - /upload endpoint (PDF ingestion via Streamlit)
"""

from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.api.schemas import AskRequest, AskResponse
from app.api.rag_service import RAGService

app = FastAPI(
    title="RAG Knowledge Engine Platform",
    version="1.0.0",
)

# -----------------------------
# RAG Service (default PDF)
# -----------------------------
rag_service = RAGService(
    pdf_path="data/raw/Supervised-Learning.pdf"
)


@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    return rag_service.ask(request.question)


# -----------------------------
# NEW: PDF Upload Endpoint
# -----------------------------
UPLOAD_DIR = "data/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    Upload PDF and rebuild RAG system.
    Phase A: synchronous rebuild (simplified).
    """

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Return early acknowledgment (IMPORTANT CHANGE)
    global rag_service

    rag_service = RAGService(pdf_path=file_path)

    return {
        "message": "PDF uploaded and RAG system rebuilt successfully",
        "file": file.filename
    }