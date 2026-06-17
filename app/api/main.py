"""
Module: main.py (FastAPI API Layer)

Purpose:
--------
REST API layer for RAG Knowledge Engine.

Key improvements:
-----------------
✔ Docker-safe file handling
✔ Robust upload directory creation
✔ Safer payload validation
✔ Prevents FileNotFoundError crashes
✔ Stable Streamlit + Gradio compatibility
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Safe upload directory (Docker-safe)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Initializing RAG system...")

llm = OllamaLLM()

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

    question = payload.get("question")

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = rag.ask(question)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG processing failed: {str(e)}"
        )


# --------------------------------------------------
# PDF Upload + Reindexing Endpoint
# --------------------------------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    try:
        # Ensure upload directory exists (CRITICAL FIX)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        # IMPORTANT FIX: ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save file safely
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )