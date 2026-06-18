"""
Module: main.py (FastAPI API Layer)

Purpose:
--------
This module exposes the RAG system as a REST API.

It is a thin orchestration layer:
- FastAPI endpoints only
- No ML logic inside
- No embeddings or FAISS logic inside

Fixes included:
---------------
✔ Safe error handling for /ask (prevents silent 500 crashes)
✔ Better logging for debugging inside Docker
✔ Robust response format
✔ Prevents container "silent failures"
"""

from fastapi import FastAPI, UploadFile, File
import shutil
import os
import traceback

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

rag = RAGPipeline(
    retriever=Retriever(),
    llm=llm
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "RAG API is running"
    }


# --------------------------------------------------
# Ask endpoint (FIXED - SAFE VERSION)
# --------------------------------------------------

@app.post("/ask")
def ask_question(payload: dict):

    try:
        question = payload.get("question", "")

        if not question:
            return {
                "error": "Empty question received"
            }

        result = rag.ask(question)

        # Safe extraction (prevents crash if keys missing)
        answer = result.get("answer", "No answer generated")
        sources = result.get("sources", [])

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    except Exception as e:

        print("🔥 ERROR in /ask endpoint")
        traceback.print_exc()

        return {
            "error": str(e),
            "answer": "",
            "sources": []
        }


# --------------------------------------------------
# Upload endpoint (FIXED SAFETY)
# --------------------------------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

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

        print("🔥 ERROR in /upload endpoint")
        traceback.print_exc()

        return {
            "status": "error",
            "message": str(e)
        }