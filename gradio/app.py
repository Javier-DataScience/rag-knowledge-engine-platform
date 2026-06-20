"""
Module: app.py (Gradio Frontend)

Purpose:
--------
Gradio UI for RAG Knowledge Engine Platform.

This UI ONLY:
- Uploads PDFs to FastAPI
- Sends questions to FastAPI
- Displays answers + sources

No ML logic, no embeddings, no FAISS here.
"""

import os
import requests
import gradio as gr


# --------------------------------------------------
# API CONFIGURATION (Docker-safe)
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# --------------------------------------------------
# UPLOAD PDF FUNCTION (FIXED)
# --------------------------------------------------

def upload_pdf(file):

    if file is None:
        return "❌ Please select a PDF file."

    try:
        # Gradio provides a file object, NOT a path string
        file_path = file.name

        with open(file_path, "rb") as f:

            response = requests.post(
                f"{API_URL}/upload",
                files={
                    "file": (
                        os.path.basename(file_path),
                        f,
                        "application/pdf"
                    )
                },
                timeout=600
            )

        response.raise_for_status()
        result = response.json()

        return (
            f"✅ {result.get('message', 'Upload successful')}\n"
            f"File: {result.get('file', file_path)}"
        )

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to FastAPI backend."

    except requests.exceptions.Timeout:
        return "❌ Upload timed out. Backend still processing."

    except Exception as e:
        return f"❌ Upload failed: {str(e)}"


# --------------------------------------------------
# ASK QUESTION FUNCTION
# --------------------------------------------------

def ask_question(question):

    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question},
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        if not sources:
            return answer, "No sources returned."

        unique_sources = set()

        for source in sources:

            if isinstance(source, dict):
                source_name = source.get("source", "Unknown Source")
                page = source.get("page", "N/A")
            else:
                source_name = str(source)
                page = "N/A"

            unique_sources.add((source_name, page))

        formatted_sources = "\n".join(
            f"• {src} (Page {page})"
            for src, page in sorted(unique_sources, key=lambda x: str(x[1]))
        )

        return answer, formatted_sources

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to backend.", ""

    except requests.exceptions.Timeout:
        return "Error: Request timed out.", ""

    except Exception as e:
        return f"Error: {str(e)}", ""


# --------------------------------------------------
# GRADIO UI (FIXED DARK MODE + DOCKER SAFE)
# --------------------------------------------------

with gr.Blocks(
    title="RAG Knowledge Engine Platform",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# RAG Knowledge Engine Platform")
    gr.Markdown("Upload a PDF and ask questions about it.")

    # --------------------------
    # Upload Section
    # --------------------------

    gr.Markdown("## Upload PDF")

    pdf_file = gr.File(
        file_types=[".pdf"],
        label="Select PDF Document"
    )

    upload_button = gr.Button("Upload and Index PDF")

    upload_status = gr.Textbox(
        label="Upload Status"
    )

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_file,
        outputs=upload_status
    )

    # --------------------------
    # Q&A Section
    # --------------------------

    gr.Markdown("## Ask Questions")

    question = gr.Textbox(
        label="Question",
        placeholder="What is supervised machine learning?"
    )

    ask_button = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    sources = gr.Textbox(
        label="Sources",
        lines=5
    )

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources]
    )


# --------------------------------------------------
# DOCKER-SAFE LAUNCH
# --------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )