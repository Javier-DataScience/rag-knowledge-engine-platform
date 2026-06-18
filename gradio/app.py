"""
Module: app.py (Gradio Frontend)

Purpose:
--------
Provides a lightweight Gradio interface for the RAG Knowledge Engine.

Architecture:
-------------
User
 ↓
Gradio UI
 ↓
FastAPI Backend
 ↓
RAG Pipeline
 ↓
FAISS Retriever + Ollama

Features:
---------
- Upload PDF documents
- Ask questions
- Display answers
- Display source citations
- Dark theme UI
- Reuse the same FastAPI backend used by Streamlit

Notes:
------
- No RAG logic lives here.
- No embedding logic lives here.
- No FAISS logic lives here.
- No LLM logic lives here.

This file only consumes the API.
"""

import requests
import gradio as gr

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

def upload_pdf(file):

    if file is None:
        return "Please select a PDF file."

    try:

        with open(file, "rb") as f:

            response = requests.post(
                f"{API_URL}/upload",
                files={
                    "file": (
                        file.split("\\")[-1],
                        f,
                        "application/pdf"
                    )
                },
                timeout=600
            )

        response.raise_for_status()

        result = response.json()

        return (
            f"✅ {result['message']}\n"
            f"File: {result['file']}"
        )

    except Exception as e:

        return f"❌ Upload failed: {str(e)}"


# --------------------------------------------------
# Ask Question
# --------------------------------------------------

def ask_question(question):

    if not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    try:

        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question
            },
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "answer",
            ""
        )

        sources = result.get(
            "sources",
            []
        )

        if len(sources) == 0:

            sources_text = (
                "No sources returned."
            )

        else:

            unique_sources = set()

            for source in sources:

                source_name = source.get(
                    "source",
                    "Unknown Source"
                )

                page = source.get(
                    "page",
                    "N/A"
                )

                unique_sources.add(
                    (source_name, page)
                )

            lines = []

            for source_name, page in sorted(
                unique_sources,
                key=lambda x: str(x[1])
            ):

                lines.append(
                    f"• {source_name} (Page {page})"
                )

            sources_text = "\n".join(lines)

        return (
            answer,
            sources_text
        )

    except Exception as e:

        return (
            f"Error: {str(e)}",
            ""
        )


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

with gr.Blocks(
    title="RAG Knowledge Engine Platform",
    css="""
    body {
        background-color: #111111;
    }

    .gradio-container {
        background-color: #111111 !important;
    }

    /* Main title */
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Section titles */
    h2, h3 {
        color: #F5F5F5 !important;
    }

    /* Text */
    p, label, span {
        color: #D9D9D9 !important;
    }

    /* Cards / containers */
    .gr-group,
    .gr-box,
    .gr-form,
    .block {
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
    }

    /* Textbox containers */
    .gr-textbox {
        background-color: #1A1A1A !important;
    }

    /* Input fields */
    textarea,
    input {
        background-color: #222222 !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
    }

    /* Buttons */
    button {
        background-color: #2F2F2F !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
    }

    footer {
        display: none !important;
    }
    
        /* ------------------------------------
   File Upload Area
------------------------------------ */

[data-testid="file-upload"] {
    background-color: #1E1E1E !important;
    border: 1px solid #444444 !important;
    color: #FFFFFF !important;
}

[data-testid="file-upload"] * {
    color: #D9D9D9 !important;
}

.file-preview,
.file-upload,
.upload-box {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
}
    """
) as demo:

    gr.Markdown(
        "# RAG Knowledge Engine Platform"
    )

    gr.Markdown(
        "Upload a PDF and ask questions about it."
    )

    # ----------------------------------------------
    # Upload Section
    # ----------------------------------------------

    gr.Markdown("## Upload PDF")

    pdf_file = gr.File(
        file_types=[".pdf"],
        label="PDF Document"
    )

    upload_button = gr.Button(
        "Upload and Index PDF"
    )

    upload_status = gr.Textbox(
        label="Upload Status"
    )

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_file,
        outputs=upload_status
    )

    # ----------------------------------------------
    # Question Answering Section
    # ----------------------------------------------

    gr.Markdown("## Ask Questions")

    question = gr.Textbox(
        label="Question",
        placeholder="What is supervised machine learning?"
    )

    ask_button = gr.Button(
        "Ask"
    )

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
        outputs=[
            answer,
            sources
        ]
    )


# --------------------------------------------------
# Launch
# --------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        inbrowser=True
    )