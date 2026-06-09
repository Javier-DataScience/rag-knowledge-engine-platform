"""
Module: app.py

Purpose:
    Streamlit frontend for the RAG Knowledge Engine Platform:
    PDF-RAG Assistant.

Features:
    - Upload PDF documents
    - Ask questions about uploaded documents
    - Display answers
    - Display source references
    - Upload progress feedback
    - Basic error handling
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Knowledge Engine Platform: PDF-RAG Assistant",
    layout="wide"
)

st.title("RAG Knowledge Engine Platform: PDF-RAG Assistant")

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "indexing" not in st.session_state:
    st.session_state.indexing = False

# --------------------------------------------------
# PDF Upload Section
# --------------------------------------------------

st.header("Upload PDF")

uploaded_file = st.file_uploader(
    "Select a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Upload and Index PDF"):

        st.session_state.indexing = True

        try:

            st.info("Uploading document...")

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            with st.spinner(
                "Uploading and indexing PDF. This may take a while..."
            ):

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=300
                )

                response.raise_for_status()

            st.success("Indexing completed successfully.")

        except requests.exceptions.Timeout:

            st.error(
                "Upload request timed out. "
                "The backend may still be processing the document."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI backend. "
                "Make sure the API is running."
            )

        except Exception as e:

            st.error(f"Unexpected error: {str(e)}")

        finally:

            st.session_state.indexing = False

# --------------------------------------------------
# Question Answering Section
# --------------------------------------------------

st.header("Ask Questions")

if st.session_state.indexing:

    st.warning(
        "Please wait until indexing is completed before asking questions."
    )

else:

    question = st.text_input(
        "Ask a question about the document:"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                with st.spinner("Generating answer..."):

                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question
                        },
                        timeout=120
                    )

                    response.raise_for_status()

                    result = response.json()

                st.subheader("Answer")

                st.write(result["answer"])

                st.subheader("Sources")

                sources = result.get("sources", [])

                if len(sources) == 0:

                    st.info("No sources returned.")

                else:

                    for source in sources:

                        source_file = source.get(
                            "source",
                            "Unknown Source"
                        )

                        page = source.get(
                            "page",
                            "N/A"
                        )

                        st.write(
                            f"• {source_file} (Page {page})"
                        )

            except requests.exceptions.Timeout:

                st.error(
                    "Request timed out. "
                    "The model may be taking too long to respond."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI backend. "
                    "Make sure the API is running."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )