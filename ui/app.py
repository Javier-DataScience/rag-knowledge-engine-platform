"""
Module: app.py

Streamlit frontend for RAG Knowledge Engine Platform.

Frontend ONLY:
- Sends requests to FastAPI backend
- Displays answers and sources
- No ML logic inside UI
"""

import os
import requests
import streamlit as st

# --------------------------------------------------
# Backend Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

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

            with st.spinner("Uploading and indexing PDF..."):

                response = requests.post(
                    f"{API_URL}/upload",
                    files=files,
                    timeout=300
                )

                response.raise_for_status()

            st.success("Indexing completed successfully.")

        except requests.exceptions.Timeout:
            st.error("Upload timed out. Backend may still be processing.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI backend.")

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
        "Wait until indexing is finished before asking questions."
    )

else:

    question = st.text_input(
        "Ask a question about the document:"
    )

    if st.button("Ask"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Generating answer..."
                ):

                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question
                        },
                        timeout=300
                    )

                    response.raise_for_status()

                    result = response.json()

                st.subheader("Answer")

                st.write(
                    result.get(
                        "answer",
                        "No answer returned"
                    )
                )

                # --------------------------------------------------
                # SAFE SOURCE HANDLING
                # --------------------------------------------------

                st.subheader("Sources")

                sources = result.get(
                    "sources",
                    None
                )

                if not sources:

                    st.info(
                        "No sources returned from backend."
                    )

                else:

                    unique_sources = set()

                    for source in sources:

                        if isinstance(
                            source,
                            dict
                        ):

                            source_file = source.get(
                                "source",
                                "Unknown Source"
                            )

                            page = source.get(
                                "page",
                                "N/A"
                            )

                        else:

                            source_file = str(
                                source
                            )

                            page = "N/A"

                        unique_sources.add(
                            (
                                source_file,
                                page
                            )
                        )

                    for (
                        source_file,
                        page
                    ) in sorted(
                        unique_sources,
                        key=lambda x: str(x[1])
                    ):

                        st.write(
                            f"• {source_file} (Page {page})"
                        )

            except requests.exceptions.Timeout:

                st.error(
                    "Request timed out. Model may be slow."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI backend."
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )