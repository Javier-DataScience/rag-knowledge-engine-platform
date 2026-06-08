"""
Purpose:
    Streamlit frontend for the RAG Knowledge Engine Platform.

Current Version:
    Ask questions through FastAPI and display
    answers and sources.
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="RAG Knowledge Engine",
)

st.title("RAG Knowledge Engine Platform: PDF-RAG Assistant")

question = st.text_input(
    "Ask a question about the loaded document:"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Generating answer..."):

            response = requests.post(
                API_URL,
                json={
                    "question": question
                },
            )

            result = response.json()

            st.subheader("Answer")

            st.write(
                result["answer"]
            )

            st.subheader("Sources")

            for source in result["sources"]:

                st.write(
                    f"• {source['source']} (Page {source['page']})"
                )