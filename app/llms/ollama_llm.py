"""
Ollama LLM Wrapper (Stable Environment-Safe Version)

Purpose:
--------
This module provides a clean interface to communicate with Ollama.

It is designed to work in BOTH:
- Local execution (Windows / Python)
- Docker containers (FastAPI, Streamlit, Gradio)

Key principle:
- Host is controlled via environment variable
- Falls back safely to local Ollama instance
"""

import os
from ollama import Client


class OllamaLLM:
    def __init__(self):
        """
        Initialize Ollama client with environment-safe configuration.
        """

        # If running locally → uses localhost
        # If running in Docker → override via OLLAMA_HOST env variable
        self.host = os.getenv(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434"
        )

        self.client = Client(host=self.host)

    def generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama and return generated response.
        """

        try:
            response = self.client.chat(
                model=os.getenv("OLLAMA_MODEL_NAME", "llama3"),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"]

        except Exception as e:
            raise RuntimeError(
                f"Ollama generation failed. Host={self.host}. Error={str(e)}"
            )