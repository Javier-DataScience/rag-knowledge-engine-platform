"""
Purpose:
    Ollama LLM wrapper for the RAG Knowledge Engine Platform.

Why this file exists:
    Provides a clean interface for interacting with local LLMs
    running through Ollama.

Responsibilities:
    - Send prompts to Ollama
    - Return model responses
    - Support configurable Ollama hosts
"""

from ollama import Client

from app.config.settings import (
    OLLAMA_MODEL_NAME,
    OLLAMA_HOST,
)


class OllamaLLM:
    """
    Wrapper around an Ollama-hosted LLM.
    """

    def __init__(self):
        self.model_name = OLLAMA_MODEL_NAME

        self.client = Client(
            host=OLLAMA_HOST
        )

    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response from the model.
        """

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        return response["message"]["content"]