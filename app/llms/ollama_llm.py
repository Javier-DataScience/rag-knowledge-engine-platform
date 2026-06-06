"""
Purpose:
    Ollama LLM wrapper for the RAG Knowledge Engine Platform.

Why this file exists:
    Provides a clean interface for interacting with
    local LLMs running through Ollama.

How it fits into the architecture:

    Prompt
        ↓
    OllamaLLM
        ↓
    Llama 3
        ↓
    Response

Responsibilities:
    - Send prompts to Ollama
    - Return model responses

Non-responsibilities:
    - Retrieval
    - Chunking
    - Embeddings
    - Prompt construction
"""

from ollama import chat

from app.config.settings import OLLAMA_MODEL_NAME


class OllamaLLM:
    """
    Wrapper around an Ollama-hosted LLM.
    """

    def __init__(self):
        self.model_name = OLLAMA_MODEL_NAME

    def generate(
        self,
        prompt: str
    ) -> str:
        """
        Generate a response from the model.
        """

        response = chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        return response["message"]["content"]