"""
Ollama LLM Wrapper (Phase A Stable Version)

Purpose:
- Provide a clean interface to Ollama local server
- Remove environment ambiguity (localhost vs docker vs host.docker.internal)
- Ensure reliable chat() calls from both local and future Docker setups

Key decision:
- We explicitly use 127.0.0.1 because this module is executed on the HOST (not inside Docker)
"""

import os
from ollama import Client


class OllamaLLM:
    def __init__(self):
        """
        Initialize Ollama client with a stable host configuration.
        """

        # Explicit and safe default for local execution
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

        self.client = Client(host=self.host)

    def generate(self, prompt: str) -> str:
        """
        Generate response using Ollama chat model.
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

            # Ollama returns structured response
            return response["message"]["content"]

        except Exception as e:
            raise RuntimeError(
                f"Ollama generation failed. Host={self.host}. Error={str(e)}"
            )