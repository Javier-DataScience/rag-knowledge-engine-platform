"""
Purpose:
    Validate Python communication with Ollama.

Why this file exists:
    Before building the LLM wrapper, we verify that
    Python can successfully send prompts to Ollama
    and receive responses.

Architecture Tested:

    Python
        ↓
    Ollama Client
        ↓
    Llama 3
"""

from ollama import chat


def main():
    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": "What is supervised machine learning? Answer in one sentence."
            }
        ]
    )

    print("\nModel Response:\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()