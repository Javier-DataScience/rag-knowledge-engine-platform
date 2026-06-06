"""
Purpose:
    Validate the OllamaLLM wrapper.

Architecture Tested:

    Prompt
        ↓
    OllamaLLM
        ↓
    Llama 3
        ↓
    Response
"""

from app.llms.ollama_llm import OllamaLLM


def main():
    llm = OllamaLLM()

    response = llm.generate(
        "What is supervised machine learning? Answer in one sentence."
    )

    print("\nModel Response:\n")
    print(response)


if __name__ == "__main__":
    main()