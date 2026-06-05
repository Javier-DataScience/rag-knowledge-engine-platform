"""
Purpose:
    Simple validation script for the configuration module.

Why this file exists:
    Before building the rest of the application,
    we want to verify that imports and configuration
    values work correctly.

How it fits into the architecture:
    This is a temporary validation script that confirms
    our project structure is functioning as expected.
"""

from app.config.settings import (
    OLLAMA_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


print("Configuration Loaded Successfully")
print(f"Model: {OLLAMA_MODEL_NAME}")
print(f"Chunk Size: {CHUNK_SIZE}")
print(f"Chunk Overlap: {CHUNK_OVERLAP}")
print(f"Top K Results: {TOP_K_RESULTS}")