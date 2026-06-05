"""
Purpose:
    Central configuration module for the RAG Knowledge Engine Platform.

Why this file exists:
    Instead of hardcoding values throughout the application,
    we keep important settings in a single location.

How it fits into the architecture:
    Every component of the system can import configuration
    values from this module.

Examples:
    - LLM model name
    - Chunk size
    - Chunk overlap
    - Retrieval top-k value
    - Data directories
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------

OLLAMA_MODEL_NAME = "llama3"


# ---------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------

TOP_K_RESULTS = 5