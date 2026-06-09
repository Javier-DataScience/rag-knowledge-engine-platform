"""
Purpose:
Central configuration module for the RAG Knowledge Engine Platform.

Why this file exists:
Instead of hardcoding values throughout the application,
we keep important settings in a single location and
load them from environment variables when possible.

How it fits into the architecture:
Every component of the system can import configuration
values from this module.

Examples:
- LLM model name
- Ollama host
- Embedding model name
- Chunk size
- Chunk overlap
- Retrieval top-k value
- Data directories
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------

# Project Paths

# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# ---------------------------------------------------------------------

# Environment Variables

# ---------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------

# LLM Configuration

# ---------------------------------------------------------------------

OLLAMA_MODEL_NAME = os.getenv(
"OLLAMA_MODEL_NAME",
"llama3"
)

OLLAMA_HOST = os.getenv(
"OLLAMA_HOST",
"http://localhost:11434"
)

# ---------------------------------------------------------------------

# Embedding Configuration

# ---------------------------------------------------------------------

EMBEDDING_MODEL_NAME = os.getenv(
"EMBEDDING_MODEL_NAME",
"BAAI/bge-small-en-v1.5"
)

# ---------------------------------------------------------------------

# Chunking Configuration

# ---------------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# ---------------------------------------------------------------------

# Retrieval Configuration

# ---------------------------------------------------------------------

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
