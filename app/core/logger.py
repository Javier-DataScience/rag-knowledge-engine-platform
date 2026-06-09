"""
Module: logger.py

Purpose:
    Lightweight logging utility for the RAG Knowledge Engine Platform.

    Provides structured logging for:
    - API calls
    - RAG pipeline steps
    - Errors
    - Debug information

    Phase A: simple console logging (no external services)
"""

import logging
import sys


def get_logger(name: str = "rag_app") -> logging.Logger:
    """
    Create and configure a logger instance.
    """

    logger = logging.getLogger(name)

    # Avoid duplicate logs in reload mode (FastAPI / Streamlit)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger