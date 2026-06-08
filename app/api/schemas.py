"""
Purpose:
    API request and response schemas.

Why this file exists:
    Defines the data structures exchanged
    between API clients and the backend.

Current Schemas:
    - AskRequest
    - SourceInfo
    - AskResponse
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    User question sent to the API.
    """

    question: str


class SourceInfo(BaseModel):
    """
    Source information returned by RAG.
    """

    source: str
    page: int


class AskResponse(BaseModel):
    """
    API response containing the answer
    and supporting sources.
    """

    answer: str
    sources: list[SourceInfo]