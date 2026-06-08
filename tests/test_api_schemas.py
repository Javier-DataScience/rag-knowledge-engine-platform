"""
Purpose:
    Validate API schemas.

Architecture Tested:

    AskRequest
    SourceInfo
    AskResponse
"""

from app.api.schemas import (
    AskRequest,
    SourceInfo,
    AskResponse,
)


def main():
    request = AskRequest(
        question="What is supervised machine learning?"
    )

    source = SourceInfo(
        source="data/raw/Supervised-Learning.pdf",
        page=0,
    )

    response = AskResponse(
        answer="Example answer",
        sources=[source],
    )

    print("\nRequest:\n")
    print(request.model_dump())

    print("\nResponse:\n")
    print(response.model_dump())


if __name__ == "__main__":
    main()