"""
Unified API response model — mirrors Java Result<T> / Results utility.

Every endpoint returns this shape (HTTP 200 always):
{
    "code": "0",
    "message": null,
    "data": { ... },
    "requestId": "req_xxxx"
}
"""

from __future__ import annotations

import uuid
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.core.error_code import BaseErrorCode

T = TypeVar("T")

# Module-level constant (avoids Pydantic v2 class-attribute issues)
SUCCESS_CODE = "0"


class Result(BaseModel, Generic[T]):
    """Global unified response wrapper."""

    code: str = "0"
    message: str | None = None
    data: T | None = None
    requestId: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:16]}")

    @property
    def is_success(self) -> bool:
        return self.code == SUCCESS_CODE


# ---------------------------------------------------------------------------
# Builder helpers — mirrors Java Results utility class
# ---------------------------------------------------------------------------

def success(data: Any = None) -> dict[str, Any]:
    """Build a success response dict (orjson-serialisable)."""
    return {
        "code": SUCCESS_CODE,
        "message": None,
        "data": data,
        "requestId": f"req_{uuid.uuid4().hex[:16]}",
    }


def failure(
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build a failure response dict."""
    return {
        "code": error_code or BaseErrorCode.SERVICE_ERROR.code,
        "message": error_message or BaseErrorCode.SERVICE_ERROR.message,
        "data": None,
        "requestId": f"req_{uuid.uuid4().hex[:16]}",
    }


def failure_from_exception(error_code: str, error_message: str) -> dict[str, Any]:
    """Build a failure response from an AbstractException's fields."""
    return {
        "code": error_code,
        "message": error_message,
        "data": None,
        "requestId": f"req_{uuid.uuid4().hex[:16]}",
    }
