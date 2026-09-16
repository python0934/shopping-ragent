"""
Three-tier exception hierarchy — mirrors Java AbstractException / ClientException / ServiceException / RemoteException.

All exceptions carry an error_code string and an error_message string.
The global exception handler in middleware.py converts them to Result JSON
with HTTP 200 (never 4xx/5xx).
"""

from __future__ import annotations

from app.core.error_code import BaseErrorCode, ErrorCode


class AbstractException(Exception):
    """Base class for all business exceptions."""

    def __init__(
        self,
        message: str | None = None,
        error_code: ErrorCode | None = None,
        cause: Exception | None = None,
    ):
        self.error_code: str = error_code.code if error_code else BaseErrorCode.SERVICE_ERROR.code
        self.errorMessage: str = (
            message if message else (error_code.message if error_code else BaseErrorCode.SERVICE_ERROR.message)
        )
        self.__cause__ = cause
        super().__init__(self.errorMessage)

    def __str__(self) -> str:
        return f"{type(self).__name__}(code={self.error_code!r}, message={self.errorMessage!r})"


class ClientException(AbstractException):
    """User-side error (bad request, validation, auth, etc.)."""

    def __init__(
        self,
        message: str | None = None,
        error_code: ErrorCode | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message,
            error_code or BaseErrorCode.CLIENT_ERROR,
            cause,
        )


class ServiceException(AbstractException):
    """System execution error (unexpected internal failures)."""

    def __init__(
        self,
        message: str | None = None,
        error_code: ErrorCode | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message or (error_code.message if error_code else None),
            error_code or BaseErrorCode.SERVICE_ERROR,
            cause,
        )


class RemoteException(AbstractException):
    """Third-party service error (AI model timeout, S3 failure, etc.)."""

    def __init__(
        self,
        message: str | None = None,
        error_code: ErrorCode | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message,
            error_code or BaseErrorCode.REMOTE_ERROR,
            cause,
        )


# ---- Convenience aliases used across modules ----

class NotLoginException(ClientException):
    """Raised when the user is not authenticated."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or BaseErrorCode.NOT_LOGIN.message,
            BaseErrorCode.NOT_LOGIN,
        )


class NotRoleException(ClientException):
    """Raised when the user lacks the required role."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or BaseErrorCode.NOT_ROLE.message,
            BaseErrorCode.NOT_ROLE,
        )


class IdempotentRejectException(ClientException):
    """Raised when a duplicate-submission is detected."""

    def __init__(self):
        super().__init__(
            BaseErrorCode.IDEMPOTENT_REJECT.message,
            BaseErrorCode.IDEMPOTENT_REJECT,
        )


class RateLimitRejectException(ClientException):
    """Raised when the rate-limit queue is full."""

    def __init__(self):
        super().__init__(
            BaseErrorCode.RATE_LIMIT_REJECT.message,
            BaseErrorCode.RATE_LIMIT_REJECT,
        )


class DemoModeRejectException(ClientException):
    """Raised when a write operation is attempted in demo mode."""

    def __init__(self):
        super().__init__(
            BaseErrorCode.DEMO_MODE_REJECT.message,
            BaseErrorCode.DEMO_MODE_REJECT,
        )
