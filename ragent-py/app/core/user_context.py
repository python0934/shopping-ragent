"""
User context — mirrors Java UserContext / LoginUser.

Uses contextvars for async-safe per-request user propagation
(replaces Java's TransmittableThreadLocal).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field

from app.core.exceptions import ClientException

# Per-request context variable
_user_var: ContextVar["LoginUser | None"] = ContextVar("login_user", default=None)


@dataclass
class LoginUser:
    """Snapshot of the currently authenticated user."""
    userId: str
    username: str
    role: str
    avatar: str = ""


class UserContext:
    """Static-style accessor for the current request's LoginUser."""

    DEFAULT_AVATAR = "https://avatars.githubusercontent.com/u/583231?v=4"

    @staticmethod
    def set(user: LoginUser) -> None:
        _user_var.set(user)

    @staticmethod
    def get() -> LoginUser | None:
        return _user_var.get()

    @staticmethod
    def require_user() -> LoginUser:
        user = _user_var.get()
        if user is None:
            raise ClientException("未获取到当前登录用户")
        return user

    @staticmethod
    def get_user_id() -> str | None:
        user = _user_var.get()
        return user.userId if user else None

    @staticmethod
    def get_username() -> str | None:
        user = _user_var.get()
        return user.username if user else None

    @staticmethod
    def get_role() -> str | None:
        user = _user_var.get()
        return user.role if user else None

    @staticmethod
    def get_avatar() -> str | None:
        user = _user_var.get()
        return user.avatar if user else None

    @staticmethod
    def clear() -> None:
        _user_var.set(None)

    @staticmethod
    def has_user() -> bool:
        return _user_var.get() is not None
