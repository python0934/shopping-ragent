"""
Idempotent submission guard — mirrors Java @IdempotentSubmit AOP.

Uses Redis SETNX to prevent duplicate submissions within a TTL window.
When hit, returns a JSON error (NOT an SSE stream).
"""

from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, Callable

from app.core.exceptions import IdempotentRejectException
from app.core.redis_client import get_redis
from app.core.user_context import UserContext

logger = logging.getLogger(__name__)

# Default TTL for idempotent keys (seconds)
DEFAULT_TTL = 10


def _make_key(endpoint: str, user_id: str | None) -> str:
    """Build the Redis key for idempotent check."""
    raw = f"{user_id or 'anon'}:{endpoint}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"idempotent:{h}"


async def check_idempotent(endpoint: str, ttl: int = DEFAULT_TTL) -> str:
    """
    Check and acquire the idempotent lock.

    Returns the Redis key on success.
    Raises IdempotentRejectException if the lock is already held.
    """
    user_id = UserContext.get_user_id()
    key = _make_key(endpoint, user_id)
    redis = get_redis()

    acquired = await redis.set(key, "1", nx=True, ex=ttl)
    if not acquired:
        raise IdempotentRejectException()
    return key


async def release_idempotent(key: str) -> None:
    """Release the idempotent lock (optional, TTL handles expiry)."""
    try:
        redis = get_redis()
        await redis.delete(key)
    except Exception:
        logger.debug("Failed to release idempotent key: %s", key)


def idempotent_submit(endpoint: str = "", ttl: int = DEFAULT_TTL):
    """
    Decorator for idempotent submission.

    Usage:
        @idempotent_submit(endpoint="rag_chat")
        async def chat_handler(...):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            ep = endpoint or func.__name__
            key = await check_idempotent(ep, ttl)
            try:
                return await func(*args, **kwargs)
            except Exception:
                # Release on failure so user can retry
                await release_idempotent(key)
                raise
        return wrapper
    return decorator
