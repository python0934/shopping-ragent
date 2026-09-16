"""
Redis connection manager — async redis-py with lifecycle hooks.

Provides a single async Redis client that is created on app startup
and closed on app shutdown.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from app.config import settings

# Module-level client (initialised in lifespan)
_redis_client: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    """Create and ping the async Redis connection."""
    global _redis_client
    _redis_client = aioredis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        password=settings.redis.password or None,
        db=settings.redis.db,
        decode_responses=True,
        socket_connect_timeout=5,
    )
    await _redis_client.ping()
    return _redis_client


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()
        _redis_client = None


def get_redis() -> aioredis.Redis:
    """Return the current Redis client (raises if not initialised)."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialised. Call init_redis() first.")
    return _redis_client
