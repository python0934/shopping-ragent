"""Shared test fixtures and helpers."""

from __future__ import annotations

import asyncio
import os

# 单测不碰真网络：置空 MCP Server 列表，让 lifespan 里的连接直接跳过。
# 全局引擎指向 SQLite 内存库：Windows 上连不上的 PG 每次要耗 2s 才报拒绝。
# 必须在导入 app.* 之前设，settings 与 engine 都是模块级单例，读一次就定了。
os.environ.setdefault("RAG__MCP__SERVERS", "[]")
os.environ.setdefault("DATABASE__URL", "sqlite+aiosqlite:///:memory:")

from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

from app.core.user_context import LoginUser, UserContext  # noqa: E402


@pytest.fixture
def mock_login_user() -> LoginUser:
    """A standard test user."""
    return LoginUser(
        userId="1234567890123456789",
        username="testuser",
        role="user",
        avatar="https://example.com/avatar.png",
    )


@pytest.fixture
def mock_admin_user() -> LoginUser:
    """An admin test user."""
    return LoginUser(
        userId="9876543210987654321",
        username="admin",
        role="admin",
        avatar="",
    )


@pytest.fixture(autouse=True)
def clean_user_context():
    """Ensure UserContext is cleaned after each test."""
    yield
    UserContext.clear()


@pytest.fixture
def mock_redis() -> AsyncMock:
    """A mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.hgetall = AsyncMock(return_value={})
    redis.expire = AsyncMock(return_value=True)
    redis.ping = AsyncMock()
    redis.aclose = AsyncMock()
    return redis
