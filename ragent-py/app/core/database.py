"""
Database connection — SQLAlchemy 2.0 async engine + session factory.

Mirrors Java HikariCP + MyBatis-Plus configuration.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.config import settings


# ---------------------------------------------------------------------------
# Engine & Session
# ---------------------------------------------------------------------------

def _build_engine() -> AsyncEngine:
    """
    建引擎：PG 走连接池，SQLite 走单连接共享。

    SQLite 不接受 pool_size 一族参数；内存库还得用 StaticPool 把多个会话钉在
    同一条连接上，否则每个会话拿到的都是一张空库。测试与集成验证靠这条路。
    """
    url = settings.database.url
    if url.startswith("sqlite"):
        return create_async_engine(
            url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return create_async_engine(
        url,
        pool_size=settings.database.pool_size,
        pool_timeout=settings.database.connection_timeout / 1000,  # ms → s
        pool_recycle=settings.database.max_lifetime / 1000,        # ms → s
        pool_pre_ping=True,
        echo=False,
    )


engine: AsyncEngine = _build_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """SQLAlchemy declarative base — all ORM models inherit from this."""
    pass


# ---------------------------------------------------------------------------
# Session dependency (for FastAPI Depends)
# ---------------------------------------------------------------------------

async def get_db() -> AsyncSession:  # type: ignore[misc]
    """Yield an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """Dispose the engine connection pool."""
    await engine.dispose()
