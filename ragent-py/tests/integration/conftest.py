"""
集成 / 端到端测试公共夹具.

真跑 HTTP 中间件、真跑 SQL（SQLite 文件库），只把两个外部依赖换成替身：
Redis（认证 + 并发闸门）与大模型（流式补全）。

SQLite 用文件库 + NullPool 而非内存库：TestClient 自带事件循环，与 pytest-asyncio
的循环不是同一个，内存库配 StaticPool 会把连接钉在创建它的那个循环上。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.database import Base
from app.core.snowflake import get_snowflake_id_str
from app.models.agent import (
    AgentContextCompactionDO,
    AgentConversationDO,
    AgentMessageDO,
    AgentProfileDO,
    AgentPromptDO,
    AgentStateDO,
)
from app.models.rag import IntentNodeDO

INTEGRATION_TABLES = [
    AgentProfileDO.__table__,
    AgentPromptDO.__table__,
    AgentConversationDO.__table__,
    AgentMessageDO.__table__,
    AgentStateDO.__table__,
    AgentContextCompactionDO.__table__,
    IntentNodeDO.__table__,
]

TEST_USER_ID = "1234567890123456789"
TEST_TOKEN = "integration-token"
AUTH_HEADERS = {"Authorization": TEST_TOKEN}


# ---------------------------------------------------------------------------
# Redis 替身
# ---------------------------------------------------------------------------

class FakeRedis:
    """认证中间件、Agent 并发闸门、提示词缓存共用的最小 Redis"""

    def __init__(self, user_id: str = TEST_USER_ID, token: str = TEST_TOKEN,
                 logged_in: bool = True) -> None:
        self.data: dict[str, Any] = {}
        self.user_id = user_id
        self.token = token
        self.logged_in = logged_in

    async def get(self, key: str) -> Any:
        if key.startswith("login:token:"):
            # 只认登记过的那个 token：带 Bearer 前缀或伪造的都该认不出
            if self.logged_in and key == f"login:token:{self.token}":
                return self.user_id
            return None
        return self.data.get(key)

    async def set(self, key: str, value: Any, nx: bool = False, px: int | None = None,
                  ex: int | None = None, **_: Any) -> bool:
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if self.data.pop(key, None) is not None:
                removed += 1
        return removed

    async def hgetall(self, key: str) -> dict[str, str]:
        if not self.logged_in:
            return {}
        return {"username": "integration", "role": "admin", "avatar": ""}

    async def expire(self, key: str, seconds: int) -> bool:
        return True

    async def eval(self, script: str, numkeys: int, key: str, expected: str) -> int:
        """对齐 CAS 删除脚本的语义"""
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0

    async def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 数据库夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def db_factory(tmp_path: Any):
    """文件库 + NullPool：每条连接都在当前事件循环里现开现关"""
    url = f"sqlite+aiosqlite:///{(tmp_path / 'integration.db').as_posix()}"
    engine = create_async_engine(url, poolclass=NullPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _create() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=INTEGRATION_TABLES))

    asyncio.run(_create())
    try:
        yield factory
    finally:
        asyncio.run(engine.dispose())


def run_async(coro: Any) -> Any:
    """在夹具（同步上下文）里跑一段异步逻辑，每次一个独立循环"""
    return asyncio.run(coro)


async def seed_builtin_agent(
    db: AsyncSession,
    prompts: dict[str, str] | None = None,
    name: str = "内置智能体",
) -> str:
    """铺一个内置且激活的智能体，附带槽位提示词"""
    agent_id = get_snowflake_id_str()
    now = datetime.now()
    db.add(AgentProfileDO(
        id=agent_id, name=name, description="集成测试用", builtin=1, active=1,
        create_time=now, update_time=now, deleted=0,
    ))
    for slot_key, content in (prompts or {}).items():
        db.add(AgentPromptDO(
            id=get_snowflake_id_str(), agent_id=agent_id, slot_key=slot_key, content=content,
            create_time=now, update_time=now, deleted=0,
        ))
    await db.commit()
    return agent_id


async def seed_intent_node(
    db: AsyncSession,
    intent_code: str,
    *,
    kind: int = 2,
    mcp_tool_id: str | None = None,
    name: str = "节点",
    description: str | None = None,
    parent_code: str | None = None,
    enabled: int = 1,
) -> None:
    now = datetime.now()
    db.add(IntentNodeDO(
        id=get_snowflake_id_str(), intent_code=intent_code, name=name, level=2,
        parent_code=parent_code, description=description, kind=kind,
        mcp_tool_id=mcp_tool_id, enabled=enabled, deleted=0, collection_names=[],
        create_time=now, update_time=now,
    ))
    await db.commit()


class FakeMcpExecutor:
    """不连真 Server 的 MCP 执行器：只需满足 agent.tools.McpToolExecutor 协议"""

    def __init__(self, tool_id: str, description: str = "远端工具", reply: str = "工具输出") -> None:
        self._tool_id = tool_id
        self._description = description
        self._reply = reply
        self.calls: list[dict[str, Any]] = []

    @property
    def tool_id(self) -> str:
        return self._tool_id

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self._tool_id,
            "description": self._description,
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市"}},
                "required": ["city"],
            },
            "annotations": {"readOnlyHint": True},
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(arguments))
        return {"content": [{"type": "text", "text": self._reply}], "isError": False}


# ---------------------------------------------------------------------------
# HTTP 客户端夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def client(db_factory: Any, monkeypatch: pytest.MonkeyPatch):
    """
    带真中间件的 TestClient。

    全局 ``async_session_factory`` 一并换掉：Agent 后台运行体自建会话，
    只覆盖 ``get_db`` 依赖的话它会去连真 PG。
    """
    import app.core.database
    import app.core.redis_client
    import app.services.agent_chat_service

    fake_redis = FakeRedis()
    monkeypatch.setattr(app.core.redis_client, "_redis_client", fake_redis)
    monkeypatch.setattr(app.core.database, "async_session_factory", db_factory)
    monkeypatch.setattr(app.services.agent_chat_service, "async_session_factory", db_factory)

    async def override_get_db():
        async with db_factory() as session:
            yield session

    async def fake_init_redis():
        return fake_redis

    async def fake_close_redis():
        return None

    from unittest.mock import AsyncMock, patch

    with patch("app.core.redis_client.init_redis", new=AsyncMock(side_effect=fake_init_redis)), \
         patch("app.core.redis_client.close_redis", new=AsyncMock(side_effect=fake_close_redis)):
        import app.main
        from app.core.database import get_db

        app.main.app.dependency_overrides[get_db] = override_get_db
        try:
            with TestClient(app.main.app, raise_server_exceptions=False) as test_client:
                test_client.fake_redis = fake_redis  # type: ignore[attr-defined]
                test_client.db_factory = db_factory   # type: ignore[attr-defined]
                yield test_client
        finally:
            app.main.app.dependency_overrides.clear()
