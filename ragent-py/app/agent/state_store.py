"""
Agent 工作状态存储 — mirrors agent.state.PgAgentStateStore.

官方 AgentScope 2.0.2 仅提供 in-memory / JSON 文件 / Redis / MySQL 四种状态介质，
本项目主存储为 PostgreSQL，故自实现挂 t_agent_state。

payload 是框架自有编解码的不透明 JSON，不与业务表建立结构约定：这里存的是
Msg 列表（会话上下文）序列化后的 dict，读回来由调用方按 Msg.from_dict 还原。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import delete as sa_delete
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentStateDO

logger = logging.getLogger(__name__)

ANONYMOUS_USER = "__anon__"
"""与官方 JsonFileAgentStateStore 对齐的匿名用户哨兵，PG 主键列不可为空"""

CONTEXT_STATE_KEY = "context"
"""会话上下文在 t_agent_state 里的状态键，对应 AgentScope AgentState 的 memory 段"""


def _safe_user(user_id: str | None) -> str:
    return user_id if user_id and user_id.strip() else ANONYMOUS_USER


class PgAgentStateStore:
    """AgentStateStore 的 PostgreSQL 实现（复合主键 user_id + session_id + state_key）"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # -- 写 -----------------------------------------------------------------

    async def save(self, user_id: str | None, session_id: str, key: str, value: Any) -> None:
        """
        插入或整体覆盖一个状态键。

        ``value`` 可以是 dict / list / 任意 JSON 可序列化对象；None 存成 NULL。
        """
        payload = self._to_payload(value)
        if self._dialect_name() != "postgresql":
            await self._save_without_upsert(user_id, session_id, key, payload)
            return

        stmt = pg_insert(AgentStateDO).values(
            user_id=_safe_user(user_id),
            session_id=session_id,
            state_key=key,
            payload=payload,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "session_id", "state_key"],
            set_={"payload": stmt.excluded.payload, "update_time": func.now()},
        )
        await self.db.execute(stmt)
        await self.db.flush()

    async def _save_without_upsert(
        self,
        user_id: str | None,
        session_id: str,
        key: str,
        payload: Any,
    ) -> None:
        """
        非 PG 方言的等价写法：先查再插/改。

        只为集成测试能在 SQLite 上跑真 SQL；生产走上面的原子 UPSERT，
        这里不拼并发——同一 (user, session) 的状态由并发闸门保证只有一条流在写。
        """
        existing = (await self.db.execute(
            select(AgentStateDO).where(
                AgentStateDO.user_id == _safe_user(user_id),
                AgentStateDO.session_id == session_id,
                AgentStateDO.state_key == key,
            )
        )).scalar_one_or_none()

        if existing is not None:
            existing.payload = payload
            existing.update_time = datetime.now()
        else:
            now = datetime.now()
            self.db.add(AgentStateDO(
                user_id=_safe_user(user_id),
                session_id=session_id,
                state_key=key,
                payload=payload,
                create_time=now,
                update_time=now,
            ))
        await self.db.flush()

    def _dialect_name(self) -> str:
        bind = self.db.get_bind() if hasattr(self.db, "get_bind") else None
        return getattr(getattr(bind, "dialect", None), "name", "") or ""

    async def save_list(self, user_id: str | None, session_id: str, key: str, values: list[Any]) -> None:
        """保存一个状态列表（对应 Java 的 save(userId, sessionId, key, List<State>)）"""
        await self.save(user_id, session_id, key, list(values or []))

    # -- 读 -----------------------------------------------------------------

    async def get(self, user_id: str | None, session_id: str, key: str) -> Any | None:
        """取单个状态键的原始 payload，不存在返回 None"""
        return await self._query_payload(user_id, session_id, key)

    async def get_list(self, user_id: str | None, session_id: str, key: str) -> list[Any]:
        """取列表型状态，不存在或不是列表返回空列表"""
        payload = await self._query_payload(user_id, session_id, key)
        if payload is None:
            return []
        return list(payload) if isinstance(payload, list) else []

    async def exists(self, user_id: str | None, session_id: str) -> bool:
        result = await self.db.execute(
            select(AgentStateDO.state_key).where(
                AgentStateDO.user_id == _safe_user(user_id),
                AgentStateDO.session_id == session_id,
            ).limit(1)
        )
        return result.first() is not None

    async def list_session_ids(self, user_id: str | None) -> list[str]:
        """该用户全部有状态的会话 ID，按出现顺序去重"""
        result = await self.db.execute(
            select(AgentStateDO.session_id)
            .where(AgentStateDO.user_id == _safe_user(user_id))
            .order_by(AgentStateDO.session_id.asc())
        )
        seen: list[str] = []
        for (session_id,) in result.all():
            if session_id not in seen:
                seen.append(session_id)
        return seen

    # -- 删 -----------------------------------------------------------------

    async def delete(self, user_id: str | None, session_id: str) -> None:
        """删掉整个会话的全部状态键"""
        await self.db.execute(
            sa_delete(AgentStateDO).where(
                AgentStateDO.user_id == _safe_user(user_id),
                AgentStateDO.session_id == session_id,
            )
        )
        await self.db.flush()

    async def delete_key(self, user_id: str | None, session_id: str, key: str) -> None:
        await self.db.execute(
            sa_delete(AgentStateDO).where(
                AgentStateDO.user_id == _safe_user(user_id),
                AgentStateDO.session_id == session_id,
                AgentStateDO.state_key == key,
            )
        )
        await self.db.flush()

    # -- 内部 ---------------------------------------------------------------

    async def _query_payload(self, user_id: str | None, session_id: str, key: str) -> Any | None:
        result = await self.db.execute(
            select(AgentStateDO.payload).where(
                AgentStateDO.user_id == _safe_user(user_id),
                AgentStateDO.session_id == session_id,
                AgentStateDO.state_key == key,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _to_payload(value: Any) -> Any:
        """
        归一成 JSONB 可存的形态。

        带 to_dict 的对象（Msg）走它自己的序列化；其余原样交给驱动。
        """
        if value is None:
            return None
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if isinstance(value, (list, tuple)):
            return [PgAgentStateStore._to_payload(item) for item in value]
        return value
