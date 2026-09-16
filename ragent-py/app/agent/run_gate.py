"""
Agent 并发闸门 — mirrors agent.service.handler.AgentRunGate.

一个用户同一时刻只跑一条流，覆盖整个流生命周期。
运行位存 ``taskId|conversationId``，删会话时凭它认出该停哪条流。
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from app.config import settings
from app.core.exceptions import ClientException

logger = logging.getLogger(__name__)

RUNNING_KEY_PREFIX = "ragent:agent:running:"
SLOT_SEPARATOR = "|"

# 只放自己占的位：运行位若被 TTL 挤掉又被下一轮抢走，无条件删会放掉别人的闸门。
# GET + DEL 非原子，交给 Lua 在 Redis 单线程里一次做完。
_COMPARE_AND_DELETE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

ReleaseGate = Callable[[], Awaitable[None]]


class AgentRunGate:
    """用户维度的 Agent 并发闸门"""

    def __init__(self, redis: Any, sse_timeout_ms: int | None = None) -> None:
        self.redis = redis
        self._sse_timeout_ms = (
            sse_timeout_ms if sse_timeout_ms is not None else settings.agent.sse_timeout_ms
        )

    async def acquire(self, user_id: str, task_id: str, conversation_id: str) -> ReleaseGate:
        """抢运行位，抢不到直接拒绝；返回的释放动作由调用方挂到收尾路上"""
        slot_value = f"{task_id}{SLOT_SEPARATOR}{conversation_id}"
        acquired = await self.redis.set(
            self._running_key(user_id), slot_value, nx=True, px=self.ttl_ms,
        )
        if not acquired:
            raise ClientException("当前会话处理中，请稍后再发起新的对话")
        return lambda: self._release(user_id, slot_value)

    async def running_task_id(self, user_id: str, conversation_id: str) -> str | None:
        """该用户此刻正跑的流若属于这个会话，返回它的 taskId，否则 None"""
        slot_value = await self.redis.get(self._running_key(user_id))
        if not slot_value:
            return None
        if isinstance(slot_value, bytes):
            slot_value = slot_value.decode("utf-8")

        separator = slot_value.find(SLOT_SEPARATOR)
        if separator < 0 or slot_value[separator + 1:] != conversation_id:
            return None
        return slot_value[:separator]

    @property
    def ttl_ms(self) -> int:
        """进程崩溃时没人来释放，TTL 是唯一出路；取 SSE 超时的两倍"""
        return self._sse_timeout_ms * 2

    async def _release(self, user_id: str, slot_value: str) -> None:
        """重复调用天然安全，值对不上就是空操作"""
        try:
            await self.redis.eval(
                _COMPARE_AND_DELETE_LUA, 1, self._running_key(user_id), slot_value,
            )
        except Exception as e:
            # 释放失败只报警：TTL 兜底，最坏是该用户被挡到运行位过期
            logger.warning("Agent 运行位释放失败, userId: %s, error: %s", user_id, e)

    @staticmethod
    def _running_key(user_id: str) -> str:
        return RUNNING_KEY_PREFIX + user_id
