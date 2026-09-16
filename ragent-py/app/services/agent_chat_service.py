"""
Agent 对话编排 — mirrors agent.service.impl.AgentChatServiceImpl.

一次流式问答的完整编排：闸门 -> meta -> 建会话/落提问 -> 装配 Agent ->
事件桥推 SSE -> 收尾释放。

闸门抢在任何副作用之前：抢不到就直接拒，不能已经写了半截数据才告诉用户「忙」。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.enums import AgentMessageStatus
from app.agent.event_bridge import AgentStreamEventBridge
from app.agent.provider import ReActAgentProvider
from app.agent.run_gate import AgentRunGate
from app.agent.tools import RuntimeContext
from app.core.database import async_session_factory
from app.core.exceptions import ClientException
from app.core.redis_client import get_redis
from app.core.snowflake import get_snowflake_id_str
from app.core.sse import SseSender, sse_event_stream, stream_task_manager
from app.services.agent_conversation_service import AgentConversationService

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
"""``X-Accel-Buffering: no`` 让 Nginx 别攒帧，否则流式会退化成一次性返回"""

MAX_QUESTION_LENGTH = 8_000


@dataclass
class _RunState:
    """
    一次运行的可变状态。

    落库回调建在提问落库之前（meta 帧得先走），replyToMessageId 只能后填，
    故用引用对象跨阶段传递。同进程可能并发多条流，挂到实例属性上会串。
    """

    user_message_id: str | None = None


class AgentChatService:
    """Agent 流式对话服务"""

    def __init__(
        self,
        db: AsyncSession,
        provider: ReActAgentProvider | None = None,
        redis: Any | None = None,
    ) -> None:
        self.db = db
        self.provider = provider or ReActAgentProvider()
        self.redis = redis if redis is not None else get_redis()
        self.gate = AgentRunGate(self.redis)

    # ------------------------------------------------------------------
    # 流式对话
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        user_id: str,
        question: str,
        conversation_id: str | None = None,
    ) -> StreamingResponse:
        """
        发起一次流式问答，立即返回 SSE 响应体。

        真正的运行跑在后台任务里：控制器返回后连接还得挂着持续推帧。
        """
        trimmed = (question or "").strip()
        if not trimmed:
            raise ClientException("问题不能为空")
        if len(trimmed) > MAX_QUESTION_LENGTH:
            raise ClientException(f"问题长度不能超过 {MAX_QUESTION_LENGTH} 字")

        cid = (conversation_id or "").strip() or get_snowflake_id_str()
        task_id = get_snowflake_id_str()

        # 闸门先于一切副作用：抢不到就不该留下任何痕迹
        release_gate = await self.gate.acquire(user_id, task_id, cid)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        sender = SseSender(queue)
        started = False

        try:
            task = asyncio.create_task(
                self._run(user_id, cid, task_id, trimmed, sender, release_gate)
            )
            await stream_task_manager.register(task_id, task, sender)
            started = True
        finally:
            if not started:
                await release_gate()
                await stream_task_manager.unregister(task_id)

        return StreamingResponse(
            sse_event_stream(queue),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    async def _run(
        self,
        user_id: str,
        conversation_id: str,
        task_id: str,
        question: str,
        sender: SseSender,
        release_gate: Any,
    ) -> None:
        """后台运行体：自建会话，与请求作用域的 AsyncSession 无关"""
        try:
            async with async_session_factory() as db:
                conversation = AgentConversationService(db)
                state = _RunState()
                bridge = AgentStreamEventBridge(
                    sender=sender,
                    conversation_id=conversation_id,
                    task_id=task_id,
                    persist_callback=self._make_persist_callback(
                        db, conversation, user_id, conversation_id, state,
                    ),
                )

                await bridge.send_meta()

                # 会话与提问先落库：即便后面 Agent 装配炸了，用户这轮提问也不该凭空消失
                await conversation.touch_conversation(user_id, conversation_id, question)
                state.user_message_id = await conversation.add_user_message(
                    user_id, conversation_id, question,
                )
                await db.commit()

                agent, catalog = await self.provider.build(db, user_id, conversation_id)
                bridge.set_display_name_resolver(catalog.display_name_of)

                await bridge.run(agent.stream_events(
                    question,
                    RuntimeContext(user_id=user_id, session_id=conversation_id),
                ))
                await db.commit()
        except asyncio.CancelledError:
            logger.info("Agent 运行被取消, taskId: %s", task_id)
            if not sender.is_closed:
                await sender.complete()
            raise
        except Exception as e:
            logger.exception("Agent 运行异常, taskId: %s, error: %s", task_id, e)
            if not sender.is_closed:
                await sender.fail(e)
        finally:
            await release_gate()
            await stream_task_manager.unregister(task_id)

    def _make_persist_callback(
        self,
        db: AsyncSession,
        conversation: AgentConversationService,
        user_id: str,
        conversation_id: str,
        state: _RunState,
    ) -> Any:
        """收尾落库口：答复挂在触发它的那条提问下面，前端据此配对渲染"""

        async def persist(
            content: str,
            thinking: str,
            blocks: list[dict[str, Any]] | None,
            status: AgentMessageStatus,
        ) -> str:
            message_id = await conversation.add_assistant_message(
                user_id=user_id,
                conversation_id=conversation_id,
                reply_to_message_id=state.user_message_id,
                content=content,
                thinking_content=thinking or None,
                blocks=blocks,
                message_status=status,
            )
            await db.commit()
            return message_id

        return persist

    # ------------------------------------------------------------------
    # 停止
    # ------------------------------------------------------------------

    async def stop(self, user_id: str, task_id: str) -> bool:
        """
        按 taskId 停流。

        取消后台任务即可：事件桥在 ``CancelledError`` 里走 cancel 收尾，
        已产出的内容会落库并标 INTERRUPTED。
        """
        if not task_id:
            raise ClientException("taskId 不能为空")
        return await stream_task_manager.cancel(task_id)

    async def stop_by_conversation(self, user_id: str, conversation_id: str) -> bool:
        """删会话前先停该会话正在跑的流，否则流会把已删会话的消息又写回来"""
        task_id = await self.gate.running_task_id(user_id, conversation_id)
        if not task_id:
            return False
        return await stream_task_manager.cancel(task_id)
