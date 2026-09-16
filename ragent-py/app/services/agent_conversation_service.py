"""
Agent 会话与消息存储 — mirrors agent.service.impl.AgentConversationServiceImpl.

会话列表、消息轨迹的读写，以及删会话时对运行期状态（闸门 + Agent 上下文）的连带清理。
删会话必须先停正在跑的流：否则流会把已删会话的消息又写回来。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.enums import AgentMessageStatus
from app.core.exceptions import ClientException
from app.core.snowflake import get_snowflake_id_str
from app.models.agent import AgentConversationDO, AgentMessageDO, AgentStateDO
from app.schemas.agent import AgentBlock, AgentConversationVO, AgentMessageVO

logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 30
"""自动标题取首条提问的前若干字，列表里放不下更长的"""

RENAME_MAX_LENGTH = 128
"""手动改名放宽到列宽，但仍要挡住越界写库"""

DEFAULT_TITLE = "新对话"

RECENT_TURN_SCAN_MULTIPLIER = 4
"""一轮问答最多摊成 user/assistant/tool 若干行，扫 4 倍窗口才够凑齐 turns 轮"""

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


def _clip_title(text: str | None, limit: int, fallback: str = DEFAULT_TITLE) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return fallback
    return stripped[:limit]


class AgentConversationService:
    """Agent 会话服务"""

    def __init__(self, db: AsyncSession, on_release_runtime: Callable[[str, str], Any] | None = None) -> None:
        """
        :param on_release_runtime: 删会话后驱逐运行期状态的回调 (user_id, conversation_id)
        """
        self.db = db
        self._on_release_runtime = on_release_runtime

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    async def list_conversations(self, user_id: str) -> list[AgentConversationVO]:
        """会话列表：按最后活动时间倒序，带上轮次数"""
        rows = (await self.db.execute(
            select(AgentConversationDO)
            .where(AgentConversationDO.user_id == user_id, AgentConversationDO.deleted == 0)
            .order_by(AgentConversationDO.last_time.desc().nulls_last(), AgentConversationDO.id.desc())
        )).scalars().all()

        turns = await self._count_turns(user_id, [r.conversation_id for r in rows])
        return [
            AgentConversationVO(
                conversationId=r.conversation_id,
                title=r.title,
                lastTime=r.last_time,
                turns=turns.get(r.conversation_id, 0),
            )
            for r in rows
        ]

    async def _count_turns(self, user_id: str, conversation_ids: list[str]) -> dict[str, int]:
        """一次 groupBy 拿全轮次，避免逐会话查一次的 N+1"""
        if not conversation_ids:
            return {}
        rows = (await self.db.execute(
            select(AgentMessageDO.conversation_id, func.count(AgentMessageDO.id))
            .where(
                AgentMessageDO.user_id == user_id,
                AgentMessageDO.deleted == 0,
                AgentMessageDO.role == ROLE_USER,
                AgentMessageDO.conversation_id.in_(conversation_ids),
            )
            .group_by(AgentMessageDO.conversation_id)
        )).all()
        return {str(cid): int(count) for cid, count in rows}

    async def list_messages(self, user_id: str, conversation_id: str) -> list[AgentMessageVO]:
        """整段对话轨迹，按时间正序"""
        rows = (await self.db.execute(
            select(AgentMessageDO)
            .where(
                AgentMessageDO.user_id == user_id,
                AgentMessageDO.conversation_id == conversation_id,
                AgentMessageDO.deleted == 0,
            )
            .order_by(AgentMessageDO.create_time.asc(), AgentMessageDO.id.asc())
        )).scalars().all()
        return [self._to_vo(r) for r in rows]

    @staticmethod
    def _to_vo(row: AgentMessageDO) -> AgentMessageVO:
        blocks = None
        raw_blocks = row.blocks
        if isinstance(raw_blocks, list) and raw_blocks:
            blocks = [AgentBlock(**b) for b in raw_blocks if isinstance(b, dict)]
        return AgentMessageVO(
            id=row.id,
            role=row.role,
            content=row.content,
            thinkingContent=row.thinking_content,
            blocks=blocks,
            messageStatus=row.message_status,
            createTime=row.create_time,
        )

    async def load_recent_turns(
        self,
        user_id: str,
        conversation_id: str,
        turns: int,
    ) -> list[dict[str, str]]:
        """
        近 N 轮可用问答，供工具侧改写消解指代。

        只认成对且正常收尾的答复：半截被打断的答案带进改写会把模型带偏。
        返回顺序为时间正序，元素形如 ``{"role": ..., "content": ...}``。
        """
        if turns <= 0:
            return []

        window = turns * RECENT_TURN_SCAN_MULTIPLIER + 1
        rows = (await self.db.execute(
            select(AgentMessageDO)
            .where(
                AgentMessageDO.user_id == user_id,
                AgentMessageDO.conversation_id == conversation_id,
                AgentMessageDO.deleted == 0,
                AgentMessageDO.role.in_([ROLE_USER, ROLE_ASSISTANT]),
            )
            .order_by(AgentMessageDO.create_time.desc(), AgentMessageDO.id.desc())
            .limit(window)
        )).scalars().all()

        by_reply_to: dict[str, AgentMessageDO] = {}
        for row in rows:
            if row.role == ROLE_ASSISTANT and self._is_usable_answer(row):
                by_reply_to.setdefault(str(row.reply_to_message_id), row)

        history: list[dict[str, str]] = []
        for row in rows:
            if row.role != ROLE_USER:
                continue
            answer = by_reply_to.get(row.id)
            if answer is None:
                continue
            # 先塞答案再塞提问，最后整体反转 —— 反转后自然是「问在前、答在后」
            history.append({"role": ROLE_ASSISTANT, "content": answer.content or ""})
            history.append({"role": ROLE_USER, "content": row.content or ""})
            if len(history) // 2 >= turns:
                break

        history.reverse()
        return history

    @staticmethod
    def _is_usable_answer(row: AgentMessageDO) -> bool:
        return (
            bool(row.reply_to_message_id)
            and bool((row.content or "").strip())
            and row.message_status != AgentMessageStatus.INTERRUPTED.value
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    async def touch_conversation(self, user_id: str, conversation_id: str, title: str | None) -> None:
        """建会话或刷新活动时间；并发首问只允许一个赢家建行"""
        now = datetime.now()
        clipped = _clip_title(title, TITLE_MAX_LENGTH)

        existing = (await self.db.execute(
            select(AgentConversationDO).where(
                AgentConversationDO.conversation_id == conversation_id,
                AgentConversationDO.user_id == user_id,
            )
        )).scalar_one_or_none()

        if existing is not None:
            existing.last_time = now
            existing.deleted = 0
            existing.update_time = now
            await self.db.flush()
            return

        try:
            self.db.add(AgentConversationDO(
                id=get_snowflake_id_str(),
                conversation_id=conversation_id,
                user_id=user_id,
                title=clipped,
                last_time=now,
                create_time=now,
                update_time=now,
                deleted=0,
            ))
            await self.db.flush()
        except IntegrityError:
            # 撞唯一键说明并发首问里别人先建好了，回滚后改成刷新
            await self.db.rollback()
            winner = (await self.db.execute(
                select(AgentConversationDO).where(
                    AgentConversationDO.conversation_id == conversation_id,
                    AgentConversationDO.user_id == user_id,
                )
            )).scalar_one_or_none()
            if winner is not None:
                winner.last_time = now
                winner.deleted = 0
                winner.update_time = now
                await self.db.flush()

    async def add_user_message(self, user_id: str, conversation_id: str, content: str) -> str:
        now = datetime.now()
        message_id = get_snowflake_id_str()
        self.db.add(AgentMessageDO(
            id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            role=ROLE_USER,
            content=content,
            message_status=AgentMessageStatus.NORMAL.value,
            create_time=now,
            update_time=now,
            deleted=0,
        ))
        await self.db.flush()
        return message_id

    async def add_assistant_message(
        self,
        user_id: str,
        conversation_id: str,
        reply_to_message_id: str | None,
        content: str,
        thinking_content: str | None = None,
        blocks: list[dict[str, Any]] | None = None,
        message_status: AgentMessageStatus | str = AgentMessageStatus.NORMAL,
    ) -> str:
        now = datetime.now()
        message_id = get_snowflake_id_str()
        status = message_status.value if isinstance(message_status, AgentMessageStatus) else message_status
        self.db.add(AgentMessageDO(
            id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            role=ROLE_ASSISTANT,
            content=content,
            thinking_content=thinking_content or None,
            blocks=blocks,
            reply_to_message_id=reply_to_message_id,
            message_status=status,
            create_time=now,
            update_time=now,
            deleted=0,
        ))
        await self.db.flush()
        return message_id

    # ------------------------------------------------------------------
    # 改名与删除
    # ------------------------------------------------------------------

    async def rename(self, user_id: str, conversation_id: str, title: str | None) -> None:
        clipped = _clip_title(title, RENAME_MAX_LENGTH)
        if not (title or "").strip():
            raise ClientException("会话标题不能为空")

        result = await self.db.execute(
            update(AgentConversationDO)
            .where(
                AgentConversationDO.conversation_id == conversation_id,
                AgentConversationDO.user_id == user_id,
                AgentConversationDO.deleted == 0,
            )
            .values(title=clipped, update_time=datetime.now())
        )
        await self.db.flush()
        if result.rowcount == 0:
            raise ClientException("会话不存在或已删除")

    async def delete(self, user_id: str, conversation_id: str) -> None:
        """
        逻辑删会话 + 其下全部消息，并驱逐运行期状态。

        会话行与消息行分两条 UPDATE：一张表一个语义，合成一条反而说不清删了什么。
        """
        now = datetime.now()
        await self.db.execute(
            update(AgentConversationDO)
            .where(
                AgentConversationDO.conversation_id == conversation_id,
                AgentConversationDO.user_id == user_id,
                AgentConversationDO.deleted == 0,
            )
            .values(deleted=1, update_time=now)
        )
        await self.db.execute(
            update(AgentMessageDO)
            .where(
                AgentMessageDO.conversation_id == conversation_id,
                AgentMessageDO.user_id == user_id,
                AgentMessageDO.deleted == 0,
            )
            .values(deleted=1, update_time=now)
        )
        await self.db.flush()
        await self._release_runtime_state(user_id, conversation_id)

    async def batch_delete(self, user_id: str, conversation_ids: list[str] | None) -> int:
        """批量删；逐个走 delete 以便每条都触发运行期状态驱逐"""
        if not conversation_ids:
            return 0
        # 去重保序：前端多选可能带重复项
        unique: list[str] = []
        for cid in conversation_ids:
            if cid and cid not in unique:
                unique.append(cid)

        count = 0
        for cid in unique:
            exists = (await self.db.execute(
                select(func.count(AgentConversationDO.id)).where(
                    AgentConversationDO.conversation_id == cid,
                    AgentConversationDO.user_id == user_id,
                    AgentConversationDO.deleted == 0,
                )
            )).scalar_one()
            if not exists:
                continue
            await self.delete(user_id, cid)
            count += 1
        return count

    # ------------------------------------------------------------------
    # 运行期状态
    # ------------------------------------------------------------------

    async def _release_runtime_state(self, user_id: str, conversation_id: str) -> None:
        """删掉框架侧的会话上下文，避免下次同 ID 复活时读到已删对话的记忆"""
        await self.db.execute(
            delete(AgentStateDO).where(
                AgentStateDO.user_id == user_id,
                AgentStateDO.session_id == conversation_id,
            )
        )
        await self.db.flush()
        if self._on_release_runtime is not None:
            try:
                result = self._on_release_runtime(user_id, conversation_id)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.warning("驱逐 Agent 运行期缓存失败, conversationId: %s, error: %s", conversation_id, e)
