"""
Phase 5 单元测试 — Agent 服务层.

用 SQLite 内存库跑真 SQL，覆盖会话存储、状态存储、提示词解析与流式编排。
JSONB 列在 models 里带 SQLite 变体，故建表与查询都能原样跑。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.enums import AgentMessageStatus
from app.agent.provider import ReActAgentProvider
from app.agent.run_gate import AgentRunGate
from app.agent.state_store import ANONYMOUS_USER, CONTEXT_STATE_KEY, PgAgentStateStore
from app.core.database import Base
from app.core.exceptions import ClientException
from app.core.snowflake import get_snowflake_id_str
from app.core.sse import stream_task_manager
from app.core.user_context import LoginUser, UserContext
from app.models.agent import (
    AgentContextCompactionDO,
    AgentConversationDO,
    AgentMessageDO,
    AgentProfileDO,
    AgentPromptDO,
    AgentStateDO,
)
from app.schemas.agent import AgentTitleRequest
from app.services.agent_conversation_service import (
    RENAME_MAX_LENGTH,
    TITLE_MAX_LENGTH,
    AgentConversationService,
    _clip_title,
)
from app.services.prompt_service import (
    AgentPromptCacheManager,
    AgentPromptResolver,
    AgentPromptSlot,
)

AGENT_TABLES = [
    AgentProfileDO.__table__,
    AgentPromptDO.__table__,
    AgentConversationDO.__table__,
    AgentMessageDO.__table__,
    AgentStateDO.__table__,
    AgentContextCompactionDO.__table__,
]


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session():
    """SQLite 内存库会话，只建 Agent 相关表（其余模型依赖 pgvector）"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=AGENT_TABLES))

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        yield db

    await engine.dispose()


@pytest_asyncio.fixture
async def session_factory(session: AsyncSession):
    """把内存库的会话工厂暴露给编排层替换全局 async_session_factory"""
    return session.bind and async_sessionmaker(session.bind, class_=AsyncSession, expire_on_commit=False)


USER_ID = "1234567890123456789"


async def _seed_conversation(
    db: AsyncSession,
    conversation_id: str,
    title: str = "会话",
    user_id: str = USER_ID,
    last_time: datetime | None = None,
    deleted: int = 0,
) -> None:
    now = datetime.now()
    db.add(AgentConversationDO(
        id=get_snowflake_id_str(),
        conversation_id=conversation_id,
        user_id=user_id,
        title=title,
        last_time=last_time or now,
        create_time=now,
        update_time=now,
        deleted=deleted,
    ))
    await db.flush()


async def _seed_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    message_id: str | None = None,
    reply_to: str | None = None,
    status: str = "NORMAL",
    create_time: datetime | None = None,
    blocks: list[dict[str, Any]] | None = None,
    thinking: str | None = None,
    user_id: str = USER_ID,
) -> str:
    mid = message_id or get_snowflake_id_str()
    now = create_time or datetime.now()
    db.add(AgentMessageDO(
        id=mid,
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
        content=content,
        thinking_content=thinking,
        blocks=blocks,
        reply_to_message_id=reply_to,
        message_status=status,
        create_time=now,
        update_time=now,
        deleted=0,
    ))
    await db.flush()
    return mid


async def _seed_turn(
    db: AsyncSession,
    conversation_id: str,
    question: str,
    answer: str,
    offset_seconds: int = 0,
    answer_status: str = "NORMAL",
) -> tuple[str, str]:
    """播一整轮问答，时间戳按偏移递增以便断言顺序"""
    base = datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=offset_seconds)
    qid = await _seed_message(db, conversation_id, "user", question, create_time=base)
    aid = await _seed_message(
        db, conversation_id, "assistant", answer,
        reply_to=qid, status=answer_status, create_time=base + timedelta(seconds=1),
    )
    return qid, aid


# ---------------------------------------------------------------------------
# 标题裁剪
# ---------------------------------------------------------------------------

class TestClipTitle:

    def test_clips_to_limit(self):
        assert len(_clip_title("a" * 100, TITLE_MAX_LENGTH)) == TITLE_MAX_LENGTH

    def test_strips_whitespace(self):
        assert _clip_title("  标题  ", TITLE_MAX_LENGTH) == "标题"

    def test_falls_back_when_blank(self):
        assert _clip_title("   ", TITLE_MAX_LENGTH) == "新对话"
        assert _clip_title(None, TITLE_MAX_LENGTH) == "新对话"

    def test_custom_fallback(self):
        assert _clip_title("", 10, "默认") == "默认"

    def test_rename_limit_is_wider_than_auto_title(self):
        """手动改名放宽到列宽，自动标题只取列表放得下的长度"""
        assert RENAME_MAX_LENGTH > TITLE_MAX_LENGTH


# ---------------------------------------------------------------------------
# 会话存储 — 写入
# ---------------------------------------------------------------------------

class TestConversationWrite:

    @pytest.mark.asyncio
    async def test_touch_creates_conversation(self, session: AsyncSession):
        service = AgentConversationService(session)
        await service.touch_conversation(USER_ID, "conv-1", "首个问题")

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()

        assert row.user_id == USER_ID
        assert row.title == "首个问题"
        assert row.deleted == 0
        assert row.last_time is not None

    @pytest.mark.asyncio
    async def test_touch_clips_long_title(self, session: AsyncSession):
        await AgentConversationService(session).touch_conversation(USER_ID, "conv-1", "长" * 200)

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        assert len(row.title) == TITLE_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_touch_updates_last_time_without_retitling(self, session: AsyncSession):
        """续问只刷活动时间：标题是首问定下的，不该被后续问题覆盖"""
        service = AgentConversationService(session)
        await service.touch_conversation(USER_ID, "conv-1", "首问")
        first = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        original_time = first.last_time

        await asyncio.sleep(0.01)
        await service.touch_conversation(USER_ID, "conv-1", "第二问")

        assert first.title == "首问"
        assert first.last_time >= original_time

    @pytest.mark.asyncio
    async def test_touch_revives_soft_deleted_conversation(self, session: AsyncSession):
        """同 ID 复活时要把删除标记抹掉，否则列表里查不到但消息又能写"""
        await _seed_conversation(session, "conv-1", deleted=1)

        await AgentConversationService(session).touch_conversation(USER_ID, "conv-1", "回来了")

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        assert row.deleted == 0

    @pytest.mark.asyncio
    async def test_add_user_message(self, session: AsyncSession):
        message_id = await AgentConversationService(session).add_user_message(USER_ID, "conv-1", "问题")

        row = (await session.execute(
            select(AgentMessageDO).where(AgentMessageDO.id == message_id)
        )).scalar_one()
        assert row.role == "user"
        assert row.content == "问题"
        assert row.message_status == "NORMAL"
        assert row.reply_to_message_id is None

    @pytest.mark.asyncio
    async def test_add_assistant_message_links_reply(self, session: AsyncSession):
        message_id = await AgentConversationService(session).add_assistant_message(
            USER_ID, "conv-1", "user-msg-1", "答案",
            thinking_content="思考",
            blocks=[{"kind": "answer", "text": "答案"}],
            message_status=AgentMessageStatus.INTERRUPTED,
        )

        row = (await session.execute(
            select(AgentMessageDO).where(AgentMessageDO.id == message_id)
        )).scalar_one()
        assert row.role == "assistant"
        assert row.reply_to_message_id == "user-msg-1"
        assert row.thinking_content == "思考"
        assert row.blocks == [{"kind": "answer", "text": "答案"}]
        assert row.message_status == "INTERRUPTED"

    @pytest.mark.asyncio
    async def test_add_assistant_message_accepts_plain_status_string(self, session: AsyncSession):
        message_id = await AgentConversationService(session).add_assistant_message(
            USER_ID, "conv-1", None, "答案", message_status="INTERRUPTED",
        )
        row = (await session.execute(
            select(AgentMessageDO).where(AgentMessageDO.id == message_id)
        )).scalar_one()
        assert row.message_status == "INTERRUPTED"

    @pytest.mark.asyncio
    async def test_blank_thinking_stored_as_null(self, session: AsyncSession):
        message_id = await AgentConversationService(session).add_assistant_message(
            USER_ID, "conv-1", None, "答案", thinking_content="",
        )
        row = (await session.execute(
            select(AgentMessageDO).where(AgentMessageDO.id == message_id)
        )).scalar_one()
        assert row.thinking_content is None


# ---------------------------------------------------------------------------
# 会话存储 — 查询
# ---------------------------------------------------------------------------

class TestConversationQuery:

    @pytest.mark.asyncio
    async def test_list_conversations_orders_by_last_time_desc(self, session: AsyncSession):
        base = datetime(2026, 1, 1, 12, 0, 0)
        await _seed_conversation(session, "old", last_time=base)
        await _seed_conversation(session, "new", last_time=base + timedelta(hours=1))

        rows = await AgentConversationService(session).list_conversations(USER_ID)
        assert [r.conversationId for r in rows] == ["new", "old"]

    @pytest.mark.asyncio
    async def test_list_conversations_excludes_deleted(self, session: AsyncSession):
        await _seed_conversation(session, "kept")
        await _seed_conversation(session, "dropped", deleted=1)

        rows = await AgentConversationService(session).list_conversations(USER_ID)
        assert [r.conversationId for r in rows] == ["kept"]

    @pytest.mark.asyncio
    async def test_list_conversations_scoped_to_user(self, session: AsyncSession):
        await _seed_conversation(session, "mine", user_id=USER_ID)
        await _seed_conversation(session, "theirs", user_id="other-user")

        rows = await AgentConversationService(session).list_conversations(USER_ID)
        assert [r.conversationId for r in rows] == ["mine"]

    @pytest.mark.asyncio
    async def test_list_conversations_counts_turns_by_user_messages(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        await _seed_turn(session, "conv-1", "问一", "答一", 0)
        await _seed_turn(session, "conv-1", "问二", "答二", 10)

        rows = await AgentConversationService(session).list_conversations(USER_ID)
        assert rows[0].turns == 2

    @pytest.mark.asyncio
    async def test_list_conversations_turns_ignore_other_conversations(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        await _seed_conversation(session, "conv-2")
        await _seed_turn(session, "conv-1", "问", "答", 0)
        await _seed_turn(session, "conv-2", "问", "答", 10)
        await _seed_turn(session, "conv-2", "问", "答", 20)

        rows = {r.conversationId: r.turns for r in await AgentConversationService(session).list_conversations(USER_ID)}
        assert rows == {"conv-1": 1, "conv-2": 2}

    @pytest.mark.asyncio
    async def test_list_conversations_empty(self, session: AsyncSession):
        assert await AgentConversationService(session).list_conversations(USER_ID) == []

    @pytest.mark.asyncio
    async def test_list_messages_orders_chronologically(self, session: AsyncSession):
        await _seed_turn(session, "conv-1", "问一", "答一", 0)
        await _seed_turn(session, "conv-1", "问二", "答二", 10)

        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert [r.content for r in rows] == ["问一", "答一", "问二", "答二"]

    @pytest.mark.asyncio
    async def test_list_messages_excludes_deleted(self, session: AsyncSession):
        mid = await _seed_message(session, "conv-1", "user", "将被删")
        await _seed_message(session, "conv-1", "user", "保留")
        row = (await session.execute(select(AgentMessageDO).where(AgentMessageDO.id == mid))).scalar_one()
        row.deleted = 1
        await session.flush()

        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert [r.content for r in rows] == ["保留"]

    @pytest.mark.asyncio
    async def test_list_messages_parses_blocks(self, session: AsyncSession):
        blocks = [
            {"kind": "reasoning", "text": "想"},
            {"kind": "tool", "name": "search_knowledge", "displayName": "知识库检索", "status": "done"},
            {"kind": "answer", "text": "答"},
        ]
        await _seed_message(session, "conv-1", "assistant", "答", blocks=blocks, thinking="想")

        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert rows[0].thinkingContent == "想"
        assert [b.kind for b in rows[0].blocks] == ["reasoning", "tool", "answer"]
        assert rows[0].blocks[1].displayName == "知识库检索"

    @pytest.mark.asyncio
    async def test_list_messages_null_blocks_stay_none(self, session: AsyncSession):
        await _seed_message(session, "conv-1", "user", "问")
        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert rows[0].blocks is None

    @pytest.mark.asyncio
    async def test_list_messages_empty_blocks_stay_none(self, session: AsyncSession):
        """空轨迹落库后不该被还原成一个空数组，前端会多渲染一个空气泡"""
        await _seed_message(session, "conv-1", "assistant", "答", blocks=[])
        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert rows[0].blocks is None

    @pytest.mark.asyncio
    async def test_list_messages_carries_status_and_time(self, session: AsyncSession):
        at = datetime(2026, 3, 4, 5, 6, 7)
        await _seed_message(session, "conv-1", "assistant", "半截", status="INTERRUPTED", create_time=at)

        rows = await AgentConversationService(session).list_messages(USER_ID, "conv-1")
        assert rows[0].messageStatus == "INTERRUPTED"
        assert rows[0].createTime == at


# ---------------------------------------------------------------------------
# 会话存储 — 近期轮次
# ---------------------------------------------------------------------------

class TestLoadRecentTurns:

    @pytest.mark.asyncio
    async def test_returns_pairs_in_chronological_order(self, session: AsyncSession):
        await _seed_turn(session, "conv-1", "问一", "答一", 0)
        await _seed_turn(session, "conv-1", "问二", "答二", 10)

        history = await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 2)
        assert history == [
            {"role": "user", "content": "问一"},
            {"role": "assistant", "content": "答一"},
            {"role": "user", "content": "问二"},
            {"role": "assistant", "content": "答二"},
        ]

    @pytest.mark.asyncio
    async def test_limits_to_requested_turns(self, session: AsyncSession):
        for index in range(5):
            await _seed_turn(session, "conv-1", f"问{index}", f"答{index}", index * 10)

        history = await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 2)
        assert len(history) == 4
        # 取的是最近两轮，不是最早两轮
        assert history[0]["content"] == "问3"
        assert history[-1]["content"] == "答4"

    @pytest.mark.asyncio
    async def test_zero_turns_returns_empty(self, session: AsyncSession):
        await _seed_turn(session, "conv-1", "问", "答", 0)
        assert await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 0) == []

    @pytest.mark.asyncio
    async def test_skips_interrupted_answers(self, session: AsyncSession):
        """半截被打断的答案带进改写会把模型带偏"""
        await _seed_turn(session, "conv-1", "问一", "答一", 0, answer_status="INTERRUPTED")
        await _seed_turn(session, "conv-1", "问二", "答二", 10)

        history = await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 5)
        assert [m["content"] for m in history] == ["问二", "答二"]

    @pytest.mark.asyncio
    async def test_skips_answers_without_reply_link(self, session: AsyncSession):
        await _seed_message(session, "conv-1", "user", "孤问")
        await _seed_message(session, "conv-1", "assistant", "孤答", reply_to=None)

        assert await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 5) == []

    @pytest.mark.asyncio
    async def test_skips_blank_answers(self, session: AsyncSession):
        await _seed_turn(session, "conv-1", "问", "   ", 0)
        assert await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 5) == []

    @pytest.mark.asyncio
    async def test_scoped_to_conversation(self, session: AsyncSession):
        await _seed_turn(session, "conv-1", "问一", "答一", 0)
        await _seed_turn(session, "conv-2", "问二", "答二", 10)

        history = await AgentConversationService(session).load_recent_turns(USER_ID, "conv-1", 5)
        assert [m["content"] for m in history] == ["问一", "答一"]

    @pytest.mark.asyncio
    async def test_empty_conversation(self, session: AsyncSession):
        assert await AgentConversationService(session).load_recent_turns(USER_ID, "conv-none", 2) == []


# ---------------------------------------------------------------------------
# 会话存储 — 改名与删除
# ---------------------------------------------------------------------------

class TestRenameAndDelete:

    @pytest.mark.asyncio
    async def test_rename_updates_title(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1", "旧标题")

        await AgentConversationService(session).rename(USER_ID, "conv-1", "新标题")

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        assert row.title == "新标题"

    @pytest.mark.asyncio
    async def test_rename_clips_to_column_width(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        await AgentConversationService(session).rename(USER_ID, "conv-1", "长" * 500)

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        assert len(row.title) == RENAME_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_rename_rejects_blank(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")

        with pytest.raises(ClientException):
            await AgentConversationService(session).rename(USER_ID, "conv-1", "   ")

    @pytest.mark.asyncio
    async def test_rename_rejects_missing_conversation(self, session: AsyncSession):
        with pytest.raises(ClientException):
            await AgentConversationService(session).rename(USER_ID, "conv-none", "标题")

    @pytest.mark.asyncio
    async def test_rename_scoped_to_owner(self, session: AsyncSession):
        """别人的会话不能改，越权要报「不存在」而不是静默成功"""
        await _seed_conversation(session, "conv-1", user_id="other")

        with pytest.raises(ClientException):
            await AgentConversationService(session).rename(USER_ID, "conv-1", "标题")

    @pytest.mark.asyncio
    async def test_delete_soft_deletes_conversation_and_messages(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        mid = await _seed_message(session, "conv-1", "user", "问")

        await AgentConversationService(session).delete(USER_ID, "conv-1")

        conversation = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        message = (await session.execute(
            select(AgentMessageDO).where(AgentMessageDO.id == mid)
        )).scalar_one()

        assert conversation.deleted == 1
        assert message.deleted == 1

    @pytest.mark.asyncio
    async def test_delete_purges_runtime_state(self, session: AsyncSession):
        """框架侧上下文必须一起清掉，否则同 ID 复活时会读到已删对话的记忆"""
        await _seed_conversation(session, "conv-1")
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, [{"role": "user"}])

        await AgentConversationService(session).delete(USER_ID, "conv-1")

        assert await store.get_list(USER_ID, "conv-1", CONTEXT_STATE_KEY) == []

    @pytest.mark.asyncio
    async def test_delete_invokes_release_callback(self, session: AsyncSession):
        released: list[tuple[str, str]] = []
        await _seed_conversation(session, "conv-1")

        service = AgentConversationService(
            session, on_release_runtime=lambda u, c: released.append((u, c)),
        )
        await service.delete(USER_ID, "conv-1")

        assert released == [(USER_ID, "conv-1")]

    @pytest.mark.asyncio
    async def test_delete_awaits_async_release_callback(self, session: AsyncSession):
        released: list[str] = []

        async def on_release(user_id: str, conversation_id: str) -> None:
            released.append(conversation_id)

        await _seed_conversation(session, "conv-1")
        await AgentConversationService(session, on_release_runtime=on_release).delete(USER_ID, "conv-1")

        assert released == ["conv-1"]

    @pytest.mark.asyncio
    async def test_delete_swallows_release_callback_error(self, session: AsyncSession):
        """驱逐缓存失败不该让删除事务回滚：库里的行已经该删了"""
        def boom(user_id: str, conversation_id: str) -> None:
            raise RuntimeError("redis down")

        await _seed_conversation(session, "conv-1")
        await AgentConversationService(session, on_release_runtime=boom).delete(USER_ID, "conv-1")

        row = (await session.execute(
            select(AgentConversationDO).where(AgentConversationDO.conversation_id == "conv-1")
        )).scalar_one()
        assert row.deleted == 1

    @pytest.mark.asyncio
    async def test_delete_leaves_other_conversations_alone(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        await _seed_conversation(session, "conv-2")
        await _seed_message(session, "conv-2", "user", "别删我")

        await AgentConversationService(session).delete(USER_ID, "conv-1")

        rows = await AgentConversationService(session).list_conversations(USER_ID)
        assert [r.conversationId for r in rows] == ["conv-2"]

    @pytest.mark.asyncio
    async def test_batch_delete_returns_affected_count(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")
        await _seed_conversation(session, "conv-2")

        count = await AgentConversationService(session).batch_delete(USER_ID, ["conv-1", "conv-2"])
        assert count == 2
        assert await AgentConversationService(session).list_conversations(USER_ID) == []

    @pytest.mark.asyncio
    async def test_batch_delete_skips_missing_ids(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1")

        count = await AgentConversationService(session).batch_delete(USER_ID, ["conv-1", "conv-ghost"])
        assert count == 1

    @pytest.mark.asyncio
    async def test_batch_delete_dedupes_ids(self, session: AsyncSession):
        """前端多选可能带重复项，重复删同一会话不该被算成两条"""
        await _seed_conversation(session, "conv-1")

        assert await AgentConversationService(session).batch_delete(USER_ID, ["conv-1", "conv-1"]) == 1

    @pytest.mark.asyncio
    async def test_batch_delete_skips_other_users_conversations(self, session: AsyncSession):
        await _seed_conversation(session, "conv-1", user_id="other")

        assert await AgentConversationService(session).batch_delete(USER_ID, ["conv-1"]) == 0

    @pytest.mark.asyncio
    async def test_batch_delete_empty_input(self, session: AsyncSession):
        assert await AgentConversationService(session).batch_delete(USER_ID, []) == 0
        assert await AgentConversationService(session).batch_delete(USER_ID, None) == 0


# ---------------------------------------------------------------------------
# 状态存储（真 SQL）
# ---------------------------------------------------------------------------

class TestStateStoreOnSqlite:

    @pytest.mark.asyncio
    async def test_save_then_get_roundtrip(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, {"a": 1})

        assert await store.get(USER_ID, "conv-1", CONTEXT_STATE_KEY) == {"a": 1}

    @pytest.mark.asyncio
    async def test_save_overwrites_existing_payload(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, {"v": 1})
        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, {"v": 2})

        assert await store.get(USER_ID, "conv-1", CONTEXT_STATE_KEY) == {"v": 2}

        rows = (await session.execute(select(AgentStateDO))).scalars().all()
        assert len(rows) == 1, "覆盖必须走同一行，不能追加"

    @pytest.mark.asyncio
    async def test_save_list_roundtrip(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save_list(USER_ID, "conv-1", CONTEXT_STATE_KEY, [{"i": 1}, {"i": 2}])

        assert await store.get_list(USER_ID, "conv-1", CONTEXT_STATE_KEY) == [{"i": 1}, {"i": 2}]

    @pytest.mark.asyncio
    async def test_save_none_user_uses_anonymous_sentinel(self, session: AsyncSession):
        """PG 主键列不可为空，匿名访问必须落到哨兵上"""
        store = PgAgentStateStore(session)
        await store.save(None, "conv-1", CONTEXT_STATE_KEY, {"a": 1})

        row = (await session.execute(select(AgentStateDO))).scalar_one()
        assert row.user_id == ANONYMOUS_USER
        assert await store.get(None, "conv-1", CONTEXT_STATE_KEY) == {"a": 1}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, session: AsyncSession):
        assert await PgAgentStateStore(session).get(USER_ID, "conv-x", CONTEXT_STATE_KEY) is None

    @pytest.mark.asyncio
    async def test_keys_are_isolated(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", "k1", {"a": 1})
        await store.save(USER_ID, "conv-1", "k2", {"b": 2})

        assert await store.get(USER_ID, "conv-1", "k1") == {"a": 1}
        assert await store.get(USER_ID, "conv-1", "k2") == {"b": 2}

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, {"a": 1})
        await store.save(USER_ID, "conv-2", CONTEXT_STATE_KEY, {"a": 2})

        assert await store.get(USER_ID, "conv-1", CONTEXT_STATE_KEY) == {"a": 1}
        assert await store.get(USER_ID, "conv-2", CONTEXT_STATE_KEY) == {"a": 2}

    @pytest.mark.asyncio
    async def test_exists(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        assert not await store.exists(USER_ID, "conv-1")

        await store.save(USER_ID, "conv-1", CONTEXT_STATE_KEY, {})
        assert await store.exists(USER_ID, "conv-1")

    @pytest.mark.asyncio
    async def test_list_session_ids(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        for cid in ("conv-a", "conv-b", "conv-c"):
            await store.save(USER_ID, cid, CONTEXT_STATE_KEY, {})
        await store.save("other-user", "conv-z", CONTEXT_STATE_KEY, {})

        assert sorted(await store.list_session_ids(USER_ID)) == ["conv-a", "conv-b", "conv-c"]

    @pytest.mark.asyncio
    async def test_delete_key_removes_only_that_key(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", "k1", {"a": 1})
        await store.save(USER_ID, "conv-1", "k2", {"b": 2})

        await store.delete_key(USER_ID, "conv-1", "k1")

        assert await store.get(USER_ID, "conv-1", "k1") is None
        assert await store.get(USER_ID, "conv-1", "k2") == {"b": 2}

    @pytest.mark.asyncio
    async def test_delete_removes_whole_session(self, session: AsyncSession):
        store = PgAgentStateStore(session)
        await store.save(USER_ID, "conv-1", "k1", {"a": 1})
        await store.save(USER_ID, "conv-1", "k2", {"b": 2})
        await store.save(USER_ID, "conv-2", "k1", {"c": 3})

        await store.delete(USER_ID, "conv-1")

        assert not await store.exists(USER_ID, "conv-1")
        assert await store.exists(USER_ID, "conv-2")


# ---------------------------------------------------------------------------
# 提示词解析
# ---------------------------------------------------------------------------

class _FakeCacheManager:
    def __init__(self, cached: dict[str, str] | None = None) -> None:
        self.cached = cached
        self.saved: list[dict[str, str]] = []
        self.cleared = 0

    async def get_from_cache(self) -> dict[str, str] | None:
        return self.cached

    async def save_to_cache(self, prompts: dict[str, str]) -> None:
        self.saved.append(dict(prompts))

    async def clear_cache(self) -> None:
        self.cleared += 1


async def _seed_agent(
    db: AsyncSession,
    agent_id: str,
    name: str,
    builtin: int = 0,
    active: int = 0,
    prompts: dict[str, str] | None = None,
) -> None:
    now = datetime.now()
    db.add(AgentProfileDO(
        id=agent_id, name=name, description="d", avatar=None,
        builtin=builtin, active=active,
        create_time=now, update_time=now, deleted=0,
    ))
    for slot_key, content in (prompts or {}).items():
        db.add(AgentPromptDO(
            id=get_snowflake_id_str(), agent_id=agent_id, slot_key=slot_key, content=content,
            create_time=now, update_time=now, deleted=0,
        ))
    await db.flush()


class TestPromptResolver:

    @pytest.mark.asyncio
    async def test_builtin_baseline_is_used_when_no_active_agent(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, active=0, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "内置人设",
        })

        resolver = AgentPromptResolver(session)
        assert await resolver.resolve(AgentPromptSlot.AGENT_MAIN) == "内置人设"

    @pytest.mark.asyncio
    async def test_active_agent_overrides_builtin(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "内置人设",
        })
        await _seed_agent(session, "a2", "激活", active=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "激活人设",
        })

        assert await AgentPromptResolver(session).resolve(AgentPromptSlot.AGENT_MAIN) == "激活人设"

    @pytest.mark.asyncio
    async def test_active_agent_falls_back_per_slot(self, session: AsyncSession):
        """激活体只配了一个槽位，其余槽位仍回落到内置基线"""
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "内置人设",
            AgentPromptSlot.KB_ANSWER.value: "内置答案模板",
        })
        await _seed_agent(session, "a2", "激活", active=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "激活人设",
        })

        resolver = AgentPromptResolver(session)
        assert await resolver.resolve(AgentPromptSlot.AGENT_MAIN) == "激活人设"
        assert await resolver.resolve(AgentPromptSlot.KB_ANSWER) == "内置答案模板"

    @pytest.mark.asyncio
    async def test_blank_prompt_does_not_override(self, session: AsyncSession):
        """空串不是有效配置，覆盖上去等于把槽位清空"""
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "内置人设",
        })
        await _seed_agent(session, "a2", "激活", active=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "   ",
        })

        assert await AgentPromptResolver(session).resolve(AgentPromptSlot.AGENT_MAIN) == "内置人设"

    @pytest.mark.asyncio
    async def test_resolve_unconfigured_slot_returns_empty_string(self, session: AsyncSession):
        assert await AgentPromptResolver(session).resolve(AgentPromptSlot.AGENT_MAIN) == ""

    @pytest.mark.asyncio
    async def test_resolve_none_slot_returns_empty_string(self, session: AsyncSession):
        assert await AgentPromptResolver(session).resolve(None) == ""

    @pytest.mark.asyncio
    async def test_render_fills_placeholders(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.KB_ANSWER.value: "问题：{question}\n证据：{kbContext}",
        })

        rendered = await AgentPromptResolver(session).render(
            AgentPromptSlot.KB_ANSWER, {"question": "今天几号", "kbContext": "证据文本"},
        )
        assert rendered == "问题：今天几号\n证据：证据文本"

    @pytest.mark.asyncio
    async def test_render_cleans_up_excess_blank_lines(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "甲\n\n\n\n乙",
        })

        assert await AgentPromptResolver(session).render(AgentPromptSlot.AGENT_MAIN) == "甲\n\n乙"

    @pytest.mark.asyncio
    async def test_resolve_all_omits_missing_slots(self, session: AsyncSession):
        """Java 契约：缺失的槽位不出现在 map 中，由 resolve() 兜空串"""
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "人设",
        })

        resolver = AgentPromptResolver(session)
        prompts = await resolver.resolve_all()

        assert prompts == {AgentPromptSlot.AGENT_MAIN.value: "人设"}
        assert await resolver.resolve(AgentPromptSlot.KB_ANSWER) == ""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_db(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "库里的",
        })
        cache = _FakeCacheManager({AgentPromptSlot.AGENT_MAIN.value: "缓存里的"})

        resolver = AgentPromptResolver(session, cache)
        assert await resolver.resolve(AgentPromptSlot.AGENT_MAIN) == "缓存里的"

    @pytest.mark.asyncio
    async def test_cache_miss_populates_cache(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "人设",
        })
        cache = _FakeCacheManager(None)

        await AgentPromptResolver(session, cache).resolve(AgentPromptSlot.AGENT_MAIN)
        assert cache.saved and cache.saved[0][AgentPromptSlot.AGENT_MAIN.value] == "人设"

    @pytest.mark.asyncio
    async def test_cache_manager_survives_redis_failure(self):
        """缓存炸了不能连累提示词解析，回落读库即可"""
        class _BoomRedis:
            async def get(self, *args: Any) -> Any:
                raise RuntimeError("redis down")

            async def set(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("redis down")

            async def delete(self, *args: Any) -> Any:
                raise RuntimeError("redis down")

        manager = AgentPromptCacheManager(_BoomRedis())
        assert await manager.get_from_cache() is None
        await manager.save_to_cache({"A": "B"})
        await manager.clear_cache()

    @pytest.mark.asyncio
    async def test_load_own_prompts_scoped_to_agent(self, session: AsyncSession):
        await _seed_agent(session, "a1", "甲", prompts={AgentPromptSlot.AGENT_MAIN.value: "甲的人设"})
        await _seed_agent(session, "a2", "乙", prompts={AgentPromptSlot.AGENT_MAIN.value: "乙的人设"})

        prompts = await AgentPromptResolver(session).load_own_prompts("a2")
        assert prompts.get(AgentPromptSlot.AGENT_MAIN.value) == "乙的人设"


# ---------------------------------------------------------------------------
# 请求体 schema
# ---------------------------------------------------------------------------

class TestRequestSchemas:

    def test_title_request_accepts_camel_and_snake(self):
        assert AgentTitleRequest.model_validate({"title": "甲"}).title == "甲"

    def test_title_request_defaults_to_none(self):
        assert AgentTitleRequest.model_validate({}).title is None

    def test_batch_delete_request_parses_ids(self):
        from app.schemas.agent import AgentBatchDeleteRequest

        assert AgentBatchDeleteRequest.model_validate({"ids": ["a", "b"]}).ids == ["a", "b"]
        assert AgentBatchDeleteRequest.model_validate({}).ids is None


# ---------------------------------------------------------------------------
# 并发闸门（真 Redis 桩）
# ---------------------------------------------------------------------------

class _GateRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def set(self, key: str, value: str, nx: bool = False, px: int | None = None) -> bool:
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def eval(self, script: str, numkeys: int, key: str, expected: str) -> int:
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0


class TestRunGateIntegration:

    @pytest.mark.asyncio
    async def test_gate_blocks_second_run_for_same_user(self):
        redis = _GateRedis()
        gate = AgentRunGate(redis, sse_timeout_ms=1_000)

        await gate.acquire(USER_ID, "task-1", "conv-1")
        with pytest.raises(ClientException):
            await gate.acquire(USER_ID, "task-2", "conv-2")

    @pytest.mark.asyncio
    async def test_gate_allows_next_run_after_release(self):
        redis = _GateRedis()
        gate = AgentRunGate(redis, sse_timeout_ms=1_000)

        release = await gate.acquire(USER_ID, "task-1", "conv-1")
        await release()
        await gate.acquire(USER_ID, "task-2", "conv-1")


# ---------------------------------------------------------------------------
# 装配器
# ---------------------------------------------------------------------------

class TestReActAgentProvider:

    def test_mcp_tool_count_zero_without_registries(self):
        assert ReActAgentProvider().mcp_tool_count() == 0

    def test_provider_constructs_default_model_client(self):
        provider = ReActAgentProvider()
        assert provider.model_client is not None

    @pytest.mark.asyncio
    async def test_build_assembles_agent_and_catalog(self, session: AsyncSession):
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "你是助手",
            AgentPromptSlot.KNOWLEDGE_TOOL_DESCRIPTION.value: "检索知识库",
        })

        agent, catalog = await ReActAgentProvider().build(session, USER_ID, "conv-1")

        assert agent.system_prompt == "你是助手"
        assert agent.max_iters > 0
        assert "search_knowledge" in agent.tools_by_name
        assert catalog.display_name_of("search_knowledge") == "知识库检索"

    @pytest.mark.asyncio
    async def test_build_fails_when_tool_description_missing(self, session: AsyncSession):
        """知识库工具描述缺失时装配就该炸，而不是让模型拿着空描述瞎调"""
        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "你是助手",
        })

        with pytest.raises(RuntimeError):
            await ReActAgentProvider().build(session, USER_ID, "conv-1")

    @pytest.mark.asyncio
    async def test_build_disables_memory_when_flag_off(self, session: AsyncSession, monkeypatch: Any):
        from app.agent import memory as memory_module

        await _seed_agent(session, "a1", "内置", builtin=1, prompts={
            AgentPromptSlot.AGENT_MAIN.value: "你是助手",
            AgentPromptSlot.KNOWLEDGE_TOOL_DESCRIPTION.value: "检索知识库",
        })
        monkeypatch.setattr(
            memory_module.AgentMemoryBudget, "from_settings",
            classmethod(lambda cls: cls(enabled=False, context_window_chars=1000)),
        )

        agent, _ = await ReActAgentProvider().build(session, USER_ID, "conv-1")
        assert agent.memory is None
