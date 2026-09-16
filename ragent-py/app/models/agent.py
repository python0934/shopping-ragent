"""
Agent domain ORM models — AgentProfile, AgentPrompt, AgentConversation, etc.

Mirrors Java entities:
  - rag.dao.entity.AgentProfileDO
  - rag.dao.entity.AgentPromptDO
  - agent.dao.entity.AgentConversationDO
  - agent.dao.entity.AgentMessageDO
  - (t_agent_state — no Java DO, defined in schema only)
  - agent.dao.entity.AgentContextCompactionDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Integer, JSON, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# JSONB 带 SQLite 变体：生产走 PG 仍是 JSONB，测试可在 SQLite 内存库上跑真 SQL
JsonType = JSONB().with_variant(JSON(), "sqlite")


class AgentProfileDO(Base):
    """智能体人设配置表 — t_agent_profile"""

    __tablename__ = "t_agent_profile"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="智能体名称，唯一")
    description: Mapped[str | None] = mapped_column(String(512), comment="智能体描述")
    avatar: Mapped[str | None] = mapped_column(String(32), comment="头像预设标识")
    builtin: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, server_default="0", comment="是否内置 0:否 1:是")
    active: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, server_default="0", comment="是否激活 0:否 1:是")
    create_by: Mapped[str | None] = mapped_column(String(20), comment="创建人")
    update_by: Mapped[str | None] = mapped_column(String(20), comment="更新人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class AgentPromptDO(Base):
    """智能体提示词槽位表 — t_agent_prompt"""

    __tablename__ = "t_agent_prompt"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    agent_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="所属智能体ID")
    slot_key: Mapped[str] = mapped_column(String(64), nullable=False, comment="槽位标识")
    content: Mapped[str | None] = mapped_column(Text, comment="提示词全文")
    create_by: Mapped[str | None] = mapped_column(String(20), comment="创建人")
    update_by: Mapped[str | None] = mapped_column(String(20), comment="更新人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class AgentConversationDO(Base):
    """Agent 会话列表 — t_agent_conversation"""

    __tablename__ = "t_agent_conversation"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    title: Mapped[str] = mapped_column(String(128), nullable=False, comment="会话标题")
    last_time: Mapped[datetime | None] = mapped_column(comment="最后活动时间")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class AgentMessageDO(Base):
    """Agent 消息记录 — t_agent_message"""

    __tablename__ = "t_agent_message"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="角色 user/assistant")
    content: Mapped[str | None] = mapped_column(Text, comment="消息正文")
    thinking_content: Mapped[str | None] = mapped_column(Text, comment="思考内容")
    blocks: Mapped[dict | None] = mapped_column(JsonType, comment="运行轨迹块（reasoning/answer/tool 有序序列）")
    reply_to_message_id: Mapped[str | None] = mapped_column(String(20), comment="回复的用户消息ID")
    message_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="NORMAL", server_default="'NORMAL'",
        comment="消息终态 NORMAL/INTERRUPTED",
    )
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class AgentStateDO(Base):
    """AgentScope 工作状态存储 — t_agent_state（复合主键，无 deleted）"""

    __tablename__ = "t_agent_state"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="用户ID")
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="会话ID")
    state_key: Mapped[str] = mapped_column(String(64), primary_key=True, comment="状态键")
    payload: Mapped[dict | None] = mapped_column(JsonType, comment="框架自有编码的状态 JSON")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")


class AgentContextCompactionDO(Base):
    """Agent 上下文压缩事件 — t_agent_context_compaction（追加型审计日志）"""

    __tablename__ = "t_agent_context_compaction"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    generation: Mapped[int] = mapped_column(Integer, nullable=False, comment="同一会话内的第几代摘要，从 1 起")
    summary: Mapped[str | None] = mapped_column(Text, comment="本代摘要正文")
    material_msg_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="被换出的原文消息条数")
    material_chars: Mapped[int] = mapped_column(Integer, nullable=False, comment="被换出的原文字符数")
    summary_chars: Mapped[int] = mapped_column(Integer, nullable=False, comment="摘要正文字符数")
    context_chars_before: Mapped[int] = mapped_column(Integer, nullable=False, comment="压缩前上下文总字符数")
    context_chars_after: Mapped[int] = mapped_column(Integer, nullable=False, comment="压缩后上下文总字符数")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
