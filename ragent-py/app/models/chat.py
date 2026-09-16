"""
Chat domain ORM models — Conversation, ConversationSummary, Message, MessageFeedback.

Mirrors Java entities:
  - rag.dao.entity.ConversationDO
  - rag.dao.entity.ConversationSummaryDO
  - rag.dao.entity.ConversationMessageDO
  - rag.dao.entity.MessageFeedbackDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConversationDO(Base):
    """会话列表 — t_conversation"""

    __tablename__ = "t_conversation"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    title: Mapped[str] = mapped_column(String(128), nullable=False, comment="会话名称")
    last_time: Mapped[datetime | None] = mapped_column(comment="最近消息时间")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class ConversationSummaryDO(Base):
    """会话摘要表 — t_conversation_summary"""

    __tablename__ = "t_conversation_summary"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    last_message_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="摘要最后消息ID")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="会话摘要内容")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class MessageDO(Base):
    """会话消息记录表 — t_message"""

    __tablename__ = "t_message"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="角色：user/assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")
    thinking_content: Mapped[str | None] = mapped_column(Text, comment="深度思考内容")
    thinking_duration: Mapped[int | None] = mapped_column(Integer, comment="深度思考耗时（秒）")
    sources: Mapped[dict | None] = mapped_column(JSONB, comment="回答来源")
    recommended_questions: Mapped[dict | None] = mapped_column(JSONB, comment="推荐追问问题")
    retrieved_chunks: Mapped[dict | None] = mapped_column(JSONB, comment="推荐问题 grounding 片段")
    reply_to_message_id: Mapped[str | None] = mapped_column(String(20), comment="当前助手消息对应的用户消息ID")
    message_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="NORMAL", server_default="'NORMAL'",
        comment="消息结束状态：NORMAL/INTERRUPTED/REJECTED",
    )
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class MessageFeedbackDO(Base):
    """会话消息反馈表 — t_message_feedback"""

    __tablename__ = "t_message_feedback"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    message_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="消息ID")
    conversation_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="会话ID")
    user_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="用户ID")
    vote: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="投票 1：赞 -1：踩")
    reason: Mapped[str | None] = mapped_column(String(255), comment="反馈原因")
    comment: Mapped[str | None] = mapped_column(String(1024), comment="反馈评论")
    create_time: Mapped[datetime | None] = mapped_column(nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(nullable=False, comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")
