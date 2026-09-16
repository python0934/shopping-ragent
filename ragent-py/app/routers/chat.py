"""
Chat routes — mirrors ConversationController, MessageFeedbackController, RecommendedQuestionController.

Endpoints:
  GET    /conversations
  PUT    /conversations/{conversationId}
  DELETE /conversations/{conversationId}
  GET    /conversations/{conversationId}/messages
  POST   /conversations/messages/{messageId}/feedback
  DELETE /conversations/messages/{messageId}/feedback
  POST   /conversations/messages/{messageId}/recommended-questions
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.result import success

router = APIRouter(tags=["会话管理"])


@router.get("/conversations")
async def list_conversations() -> dict:
    """获取会话列表 — Phase 3 实现"""
    return success()


@router.put("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str) -> dict:
    """重命名会话 — Phase 3 实现"""
    return success()


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict:
    """删除会话 — Phase 3 实现"""
    return success()


@router.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: str) -> dict:
    """获取会话消息列表 — Phase 3 实现"""
    return success()


@router.post("/conversations/messages/{message_id}/feedback")
async def submit_feedback(message_id: str) -> dict:
    """提交消息反馈 — Phase 3 实现"""
    return success()


@router.delete("/conversations/messages/{message_id}/feedback")
async def cancel_feedback(message_id: str) -> dict:
    """取消消息反馈 — Phase 3 实现"""
    return success()


@router.post("/conversations/messages/{message_id}/recommended-questions")
async def generate_recommended_questions(message_id: str) -> dict:
    """生成推荐追问问题 — Phase 3 实现"""
    return success()
