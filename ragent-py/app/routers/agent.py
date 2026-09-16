"""
Agent routes — mirrors AgentProfileController, IntentTreeController, QueryTermMappingController,
                AgentMetaController, AgentChatController, AgentConversationController, GraphController.

Endpoints:
  # Agent Profile (8)
  GET    /agents
  POST   /agents
  PUT    /agents/{id}
  DELETE /agents/{id}
  POST   /agents/{id}/activate
  GET    /agents/{id}/prompts
  PUT    /agents/{id}/prompts/{slot_key}
  GET    /agents/prompt-slots/{slot_key}/default

  # Intent Tree (7)
  GET    /intent-tree/trees
  POST   /intent-tree
  PUT    /intent-tree/{id}
  DELETE /intent-tree/{id}
  POST   /intent-tree/batch/enable
  POST   /intent-tree/batch/disable
  POST   /intent-tree/batch/delete

  # Query Term Mapping (5)
  GET    /mappings
  GET    /mappings/{id}
  POST   /mappings
  PUT    /mappings/{id}
  DELETE /mappings/{id}

  # Agent Chat (3)
  GET    /agent/v1/meta
  GET    /agent/v1/chat (SSE)
  POST   /agent/v1/stop

  # Agent Conversation (5)
  GET    /agent/v1/conversations
  GET    /agent/v1/conversations/{conversationId}/messages
  PUT    /agent/v1/conversations/{conversationId}/title
  DELETE /agent/v1/conversations/{conversationId}
  POST   /agent/v1/conversations/batch-delete

  # Graph (1)
  GET    /admin/kg/graph
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.provider import ReActAgentProvider
from app.config import settings
from app.core.database import get_db
from app.core.result import success
from app.core.user_context import UserContext
from app.schemas.agent import AgentBatchDeleteRequest, AgentTitleRequest

# Agent 装配器进程内共享：模型客户端与注册表无请求态，重复造只是浪费
_provider = ReActAgentProvider()


def _user_id() -> str:
    return UserContext.require_user().userId

# --- Agent Profile ---
profile_router = APIRouter(tags=["智能体配置"])


@profile_router.get("/agents")
async def list_agents() -> dict:
    """获取智能体列表 — Phase 5 实现"""
    return success()


@profile_router.post("/agents")
async def create_agent() -> dict:
    """创建智能体 — Phase 5 实现"""
    return success()


@profile_router.put("/agents/{agent_id}")
async def update_agent(agent_id: str) -> dict:
    """更新智能体 — Phase 5 实现"""
    return success()


@profile_router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    """删除智能体 — Phase 5 实现"""
    return success()


@profile_router.post("/agents/{agent_id}/activate")
async def activate_agent(agent_id: str) -> dict:
    """激活智能体 — Phase 5 实现"""
    return success()


@profile_router.get("/agents/{agent_id}/prompts")
async def get_prompts(agent_id: str) -> dict:
    """获取智能体提示词配置 — Phase 5 实现"""
    return success()


@profile_router.put("/agents/{agent_id}/prompts/{slot_key}")
async def save_prompt(agent_id: str, slot_key: str) -> dict:
    """保存提示词槽位 — Phase 5 实现"""
    return success()


@profile_router.get("/agents/prompt-slots/{slot_key}/default")
async def default_prompt(slot_key: str) -> dict:
    """获取默认提示词 — Phase 5 实现"""
    return success()


# --- Intent Tree ---
intent_router = APIRouter(tags=["意图树"])


@intent_router.get("/intent-tree/trees")
async def get_intent_tree() -> dict:
    """获取意图树 — Phase 4 实现"""
    return success()


@intent_router.post("/intent-tree")
async def create_intent_node() -> dict:
    """创建意图节点 — Phase 4 实现"""
    return success()


@intent_router.put("/intent-tree/{node_id}")
async def update_intent_node(node_id: str) -> dict:
    """更新意图节点 — Phase 4 实现"""
    return success()


@intent_router.delete("/intent-tree/{node_id}")
async def delete_intent_node(node_id: str) -> dict:
    """删除意图节点 — Phase 4 实现"""
    return success()


@intent_router.post("/intent-tree/batch/enable")
async def batch_enable_nodes() -> dict:
    """批量启用意图节点 — Phase 4 实现"""
    return success()


@intent_router.post("/intent-tree/batch/disable")
async def batch_disable_nodes() -> dict:
    """批量禁用意图节点 — Phase 4 实现"""
    return success()


@intent_router.post("/intent-tree/batch/delete")
async def batch_delete_nodes() -> dict:
    """批量删除意图节点 — Phase 4 实现"""
    return success()


# --- Query Term Mapping ---
mapping_router = APIRouter(tags=["关键词映射"])


@mapping_router.get("/mappings")
async def page_query_mappings() -> dict:
    """关键词映射分页查询 — Phase 4 实现"""
    return success()


@mapping_router.get("/mappings/{mapping_id}")
async def get_mapping(mapping_id: str) -> dict:
    """获取关键词映射详情 — Phase 4 实现"""
    return success()


@mapping_router.post("/mappings")
async def create_mapping() -> dict:
    """创建关键词映射 — Phase 4 实现"""
    return success()


@mapping_router.put("/mappings/{mapping_id}")
async def update_mapping(mapping_id: str) -> dict:
    """更新关键词映射 — Phase 4 实现"""
    return success()


@mapping_router.delete("/mappings/{mapping_id}")
async def delete_mapping(mapping_id: str) -> dict:
    """删除关键词映射 — Phase 4 实现"""
    return success()


# --- Agent Chat ---
agent_chat_router = APIRouter(tags=["Agent 对话"])


@agent_chat_router.get("/agent/v1/meta")
async def agent_meta(db: AsyncSession = Depends(get_db)) -> dict:
    """
    获取 Agent 元信息。

    capabilities / toolProvider 随 MCP 注册表现况浮动：前端据此决定是否展工具面板。
    探活前先刷意图树快照，否则控制台改了节点要等下一次对话才生效。
    """
    await _provider.refresh_intent_snapshot(db)
    mcp_configured = _provider.mcp_tool_count() > 0
    capabilities = ["react", "knowledge-base"]
    if mcp_configured:
        capabilities.append("mcp-tools")

    return success({
        "framework": "AgentScope ReAct",
        "model": settings.agent.chat.model,
        "maxIters": settings.agent.max_iters,
        "capabilities": capabilities,
        "toolProvider": "native + mcp" if mcp_configured else "native",
        "mcpConfigured": mcp_configured,
    })


@agent_chat_router.get("/agent/v1/chat")
async def agent_chat(
    question: str = Query(..., description="用户提问"),
    conversationId: str | None = Query(default=None, description="会话 ID，空则新建"),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Agent 流式对话 (SSE)"""
    from app.services.agent_chat_service import AgentChatService

    service = AgentChatService(db, _provider)
    return await service.stream_chat(_user_id(), question, conversationId)


@agent_chat_router.post("/agent/v1/stop")
async def agent_stop(
    taskId: str = Query(..., description="meta 帧里下发的 taskId"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """停止 Agent 对话"""
    from app.services.agent_chat_service import AgentChatService

    service = AgentChatService(db, _provider)
    await service.stop(_user_id(), taskId)
    return success()


# --- Agent Conversation ---
agent_conv_router = APIRouter(tags=["Agent 会话"])


@agent_conv_router.get("/agent/v1/conversations")
async def list_agent_conversations(db: AsyncSession = Depends(get_db)) -> dict:
    """获取 Agent 会话列表"""
    from app.services.agent_conversation_service import AgentConversationService

    rows = await AgentConversationService(db).list_conversations(_user_id())
    return success([r.model_dump(mode="json") for r in rows])


@agent_conv_router.get("/agent/v1/conversations/{conversation_id}/messages")
async def list_agent_messages(conversation_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """获取 Agent 消息列表（含运行轨迹块）"""
    from app.services.agent_conversation_service import AgentConversationService

    rows = await AgentConversationService(db).list_messages(_user_id(), conversation_id)
    return success([r.model_dump(mode="json", exclude_none=True) for r in rows])


@agent_conv_router.put("/agent/v1/conversations/{conversation_id}/title")
async def rename_agent_conversation(
    conversation_id: str,
    body: AgentTitleRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """重命名 Agent 会话"""
    from app.services.agent_conversation_service import AgentConversationService

    service = AgentConversationService(db)
    await service.rename(_user_id(), conversation_id, body.title)
    await db.commit()
    return success()


@agent_conv_router.delete("/agent/v1/conversations/{conversation_id}")
async def delete_agent_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """删除 Agent 会话：先停流再删，否则流会把已删会话的消息又写回来"""
    from app.services.agent_chat_service import AgentChatService
    from app.services.agent_conversation_service import AgentConversationService

    user_id = _user_id()
    await AgentChatService(db, _provider).stop_by_conversation(user_id, conversation_id)

    service = AgentConversationService(db)
    await service.delete(user_id, conversation_id)
    await db.commit()
    return success()


@agent_conv_router.post("/agent/v1/conversations/batch-delete")
async def batch_delete_agent_conversations(
    body: AgentBatchDeleteRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """批量删除 Agent 会话"""
    from app.services.agent_chat_service import AgentChatService
    from app.services.agent_conversation_service import AgentConversationService

    user_id = _user_id()
    chat_service = AgentChatService(db, _provider)
    for cid in body.ids or []:
        await chat_service.stop_by_conversation(user_id, cid)

    service = AgentConversationService(db)
    await service.batch_delete(user_id, body.ids)
    await db.commit()
    return success()


# --- Graph ---
graph_router = APIRouter(tags=["知识图谱"])


@graph_router.get("/admin/kg/graph")
async def get_graph(entity: str | None = None) -> dict:
    """获取知识图谱 — Phase 4 实现"""
    return success()
