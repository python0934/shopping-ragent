"""
RAG chat routes — mirrors RAGChatController, RAGSettingsController, RagTraceController, EvalController.

Endpoints:
  GET  /rag/v3/chat (SSE)
  POST /rag/v3/stop
  GET  /rag/settings
  GET  /rag/eval
  GET  /rag/traces/runs
  GET  /rag/traces/runs/{traceId}
  GET  /rag/traces/runs/{traceId}/nodes
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.result import success

router = APIRouter(tags=["RAG 对话"])


@router.get("/rag/v3/chat")
async def rag_chat() -> dict:
    """RAG 流式对话 (SSE) — Phase 4 实现"""
    return success()


@router.post("/rag/v3/stop")
async def rag_stop(task_id: str) -> dict:
    """停止 RAG 对话 — Phase 4 实现"""
    return success()


@router.get("/rag/settings")
async def rag_settings() -> dict:
    """获取 RAG 系统设置"""
    from app.services.knowledge_service import RagSettingsService
    data = RagSettingsService.get_settings()
    return success(data.model_dump())


@router.get("/rag/eval")
async def rag_eval(question: str) -> dict:
    """RAG 评测接口 — Phase 4 实现"""
    return success()


@router.get("/rag/traces/runs")
async def page_trace_runs() -> dict:
    """Trace 运行记录分页 — Phase 4 实现"""
    return success()


@router.get("/rag/traces/runs/{trace_id}")
async def trace_detail(trace_id: str) -> dict:
    """Trace 详情 — Phase 4 实现"""
    return success()


@router.get("/rag/traces/runs/{trace_id}/nodes")
async def trace_nodes(trace_id: str) -> dict:
    """Trace 节点列表 — Phase 4 实现"""
    return success()
