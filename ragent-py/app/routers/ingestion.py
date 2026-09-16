"""
Ingestion routes — mirrors IngestionPipelineController, IngestionTaskController.

Endpoints:
  POST   /ingestion/pipelines
  PUT    /ingestion/pipelines/{id}
  GET    /ingestion/pipelines/{id}
  GET    /ingestion/pipelines
  DELETE /ingestion/pipelines/{id}
  POST   /ingestion/tasks
  POST   /ingestion/tasks/upload
  GET    /ingestion/tasks/{id}
  GET    /ingestion/tasks/{id}/nodes
  GET    /ingestion/tasks
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.result import success

router = APIRouter(tags=["数据摄取"])


# --- Pipelines ---

@router.post("/ingestion/pipelines")
async def create_pipeline() -> dict:
    """创建摄取流水线 — Phase 4 实现"""
    return success()


@router.put("/ingestion/pipelines/{pipeline_id}")
async def update_pipeline(pipeline_id: str) -> dict:
    """更新摄取流水线 — Phase 4 实现"""
    return success()


@router.get("/ingestion/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str) -> dict:
    """获取摄取流水线详情 — Phase 4 实现"""
    return success()


@router.get("/ingestion/pipelines")
async def page_query_pipelines() -> dict:
    """摄取流水线分页查询 — Phase 4 实现"""
    return success()


@router.delete("/ingestion/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str) -> dict:
    """删除摄取流水线 — Phase 4 实现"""
    return success()


# --- Tasks ---

@router.post("/ingestion/tasks")
async def create_task() -> dict:
    """创建摄取任务 — Phase 4 实现"""
    return success()


@router.post("/ingestion/tasks/upload")
async def upload_task(pipeline_id: str) -> dict:
    """上传文件创建摄取任务 — Phase 4 实现"""
    return success()


@router.get("/ingestion/tasks/{task_id}")
async def get_task(task_id: str) -> dict:
    """获取摄取任务详情 — Phase 4 实现"""
    return success()


@router.get("/ingestion/tasks/{task_id}/nodes")
async def get_task_nodes(task_id: str) -> dict:
    """获取摄取任务节点列表 — Phase 4 实现"""
    return success()


@router.get("/ingestion/tasks")
async def page_query_tasks() -> dict:
    """摄取任务分页查询 — Phase 4 实现"""
    return success()
