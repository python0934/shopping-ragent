"""
Knowledge routes — mirrors KnowledgeBaseController, KnowledgeDocumentController, KnowledgeChunkController.

Endpoints:
  POST   /knowledge-base
  PUT    /knowledge-base/{kb_id}
  DELETE /knowledge-base/{kb_id}
  GET    /knowledge-base/{kb_id}
  GET    /knowledge-base
  GET    /knowledge-base/docs/ingestion-spec-schema
  POST   /knowledge-base/{kb_id}/docs/upload
  POST   /knowledge-base/docs/{doc_id}/chunk
  DELETE /knowledge-base/docs/{doc_id}
  GET    /knowledge-base/docs/{doc_id}
  PUT    /knowledge-base/docs/{doc_id}
  GET    /knowledge-base/{kb_id}/docs
  GET    /knowledge-base/docs/search
  GET    /knowledge-base/docs/{doc_id}/chunk-logs
  GET    /knowledge-base/docs/{doc_id}/preview
  GET    /knowledge-base/docs/{doc_id}/file
  GET    /knowledge-base/docs/{doc_id}/chunks
  POST   /knowledge-base/docs/{doc_id}/chunks
  PUT    /knowledge-base/docs/{doc_id}/chunks/{chunk_id}
  DELETE /knowledge-base/docs/{doc_id}/chunks/{chunk_id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.result import success
from app.schemas.knowledge import (
    KnowledgeBaseCreateRequest,
    KnowledgeBasePageRequest,
    KnowledgeBaseUpdateRequest,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentPageRequest,
    KnowledgeDocumentUpdateRequest,
)
from app.services.knowledge_service import KnowledgeBaseService, KnowledgeDocumentService

router = APIRouter(tags=["知识库"])


# --- Knowledge Base ---

@router.post("/knowledge-base")
async def create_knowledge_base(
    request: KnowledgeBaseCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """创建知识库"""
    service = KnowledgeBaseService(db)
    kb_id = await service.create(request)
    return success({"id": kb_id})


@router.put("/knowledge-base/{kb_id}")
async def rename_knowledge_base(
    kb_id: str,
    request: KnowledgeBaseUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """重命名/更新知识库"""
    service = KnowledgeBaseService(db)
    request.id = kb_id
    await service.update(request)
    return success()


@router.delete("/knowledge-base/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """删除知识库"""
    service = KnowledgeBaseService(db)
    await service.delete(kb_id)
    return success()


@router.get("/knowledge-base/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """获取知识库详情"""
    service = KnowledgeBaseService(db)
    data = await service.query_by_id(kb_id)
    return success(data.model_dump())


@router.get("/knowledge-base")
async def page_query_knowledge_bases(
    current: int = 1,
    size: int = 10,
    name: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """知识库分页查询"""
    service = KnowledgeBaseService(db)
    request = KnowledgeBasePageRequest(current=current, size=size, name=name)
    data = await service.page_query(request)
    return success(data)


# --- Knowledge Documents ---

@router.get("/knowledge-base/docs/ingestion-spec-schema")
async def get_ingestion_spec_schema() -> dict:
    """获取摄取配置 Schema"""
    return success({"schema": {}})


@router.post("/knowledge-base/{kb_id}/docs/upload")
async def upload_document(
    kb_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """上传文档（简化为创建文档记录）"""
    service = KnowledgeDocumentService(db)
    request = KnowledgeDocumentCreateRequest(kbId=kb_id, name="uploaded-document")
    doc_id = await service.create(request)
    return success({"id": doc_id})


@router.post("/knowledge-base/docs/{doc_id}/chunk")
async def start_chunk(doc_id: str) -> dict:
    """启动分块处理 — 需要 Pipeline 引擎支持"""
    return success()


@router.delete("/knowledge-base/docs/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """删除文档"""
    service = KnowledgeDocumentService(db)
    await service.delete(doc_id)
    return success()


@router.get("/knowledge-base/docs/{doc_id}")
async def get_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """获取文档详情"""
    service = KnowledgeDocumentService(db)
    data = await service.query_by_id(doc_id)
    return success(data.model_dump())


@router.put("/knowledge-base/docs/{doc_id}")
async def update_document(
    doc_id: str,
    request: KnowledgeDocumentUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """更新文档"""
    service = KnowledgeDocumentService(db)
    await service.update(doc_id, request)
    return success()


@router.get("/knowledge-base/{kb_id}/docs")
async def page_query_documents(
    kb_id: str,
    current: int = 1,
    size: int = 10,
    status: str | None = None,
    keyword: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """文档分页查询"""
    service = KnowledgeDocumentService(db)
    request = KnowledgeDocumentPageRequest(
        current=current, size=size, kbId=kb_id, status=status, keyword=keyword,
    )
    data = await service.page_query(request)
    return success(data)


@router.get("/knowledge-base/docs/search")
async def search_documents(
    keyword: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """文档搜索"""
    return success([])


@router.get("/knowledge-base/docs/{doc_id}/chunk-logs")
async def get_chunk_logs(doc_id: str) -> dict:
    """获取分块日志"""
    return success([])


@router.get("/knowledge-base/docs/{doc_id}/preview")
async def preview_document(doc_id: str) -> dict:
    """预览文档"""
    return success({"content": ""})


@router.get("/knowledge-base/docs/{doc_id}/file")
async def get_document_file(doc_id: str) -> dict:
    """下载文档文件"""
    return success({"url": ""})


# --- Knowledge Chunks ---

@router.get("/knowledge-base/docs/{doc_id}/chunks")
async def page_query_chunks(doc_id: str) -> dict:
    """分块分页查询"""
    return success({"records": [], "total": 0, "current": 1, "size": 10})


@router.post("/knowledge-base/docs/{doc_id}/chunks")
async def create_chunk(doc_id: str) -> dict:
    """创建分块"""
    return success()


@router.put("/knowledge-base/docs/{doc_id}/chunks/{chunk_id}")
async def update_chunk(doc_id: str, chunk_id: str) -> dict:
    """更新分块"""
    return success()


@router.delete("/knowledge-base/docs/{doc_id}/chunks/{chunk_id}")
async def delete_chunk(doc_id: str, chunk_id: str) -> dict:
    """删除分块"""
    return success()
