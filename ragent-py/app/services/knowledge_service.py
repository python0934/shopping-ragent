"""
Knowledge service — KnowledgeBase, Document, Chunk CRUD.

Mirrors Java knowledge service layer (KnowledgeBaseServiceImpl,
KnowledgeDocumentServiceImpl, KnowledgeChunkServiceImpl).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import ClientException, ServiceException
from app.core.snowflake import get_snowflake_id_str
from app.core.user_context import UserContext
from app.models.knowledge import (
    KnowledgeBaseDO,
    KnowledgeDocumentDO,
)
from app.schemas.knowledge import (
    KnowledgeBaseCreateRequest,
    KnowledgeBasePageRequest,
    KnowledgeBaseUpdateRequest,
    KnowledgeBaseVO,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentPageRequest,
    KnowledgeDocumentUpdateRequest,
    KnowledgeDocumentVO,
    RagSettingsVO,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# KnowledgeBaseService
# ===========================================================================

class KnowledgeBaseService:
    """知识库服务 — CRUD + 分页查询"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, request: KnowledgeBaseCreateRequest) -> str:
        name = (request.name or "").replace(" ", "")
        if not name:
            raise ClientException("知识库名称不能为空")

        # Name duplicate check
        result = await self.db.execute(
            select(func.count()).select_from(KnowledgeBaseDO).where(
                KnowledgeBaseDO.name == name,
                KnowledgeBaseDO.deleted == 0,
            )
        )
        if (result.scalar() or 0) > 0:
            raise ServiceException(f"知识库名称已存在：{request.name}")

        # Collection name duplicate check
        collection_name = request.collectionName or ""
        if collection_name:
            result = await self.db.execute(
                select(func.count()).select_from(KnowledgeBaseDO).where(
                    KnowledgeBaseDO.collection_name == collection_name,
                    KnowledgeBaseDO.deleted == 0,
                )
            )
            if (result.scalar() or 0) > 0:
                raise ServiceException(f"Collection 名称已存在：{collection_name}")

        username = UserContext.get_username() or ""
        kb_id = get_snowflake_id_str()
        record = KnowledgeBaseDO(
            id=kb_id,
            name=name,
            embedding_model=request.embeddingModel or "",
            collection_name=collection_name,
            created_by=username,
            updated_by=username,
        )
        self.db.add(record)
        await self.db.flush()
        return kb_id

    async def update(self, request: KnowledgeBaseUpdateRequest) -> None:
        result = await self.db.execute(
            select(KnowledgeBaseDO).where(
                KnowledgeBaseDO.id == request.id,
                KnowledgeBaseDO.deleted == 0,
            )
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            raise ClientException(f"知识库不存在：{request.id}")

        if request.embeddingModel and request.embeddingModel != kb.embedding_model:
            doc_count_result = await self.db.execute(
                select(func.count()).select_from(KnowledgeDocumentDO).where(
                    KnowledgeDocumentDO.kb_id == request.id,
                    KnowledgeDocumentDO.chunk_count > 0,
                    KnowledgeDocumentDO.deleted == 0,
                )
            )
            if (doc_count_result.scalar() or 0) > 0:
                raise ClientException("知识库已存在向量化文档，不允许修改嵌入模型")
            kb.embedding_model = request.embeddingModel

        if request.name:
            kb.name = request.name

        kb.updated_by = UserContext.get_username() or ""
        await self.db.flush()

    async def delete(self, kb_id: str) -> None:
        result = await self.db.execute(
            select(KnowledgeBaseDO).where(
                KnowledgeBaseDO.id == kb_id,
                KnowledgeBaseDO.deleted == 0,
            )
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            raise ClientException("知识库不存在")

        # Check for documents
        doc_count_result = await self.db.execute(
            select(func.count()).select_from(KnowledgeDocumentDO).where(
                KnowledgeDocumentDO.kb_id == kb_id,
                KnowledgeDocumentDO.deleted == 0,
            )
        )
        if (doc_count_result.scalar() or 0) > 0:
            raise ClientException("当前知识库下还有文档，请删除文档")

        kb.deleted = 1
        kb.updated_by = UserContext.get_username() or ""
        await self.db.flush()

    async def query_by_id(self, kb_id: str) -> KnowledgeBaseVO:
        result = await self.db.execute(
            select(KnowledgeBaseDO).where(
                KnowledgeBaseDO.id == kb_id,
                KnowledgeBaseDO.deleted == 0,
            )
        )
        kb = result.scalar_one_or_none()
        if kb is None:
            raise ClientException("知识库不存在")
        return self._to_vo(kb)

    async def page_query(self, request: KnowledgeBasePageRequest) -> dict[str, Any]:
        query = select(KnowledgeBaseDO).where(KnowledgeBaseDO.deleted == 0)
        count_query = select(func.count()).select_from(KnowledgeBaseDO).where(KnowledgeBaseDO.deleted == 0)

        if request.name:
            like_pattern = f"%{request.name}%"
            query = query.where(KnowledgeBaseDO.name.like(like_pattern))
            count_query = count_query.where(KnowledgeBaseDO.name.like(like_pattern))

        query = query.order_by(KnowledgeBaseDO.update_time.desc())
        query = query.offset((request.current - 1) * request.size).limit(request.size)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query)
        records = result.scalars().all()

        return {
            "records": [self._to_vo(r) for r in records],
            "total": total,
            "current": request.current,
            "size": request.size,
        }

    @staticmethod
    def _to_vo(record: KnowledgeBaseDO) -> KnowledgeBaseVO:
        return KnowledgeBaseVO(
            id=record.id,
            name=record.name or "",
            embeddingModel=record.embedding_model or "",
            collectionName=record.collection_name or "",
            createdBy=record.created_by or "",
            updatedBy=record.updated_by or "",
            createTime=record.create_time,
            updateTime=record.update_time,
        )


# ===========================================================================
# KnowledgeDocumentService
# ===========================================================================

class KnowledgeDocumentService:
    """知识库文档服务 — CRUD + 分页查询"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, request: KnowledgeDocumentCreateRequest) -> str:
        # Verify KB exists
        kb_result = await self.db.execute(
            select(KnowledgeBaseDO).where(
                KnowledgeBaseDO.id == request.kbId,
                KnowledgeBaseDO.deleted == 0,
            )
        )
        if kb_result.scalar_one_or_none() is None:
            raise ClientException("知识库不存在")

        username = UserContext.get_username() or ""
        doc_id = get_snowflake_id_str()
        record = KnowledgeDocumentDO(
            id=doc_id,
            kb_id=request.kbId,
            doc_name=request.name or "",
            source_type=request.sourceType or "upload",
            source_location=request.sourceRef or "",
            file_url=request.sourceRef or "",
            file_type="unknown",
            created_by=username,
        )
        self.db.add(record)
        await self.db.flush()
        return doc_id

    async def update(self, doc_id: str, request: KnowledgeDocumentUpdateRequest) -> None:
        result = await self.db.execute(
            select(KnowledgeDocumentDO).where(
                KnowledgeDocumentDO.id == doc_id,
                KnowledgeDocumentDO.deleted == 0,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise ClientException("文档不存在")
        if request.name is not None:
            doc.doc_name = request.name
        await self.db.flush()

    async def delete(self, doc_id: str) -> None:
        result = await self.db.execute(
            select(KnowledgeDocumentDO).where(
                KnowledgeDocumentDO.id == doc_id,
                KnowledgeDocumentDO.deleted == 0,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise ClientException("文档不存在")
        doc.deleted = 1
        await self.db.flush()

    async def query_by_id(self, doc_id: str) -> KnowledgeDocumentVO:
        result = await self.db.execute(
            select(KnowledgeDocumentDO).where(
                KnowledgeDocumentDO.id == doc_id,
                KnowledgeDocumentDO.deleted == 0,
            )
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise ClientException("文档不存在")
        return self._to_vo(doc)

    async def page_query(self, request: KnowledgeDocumentPageRequest) -> dict[str, Any]:
        query = select(KnowledgeDocumentDO).where(
            KnowledgeDocumentDO.kb_id == request.kbId,
            KnowledgeDocumentDO.deleted == 0,
        )
        count_query = select(func.count()).select_from(KnowledgeDocumentDO).where(
            KnowledgeDocumentDO.kb_id == request.kbId,
            KnowledgeDocumentDO.deleted == 0,
        )

        if request.status:
            query = query.where(KnowledgeDocumentDO.status == request.status)
            count_query = count_query.where(KnowledgeDocumentDO.status == request.status)
        if request.keyword:
            like_pattern = f"%{request.keyword}%"
            query = query.where(KnowledgeDocumentDO.doc_name.like(like_pattern))
            count_query = count_query.where(KnowledgeDocumentDO.doc_name.like(like_pattern))

        query = query.order_by(KnowledgeDocumentDO.update_time.desc())
        query = query.offset((request.current - 1) * request.size).limit(request.size)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query)
        records = result.scalars().all()

        return {
            "records": [self._to_vo(r) for r in records],
            "total": total,
            "current": request.current,
            "size": request.size,
        }

    @staticmethod
    def _to_vo(record: KnowledgeDocumentDO) -> KnowledgeDocumentVO:
        return KnowledgeDocumentVO(
            id=record.id,
            kbId=record.kb_id or "",
            name=record.doc_name or "",
            sourceType=record.source_type or "",
            sourceRef=record.source_location or "",
            status=record.status or "pending",
            chunkCount=record.chunk_count or 0,
            wordCount=0,
            ingestionSpec=record.ingestion_spec,
            createTime=record.create_time,
            updateTime=record.update_time,
        )


# ===========================================================================
# RagSettingsService
# ===========================================================================

class RagSettingsService:
    """RAG 检索配置服务 — 从 settings 读取"""

    @staticmethod
    def get_settings() -> RagSettingsVO:
        rag = settings.rag
        return RagSettingsVO(
            queryRewriteEnabled=rag.query_rewrite_enabled,
            rerankEnabled=rag.rerank_enabled,
            citationEnabled=rag.citation_enabled,
            defaultTopK=rag.search.default_top_k,
            minRerankScore=rag.search.evidence.min_rerank_score,
            vectorEnabled=rag.search.channels.vector.enabled,
            keywordEnabled=rag.search.channels.keyword.enabled,
            graphEnabled=rag.search.channels.graph.enabled,
            webSearchEnabled=rag.search.channels.web_search.enabled,
            fusionStrategy=rag.search.fusion.strategy,
        )
