"""
Phase 4 unit tests — RAG core engine (knowledge base, documents, settings).

Tests cover:
  - Knowledge schemas validation
  - KnowledgeBaseService logic (CRUD with mocked DB)
  - KnowledgeDocumentService logic
  - RagSettingsService (reads from config)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.knowledge import (
    KnowledgeBaseCreateRequest,
    KnowledgeBasePageRequest,
    KnowledgeBaseUpdateRequest,
    KnowledgeBaseVO,
    KnowledgeChunkCreateRequest,
    KnowledgeChunkPageRequest,
    KnowledgeChunkVO,
    KnowledgeDocumentCreateRequest,
    KnowledgeDocumentPageRequest,
    KnowledgeDocumentUpdateRequest,
    KnowledgeDocumentVO,
    RagSettingsVO,
)


# ===========================================================================
# TestKnowledgeSchemas
# ===========================================================================

class TestKnowledgeBaseCreateRequest:
    def test_valid(self):
        req = KnowledgeBaseCreateRequest(name="Test KB", embeddingModel="text-emb", collectionName="test_coll")
        assert req.name == "Test KB"
        assert req.embeddingModel == "text-emb"
        assert req.collectionName == "test_coll"

    def test_missing_name(self):
        with pytest.raises(Exception):
            KnowledgeBaseCreateRequest()


class TestKnowledgeBaseUpdateRequest:
    def test_valid(self):
        req = KnowledgeBaseUpdateRequest(id="123", name="New Name")
        assert req.id == "123"
        assert req.name == "New Name"
        assert req.embeddingModel is None


class TestKnowledgeBasePageRequest:
    def test_defaults(self):
        req = KnowledgeBasePageRequest()
        assert req.current == 1
        assert req.size == 10
        assert req.name is None


class TestKnowledgeBaseVO:
    def test_creation(self):
        vo = KnowledgeBaseVO(id="1", name="Test KB", embeddingModel="emb", collectionName="coll")
        assert vo.id == "1"
        assert vo.name == "Test KB"
        assert vo.documentCount == 0


class TestKnowledgeDocumentCreateRequest:
    def test_valid(self):
        req = KnowledgeDocumentCreateRequest(kbId="kb1", name="doc.pdf")
        assert req.kbId == "kb1"
        assert req.sourceType == "upload"


class TestKnowledgeDocumentPageRequest:
    def test_valid(self):
        req = KnowledgeDocumentPageRequest(kbId="kb1")
        assert req.kbId == "kb1"
        assert req.current == 1


class TestKnowledgeDocumentVO:
    def test_creation(self):
        vo = KnowledgeDocumentVO(id="1", kbId="kb1", name="doc.pdf", status="completed")
        assert vo.status == "completed"
        assert vo.chunkCount == 0


class TestKnowledgeChunkCreateRequest:
    def test_valid(self):
        req = KnowledgeChunkCreateRequest(kbId="kb1", docId="doc1", content="Hello")
        assert req.content == "Hello"


class TestKnowledgeChunkVO:
    def test_creation(self):
        vo = KnowledgeChunkVO(id="1", kbId="kb1", docId="doc1", content="Hello")
        assert vo.position == 0
        assert vo.wordCount == 0


class TestRagSettingsVO:
    def test_defaults(self):
        vo = RagSettingsVO()
        assert vo.queryRewriteEnabled is True
        assert vo.rerankEnabled is True
        assert vo.defaultTopK == 10
        assert vo.fusionStrategy == "rrf"


# ===========================================================================
# TestKnowledgeBaseServiceLogic
# ===========================================================================

class TestKnowledgeBaseServiceToVO:
    def test_conversion(self):
        from app.models.knowledge import KnowledgeBaseDO
        from app.services.knowledge_service import KnowledgeBaseService
        kb = KnowledgeBaseDO(
            id="123",
            name="Test KB",
            embedding_model="text-emb",
            collection_name="test_coll",
            created_by="admin",
            updated_by="admin",
        )
        vo = KnowledgeBaseService._to_vo(kb)
        assert vo.id == "123"
        assert vo.name == "Test KB"
        assert vo.embeddingModel == "text-emb"
        assert vo.collectionName == "test_coll"
        assert vo.createdBy == "admin"


class TestKnowledgeDocumentServiceToVO:
    def test_conversion(self):
        from app.models.knowledge import KnowledgeDocumentDO
        from app.services.knowledge_service import KnowledgeDocumentService
        doc = KnowledgeDocumentDO(
            id="456",
            kb_id="123",
            doc_name="test.pdf",
            source_type="upload",
            source_location="s3://bucket/test.pdf",
            status="completed",
            chunk_count=10,
            file_url="s3://bucket/test.pdf",
            file_type="pdf",
            created_by="admin",
        )
        vo = KnowledgeDocumentService._to_vo(doc)
        assert vo.id == "456"
        assert vo.kbId == "123"
        assert vo.name == "test.pdf"
        assert vo.status == "completed"
        assert vo.chunkCount == 10


# ===========================================================================
# TestRagSettingsService
# ===========================================================================

class TestRagSettingsService:
    def test_get_settings(self):
        from app.services.knowledge_service import RagSettingsService
        settings_vo = RagSettingsService.get_settings()
        assert isinstance(settings_vo, RagSettingsVO)
        assert settings_vo.defaultTopK == 10
        assert settings_vo.fusionStrategy == "rrf"
        assert settings_vo.vectorEnabled is True
