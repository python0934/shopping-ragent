"""
Knowledge schemas — request/response models for knowledge base, document, chunk.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., description="知识库名称")
    embeddingModel: str = Field(default="", description="嵌入模型")
    collectionName: str = Field(default="", description="Collection 名称")


class KnowledgeBaseUpdateRequest(BaseModel):
    id: str = Field(..., description="知识库ID")
    name: str | None = Field(default=None, description="知识库名称")
    embeddingModel: str | None = Field(default=None, description="嵌入模型")


class KnowledgeBasePageRequest(BaseModel):
    current: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)
    name: str | None = Field(default=None, description="知识库名称搜索")


class KnowledgeBaseVO(BaseModel):
    id: str
    name: str = ""
    embeddingModel: str = ""
    collectionName: str = ""
    createdBy: str = ""
    updatedBy: str = ""
    documentCount: int = 0
    createTime: datetime | None = None
    updateTime: datetime | None = None


# ---------------------------------------------------------------------------
# Knowledge Document
# ---------------------------------------------------------------------------

class KnowledgeDocumentCreateRequest(BaseModel):
    kbId: str = Field(..., description="知识库ID")
    name: str = Field(default="", description="文档名称")
    sourceType: str = Field(default="upload", description="来源类型")
    sourceRef: str = Field(default="", description="来源引用")


class KnowledgeDocumentUpdateRequest(BaseModel):
    name: str | None = Field(default=None, description="文档名称")


class KnowledgeDocumentPageRequest(BaseModel):
    current: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)
    kbId: str = Field(..., description="知识库ID")
    status: str | None = Field(default=None, description="状态筛选")
    keyword: str | None = Field(default=None, description="关键字搜索")


class KnowledgeDocumentVO(BaseModel):
    id: str
    kbId: str = ""
    name: str = ""
    sourceType: str = ""
    sourceRef: str = ""
    status: str = "pending"
    chunkCount: int = 0
    wordCount: int = 0
    ingestionSpec: dict[str, Any] | None = None
    createTime: datetime | None = None
    updateTime: datetime | None = None


# ---------------------------------------------------------------------------
# Knowledge Chunk
# ---------------------------------------------------------------------------

class KnowledgeChunkCreateRequest(BaseModel):
    kbId: str = Field(..., description="知识库ID")
    docId: str = Field(..., description="文档ID")
    content: str = Field(..., description="内容")


class KnowledgeChunkUpdateRequest(BaseModel):
    content: str | None = Field(default=None, description="内容")


class KnowledgeChunkPageRequest(BaseModel):
    current: int = Field(default=1, ge=1)
    size: int = Field(default=10, ge=1, le=100)
    kbId: str = Field(..., description="知识库ID")
    docId: str | None = Field(default=None, description="文档ID筛选")
    keyword: str | None = Field(default=None, description="关键字搜索")


class KnowledgeChunkVO(BaseModel):
    id: str
    kbId: str = ""
    docId: str = ""
    content: str = ""
    position: int = 0
    wordCount: int = 0
    sourceRef: list[str] | None = None
    groundingChunks: list[dict[str, Any]] | None = None
    createTime: datetime | None = None
    updateTime: datetime | None = None


# ---------------------------------------------------------------------------
# RAG Settings
# ---------------------------------------------------------------------------

class RagSettingsVO(BaseModel):
    """RAG 检索配置"""
    queryRewriteEnabled: bool = True
    rerankEnabled: bool = True
    citationEnabled: bool = True
    defaultTopK: int = 10
    minRerankScore: float = 0.2
    vectorEnabled: bool = True
    keywordEnabled: bool = False
    graphEnabled: bool = False
    webSearchEnabled: bool = False
    fusionStrategy: str = "rrf"
