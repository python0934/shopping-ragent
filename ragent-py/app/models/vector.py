"""
Vector storage ORM model — KnowledgeVector (pgvector).

No direct Java DO — table defined in schema_pg.sql for pgvector-based retrieval.
"""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeVectorDO(Base):
    """知识库向量存储表 — t_knowledge_vector (pgvector)"""

    __tablename__ = "t_knowledge_vector"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="分块ID")
    collection_name: Mapped[str] = mapped_column(String(64), nullable=False, comment="知识库Collection")
    content: Mapped[str | None] = mapped_column(Text, comment="分块文本内容")
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, comment="元数据")
    # embedding column omitted from ORM — managed via raw SQL / pgvector functions
    # In queries, use: text("embedding <-> :query_vec") for cosine distance
