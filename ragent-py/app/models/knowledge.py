"""
Knowledge domain ORM models — KnowledgeBase, KnowledgeDocument, KnowledgeChunk, etc.

Mirrors Java entities:
  - knowledge.dao.entity.KnowledgeBaseDO
  - knowledge.dao.entity.KnowledgeDocumentDO
  - knowledge.dao.entity.KnowledgeChunkDO
  - knowledge.dao.entity.KnowledgeDocumentChunkLogDO
  - knowledge.dao.entity.KnowledgeDocumentScheduleDO
  - knowledge.dao.entity.KnowledgeDocumentScheduleExecDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeBaseDO(Base):
    """知识库表 — t_knowledge_base"""

    __tablename__ = "t_knowledge_base"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键 ID")
    name: Mapped[str] = mapped_column(String(128), nullable=False, comment="知识库名称")
    embedding_model: Mapped[str] = mapped_column(String(64), nullable=False, comment="嵌入模型标识")
    collection_name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="Collection名称")
    created_by: Mapped[str] = mapped_column(String(20), nullable=False, comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), comment="修改人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class KnowledgeDocumentDO(Base):
    """知识库文档表 — t_knowledge_document"""

    __tablename__ = "t_knowledge_document"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    kb_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="知识库ID")
    doc_name: Mapped[str] = mapped_column(String(256), nullable=False, comment="文档名称")
    enabled: Mapped[int] = mapped_column(SmallInteger, default=1, server_default="1", nullable=False, comment="是否启用 1：启用 0：禁用")
    chunk_count: Mapped[int | None] = mapped_column(Integer, default=0, comment="分块数量")
    file_url: Mapped[str] = mapped_column(String(1024), nullable=False, comment="文件存储路径")
    file_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="文件类型")
    mime_type: Mapped[str | None] = mapped_column(String(128), comment="真实MIME类型")
    file_size: Mapped[int | None] = mapped_column(BigInteger, comment="文件大小（字节）")
    process_mode: Mapped[str | None] = mapped_column(String(16), default="chunk", comment="处理模式：chunk/pipeline")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", server_default="'pending'", comment="状态：pending/running/success/failed")
    source_type: Mapped[str | None] = mapped_column(String(16), comment="来源类型：file/url")
    source_location: Mapped[str | None] = mapped_column(String(1024), comment="来源地址")
    schedule_enabled: Mapped[int | None] = mapped_column(SmallInteger, comment="是否启用定时刷新")
    schedule_cron: Mapped[str | None] = mapped_column(String(64), comment="定时表达式")
    ingestion_spec: Mapped[dict | None] = mapped_column(JSONB, comment="文档级摄取配置：解析档位 + 分块预算")
    pipeline_id: Mapped[str | None] = mapped_column(String(20), comment="Pipeline ID")
    created_by: Mapped[str] = mapped_column(String(20), nullable=False, comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), comment="修改人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class KnowledgeChunkDO(Base):
    """知识库文档分块表 — t_knowledge_chunk"""

    __tablename__ = "t_knowledge_chunk"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    kb_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="知识库ID")
    doc_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="文档ID")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="分块序号")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="分块内容")
    content_hash: Mapped[str | None] = mapped_column(String(64), comment="内容哈希")
    char_count: Mapped[int | None] = mapped_column(Integer, comment="字符数")
    token_count: Mapped[int | None] = mapped_column(Integer, comment="Token数")
    embedding_text: Mapped[str | None] = mapped_column(Text, comment="向量文本")
    enabled: Mapped[int] = mapped_column(SmallInteger, default=1, server_default="1", nullable=False, comment="是否启用")
    created_by: Mapped[str] = mapped_column(String(20), nullable=False, comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), comment="修改人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class KnowledgeDocumentChunkLogDO(Base):
    """知识库文档分块日志表 — t_knowledge_document_chunk_log"""

    __tablename__ = "t_knowledge_document_chunk_log"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    doc_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="文档ID")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="状态")
    process_mode: Mapped[str | None] = mapped_column(String(16), comment="处理模式")
    parse_profile: Mapped[str | None] = mapped_column(String(16), comment="解析档位")
    pipeline_id: Mapped[str | None] = mapped_column(String(20), comment="Pipeline ID")
    extract_duration: Mapped[int | None] = mapped_column(BigInteger, comment="提取耗时（毫秒）")
    chunk_duration: Mapped[int | None] = mapped_column(BigInteger, comment="分块耗时（毫秒）")
    embed_duration: Mapped[int | None] = mapped_column(BigInteger, comment="向量化耗时（毫秒）")
    persist_duration: Mapped[int | None] = mapped_column(BigInteger, comment="DB持久化耗时（毫秒）")
    total_duration: Mapped[int | None] = mapped_column(BigInteger, comment="总耗时（毫秒）")
    chunk_count: Mapped[int | None] = mapped_column(Integer, comment="分块数量")
    error_message: Mapped[str | None] = mapped_column(Text, comment="错误信息")
    start_time: Mapped[datetime | None] = mapped_column(comment="开始时间")
    end_time: Mapped[datetime | None] = mapped_column(comment="结束时间")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")


class KnowledgeDocumentScheduleDO(Base):
    """知识库文档定时刷新任务表 — t_knowledge_document_schedule"""

    __tablename__ = "t_knowledge_document_schedule"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    doc_id: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, comment="文档ID")
    kb_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="知识库ID")
    cron_expr: Mapped[str | None] = mapped_column(String(64), comment="Cron表达式")
    enabled: Mapped[int | None] = mapped_column(SmallInteger, default=0, comment="是否启用")
    next_run_time: Mapped[datetime | None] = mapped_column(comment="下次执行时间")
    last_run_time: Mapped[datetime | None] = mapped_column(comment="上次执行时间")
    last_success_time: Mapped[datetime | None] = mapped_column(comment="上次成功时间")
    last_status: Mapped[str | None] = mapped_column(String(16), comment="上次状态")
    last_error: Mapped[str | None] = mapped_column(String(512), comment="上次错误")
    last_etag: Mapped[str | None] = mapped_column(String(256), comment="上次ETag")
    last_modified: Mapped[str | None] = mapped_column(String(256), comment="上次修改时间")
    last_content_hash: Mapped[str | None] = mapped_column(String(128), comment="上次内容哈希")
    lock_owner: Mapped[str | None] = mapped_column(String(128), comment="锁持有者")
    lock_until: Mapped[datetime | None] = mapped_column(comment="锁过期时间")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")


class KnowledgeDocumentScheduleExecDO(Base):
    """知识库文档定时刷新执行记录表 — t_knowledge_document_schedule_exec"""

    __tablename__ = "t_knowledge_document_schedule_exec"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    schedule_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="调度ID")
    doc_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="文档ID")
    kb_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="知识库ID")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="状态")
    message: Mapped[str | None] = mapped_column(String(512), comment="消息")
    start_time: Mapped[datetime | None] = mapped_column(comment="开始时间")
    end_time: Mapped[datetime | None] = mapped_column(comment="结束时间")
    file_name: Mapped[str | None] = mapped_column(String(512), comment="文件名")
    file_size: Mapped[int | None] = mapped_column(BigInteger, comment="文件大小")
    content_hash: Mapped[str | None] = mapped_column(String(128), comment="内容哈希")
    etag: Mapped[str | None] = mapped_column(String(256), comment="ETag")
    last_modified: Mapped[str | None] = mapped_column(String(256), comment="最后修改时间")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")
