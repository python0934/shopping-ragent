"""
Ingestion pipeline domain ORM models — Pipeline, PipelineNode, Task, TaskNode.

Mirrors Java entities:
  - ingestion.dao.entity.IngestionPipelineDO
  - ingestion.dao.entity.IngestionPipelineNodeDO
  - ingestion.dao.entity.IngestionTaskDO
  - ingestion.dao.entity.IngestionTaskNodeDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Integer, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class IngestionPipelineDO(Base):
    """摄取流水线表 — t_ingestion_pipeline"""

    __tablename__ = "t_ingestion_pipeline"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="流水线名称")
    description: Mapped[str | None] = mapped_column(Text, comment="流水线描述")
    created_by: Mapped[str | None] = mapped_column(String(20), default="", comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), default="", comment="更新人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class IngestionPipelineNodeDO(Base):
    """摄取流水线节点表 — t_ingestion_pipeline_node"""

    __tablename__ = "t_ingestion_pipeline_node"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    pipeline_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="流水线ID")
    node_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="节点标识")
    node_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="节点类型")
    next_node_id: Mapped[str | None] = mapped_column(String(20), comment="下一个节点ID")
    settings_json: Mapped[dict | None] = mapped_column(JSONB, comment="节点配置JSON")
    condition_json: Mapped[dict | None] = mapped_column(JSONB, comment="条件JSON")
    created_by: Mapped[str | None] = mapped_column(String(20), default="", comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), default="", comment="更新人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class IngestionTaskDO(Base):
    """摄取任务表 — t_ingestion_task"""

    __tablename__ = "t_ingestion_task"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    pipeline_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="流水线ID")
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, comment="来源类型")
    source_location: Mapped[str | None] = mapped_column(Text, comment="来源地址或URL")
    source_file_name: Mapped[str | None] = mapped_column(String(255), comment="原始文件名")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="任务状态")
    chunk_count: Mapped[int | None] = mapped_column(Integer, default=0, comment="分块数量")
    error_message: Mapped[str | None] = mapped_column(Text, comment="错误信息")
    logs_json: Mapped[dict | None] = mapped_column(JSONB, comment="节点日志JSON")
    metadata_json: Mapped[dict | None] = mapped_column(JSONB, comment="扩展元数据JSON")
    started_at: Mapped[datetime | None] = mapped_column(comment="开始时间")
    completed_at: Mapped[datetime | None] = mapped_column(comment="完成时间")
    created_by: Mapped[str | None] = mapped_column(String(20), default="", comment="创建人")
    updated_by: Mapped[str | None] = mapped_column(String(20), default="", comment="更新人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class IngestionTaskNodeDO(Base):
    """摄取任务节点表 — t_ingestion_task_node"""

    __tablename__ = "t_ingestion_task_node"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    task_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="任务ID")
    pipeline_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="流水线ID")
    node_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="节点标识")
    node_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="节点类型")
    node_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0", comment="节点顺序")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="节点状态")
    duration_ms: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default="0", comment="执行耗时(毫秒)")
    message: Mapped[str | None] = mapped_column(Text, comment="节点消息")
    error_message: Mapped[str | None] = mapped_column(Text, comment="错误信息")
    output_json: Mapped[str | None] = mapped_column(Text, comment="节点输出JSON(全量)")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", nullable=False, comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")
