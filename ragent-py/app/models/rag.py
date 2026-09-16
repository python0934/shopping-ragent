"""
RAG intent & query domain ORM models — IntentNode, QueryTermMapping, Trace.

Mirrors Java entities:
  - rag.dao.entity.IntentNodeDO
  - rag.dao.entity.QueryTermMappingDO
  - rag.dao.entity.RagTraceRunDO
  - rag.dao.entity.RagTraceNodeDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Integer, JSON, SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

# JSONB 带 SQLite 变体：生产走 PG 仍是 JSONB，测试可在 SQLite 内存库上跑真 SQL
JsonType = JSONB().with_variant(JSON(), "sqlite")


class IntentNodeDO(Base):
    """意图树节点配置表 — t_intent_node"""

    __tablename__ = "t_intent_node"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    kb_id: Mapped[str | None] = mapped_column(String(20), comment="知识库ID")
    intent_code: Mapped[str] = mapped_column(String(64), nullable=False, comment="业务唯一标识")
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="展示名称")
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="层级 0:DOMAIN 1:CATEGORY 2:TOPIC")
    parent_code: Mapped[str | None] = mapped_column(String(64), comment="父节点标识")
    description: Mapped[str | None] = mapped_column(String(512), comment="语义描述")
    examples: Mapped[str | None] = mapped_column(Text, comment="示例问题")
    collection_name: Mapped[str | None] = mapped_column(String(128), comment="兼容旧版本Collection名称")
    # 库级 DEFAULT '[]'::jsonb 由 resources/database/schema_pg.sql 持有；
    # 模型侧只留 Python 默认值，免得建表 DDL 在 SQLite 上不可用
    collection_names: Mapped[dict] = mapped_column(JsonType, nullable=False, default=list, comment="知识库Collection集合")
    top_k: Mapped[int | None] = mapped_column(Integer, comment="知识库检索TopK")
    mcp_tool_id: Mapped[str | None] = mapped_column(String(128), comment="MCP工具ID")
    kind: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0, server_default="0", comment="类型 0:RAG 1:SYSTEM 2:MCP")
    prompt_snippet: Mapped[str | None] = mapped_column(Text, comment="提示词片段")
    prompt_template: Mapped[str | None] = mapped_column(Text, comment="提示词模板")
    param_prompt_template: Mapped[str | None] = mapped_column(Text, comment="参数提取提示词模板")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0", comment="排序字段")
    enabled: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1, server_default="1", comment="是否启用 1:启用 0:禁用")
    create_by: Mapped[str | None] = mapped_column(String(20), comment="创建人")
    update_by: Mapped[str | None] = mapped_column(String(20), comment="修改人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="修改时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class QueryTermMappingDO(Base):
    """关键词归一化映射表 — t_query_term_mapping"""

    __tablename__ = "t_query_term_mapping"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    domain: Mapped[str | None] = mapped_column(String(64), comment="领域")
    source_term: Mapped[str] = mapped_column(String(128), nullable=False, comment="源词")
    target_term: Mapped[str] = mapped_column(String(128), nullable=False, comment="目标词")
    match_type: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1, server_default="1", comment="匹配类型 1:精确 2:模糊")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, server_default="100", comment="优先级")
    enabled: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1, server_default="1", comment="是否启用")
    remark: Mapped[str | None] = mapped_column(String(255), comment="备注")
    create_by: Mapped[str | None] = mapped_column(String(20), comment="创建人")
    update_by: Mapped[str | None] = mapped_column(String(20), comment="修改人")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="修改时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", nullable=False, comment="是否删除 0：正常 1：删除")


class RagTraceRunDO(Base):
    """Trace 运行记录表 — t_rag_trace_run"""

    __tablename__ = "t_rag_trace_run"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="全局链路ID")
    trace_name: Mapped[str | None] = mapped_column(String(128), comment="链路名称")
    entry_method: Mapped[str | None] = mapped_column(String(256), comment="入口方法")
    conversation_id: Mapped[str | None] = mapped_column(String(20), comment="会话ID")
    task_id: Mapped[str | None] = mapped_column(String(20), comment="任务ID")
    user_id: Mapped[str | None] = mapped_column(String(20), comment="用户ID")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="RUNNING", server_default="'RUNNING'", comment="RUNNING/SUCCESS/ERROR")
    error_message: Mapped[str | None] = mapped_column(String(1000), comment="错误信息")
    start_time: Mapped[datetime | None] = mapped_column(comment="开始时间")
    end_time: Mapped[datetime | None] = mapped_column(comment="结束时间")
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, comment="耗时毫秒")
    extra_data: Mapped[str | None] = mapped_column(Text, comment="扩展字段(JSON)")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除")


class RagTraceNodeDO(Base):
    """Trace 节点记录表 — t_rag_trace_node"""

    __tablename__ = "t_rag_trace_node"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    trace_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="所属链路ID")
    node_id: Mapped[str] = mapped_column(String(20), nullable=False, comment="节点ID")
    parent_node_id: Mapped[str | None] = mapped_column(String(20), comment="父节点ID")
    depth: Mapped[int | None] = mapped_column(Integer, default=0, comment="节点深度")
    node_type: Mapped[str | None] = mapped_column(String(16), comment="节点类型")
    node_name: Mapped[str | None] = mapped_column(String(128), comment="节点名称")
    class_name: Mapped[str | None] = mapped_column(String(256), comment="类名")
    method_name: Mapped[str | None] = mapped_column(String(128), comment="方法名")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="RUNNING", server_default="'RUNNING'", comment="RUNNING/SUCCESS/ERROR")
    error_message: Mapped[str | None] = mapped_column(String(1000), comment="错误信息")
    start_time: Mapped[datetime | None] = mapped_column(comment="开始时间")
    end_time: Mapped[datetime | None] = mapped_column(comment="结束时间")
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, comment="耗时毫秒")
    extra_data: Mapped[str | None] = mapped_column(Text, comment="扩展字段(JSON)")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除")
