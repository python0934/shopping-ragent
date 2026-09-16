"""
System domain ORM models — User, BizChangeLog, SampleQuestion.

Mirrors Java entities:
  - user.dao.entity.UserDO
  - audit.dao.entity.BizChangeLogDO
  - sample.dao.entity.SampleQuestionDO
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import SmallInteger, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class UserDO(Base):
    """系统用户表 — t_user"""

    __tablename__ = "t_user"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, comment="用户名，唯一")
    password: Mapped[str] = mapped_column(String(128), nullable=False, comment="密码")
    role: Mapped[str] = mapped_column(String(32), nullable=False, comment="角色：admin/user")
    avatar: Mapped[str | None] = mapped_column(String(128), comment="用户头像")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")


class BizChangeLogDO(Base):
    """业务数据变更审计日志表 — t_biz_change_log"""

    __tablename__ = "t_biz_change_log"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="主键ID")
    biz_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="业务对象类型")
    biz_id: Mapped[str] = mapped_column(String(64), nullable=False, comment="业务对象主键")
    operation_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="操作类型")
    action_desc: Mapped[str | None] = mapped_column(String(512), comment="操作描述")
    before_snapshot: Mapped[dict | None] = mapped_column(JSONB, comment="变更前快照")
    after_snapshot: Mapped[dict | None] = mapped_column(JSONB, comment="变更后快照")
    change_diff: Mapped[dict | None] = mapped_column(JSONB, comment="变更差异")
    operator_id: Mapped[str | None] = mapped_column(String(64), comment="操作人ID")
    operator_name: Mapped[str | None] = mapped_column(String(128), comment="操作人名称")
    operator_role: Mapped[str | None] = mapped_column(String(64), comment="操作人角色")
    success: Mapped[bool] = mapped_column(default=True, server_default="TRUE", comment="是否成功")
    error_message: Mapped[str | None] = mapped_column(Text, comment="失败信息")
    class_name: Mapped[str | None] = mapped_column(String(255), comment="触发类名")
    method_name: Mapped[str | None] = mapped_column(String(255), comment="触发方法名")
    ip: Mapped[str | None] = mapped_column(String(64), comment="来源IP")
    user_agent: Mapped[str | None] = mapped_column(String(512), comment="User-Agent")
    create_time: Mapped[datetime | None] = mapped_column(server_default="now()", comment="创建时间")


class SampleQuestionDO(Base):
    """示例问题表 — t_sample_question"""

    __tablename__ = "t_sample_question"

    id: Mapped[str] = mapped_column(String(20), primary_key=True, comment="ID")
    title: Mapped[str | None] = mapped_column(String(64), comment="展示标题")
    description: Mapped[str | None] = mapped_column(String(255), comment="描述或提示")
    question: Mapped[str] = mapped_column(String(255), nullable=False, comment="示例问题内容")
    create_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="创建时间")
    update_time: Mapped[datetime | None] = mapped_column(server_default="CURRENT_TIMESTAMP", comment="更新时间")
    deleted: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0", comment="是否删除 0：正常 1：删除")
