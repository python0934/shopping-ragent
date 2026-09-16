"""
Audit schemas — BizChangeLog request/response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BizChangeLogPageRequest(BaseModel):
    """审计日志分页查询请求"""
    current: int = Field(default=1, ge=1, description="当前页码")
    size: int = Field(default=10, ge=1, le=100, description="每页条数")
    bizType: str | None = Field(default=None, description="业务类型")
    bizId: str | None = Field(default=None, description="业务ID")
    operationType: str | None = Field(default=None, description="操作类型")
    operatorId: str | None = Field(default=None, description="操作人ID")
    operatorName: str | None = Field(default=None, description="操作人名称")
    success: bool | None = Field(default=None, description="是否成功")
    beginTime: datetime | None = Field(default=None, description="开始时间")
    endTime: datetime | None = Field(default=None, description="结束时间")


class BizChangeLogVO(BaseModel):
    """审计日志视图对象"""
    id: str
    bizType: str = ""
    bizId: str = ""
    operationType: str = ""
    actionDesc: str | None = None
    beforeSnapshot: dict[str, Any] | None = None
    afterSnapshot: dict[str, Any] | None = None
    changeDiff: dict[str, Any] | None = None
    operatorId: str = ""
    operatorName: str = ""
    operatorRole: str = ""
    success: bool = True
    errorMessage: str | None = None
    createTime: datetime | None = None
