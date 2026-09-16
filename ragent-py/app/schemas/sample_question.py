"""
Sample question schemas — CRUD request/response models.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SampleQuestionCreateRequest(BaseModel):
    """创建示例问题请求"""
    question: str = Field(..., description="示例问题内容")
    title: str | None = Field(default=None, description="标题")
    description: str | None = Field(default=None, description="描述")


class SampleQuestionUpdateRequest(BaseModel):
    """更新示例问题请求"""
    question: str | None = Field(default=None, description="示例问题内容")
    title: str | None = Field(default=None, description="标题")
    description: str | None = Field(default=None, description="描述")


class SampleQuestionPageRequest(BaseModel):
    """示例问题分页查询请求"""
    current: int = Field(default=1, ge=1, description="当前页码")
    size: int = Field(default=10, ge=1, le=100, description="每页条数")
    keyword: str | None = Field(default=None, description="搜索关键字")


class SampleQuestionVO(BaseModel):
    """示例问题视图对象"""
    id: str
    title: str | None = None
    description: str | None = None
    question: str
    createTime: datetime | None = None
    updateTime: datetime | None = None
