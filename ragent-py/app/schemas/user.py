"""
User schemas — CRUD request/response models.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserCreateRequest(BaseModel):
    """创建用户请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    role: str = Field(default="user", description="角色")
    avatar: str = Field(default="", description="头像URL")


class UserUpdateRequest(BaseModel):
    """更新用户请求"""
    username: str | None = Field(default=None, description="用户名")
    role: str | None = Field(default=None, description="角色")
    avatar: str | None = Field(default=None, description="头像URL")
    password: str | None = Field(default=None, description="新密码")


class UserPageRequest(BaseModel):
    """用户分页查询请求"""
    current: int = Field(default=1, ge=1, description="当前页码")
    size: int = Field(default=10, ge=1, le=100, description="每页条数")
    keyword: str | None = Field(default=None, description="搜索关键字")


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""
    currentPassword: str = Field(..., description="当前密码")
    newPassword: str = Field(..., description="新密码")


class UserVO(BaseModel):
    """用户视图对象"""
    id: str
    username: str
    role: str
    avatar: str = ""
    createTime: datetime | None = None
    updateTime: datetime | None = None


class CurrentUserVO(BaseModel):
    """当前用户信息"""
    userId: str
    username: str
    role: str
    avatar: str = ""
