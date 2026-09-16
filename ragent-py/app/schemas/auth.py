"""
Auth schemas — login request/response.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class LoginResponse(BaseModel):
    """登录响应"""
    userId: str = Field(..., description="用户ID")
    role: str = Field(..., description="角色")
    token: str = Field(..., description="认证令牌")
    avatar: str = Field(default="", description="头像URL")
