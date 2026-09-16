"""
User routes — mirrors UserController.

Endpoints:
  GET  /user/me
  GET  /users
  POST /users
  PUT  /users/{id}
  DELETE /users/{id}
  PUT  /user/password
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.result import success
from app.schemas.user import (
    ChangePasswordRequest,
    UserCreateRequest,
    UserPageRequest,
    UserUpdateRequest,
)
from app.services.user_service import UserService

router = APIRouter(tags=["用户管理"])


@router.get("/user/me")
async def current_user(db: AsyncSession = Depends(get_db)) -> dict:
    """获取当前登录用户信息"""
    service = UserService(db)
    data = await service.get_current_user()
    return success(data.model_dump())


@router.get("/users")
async def page_query_users(
    current: int = 1,
    size: int = 10,
    keyword: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """用户分页查询"""
    service = UserService(db)
    request = UserPageRequest(current=current, size=size, keyword=keyword)
    data = await service.page_query(request)
    return success(data)


@router.post("/users")
async def create_user(
    request: UserCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """创建用户"""
    service = UserService(db)
    user_id = await service.create(request)
    return success({"id": user_id})


@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """更新用户"""
    service = UserService(db)
    await service.update(user_id, request)
    return success()


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """删除用户"""
    service = UserService(db)
    await service.delete(user_id)
    return success()


@router.put("/user/password")
async def change_password(
    request: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """修改密码"""
    service = UserService(db)
    await service.change_password(request)
    return success()
