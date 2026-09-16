"""
Auth routes — mirrors AuthController.

Endpoints:
  POST /auth/login
  POST /auth/logout
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.result import success
from app.schemas.auth import LoginRequest, LoginResponse
from app.services.auth_service import AuthService

router = APIRouter(tags=["认证"])


@router.post("/auth/login")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """用户登录"""
    service = AuthService(db)
    data: LoginResponse = await service.login(request)
    return success(data.model_dump())


@router.post("/auth/logout")
async def logout() -> dict:
    """用户登出"""
    # Token removal is best-effort; the middleware already validates on each request
    return success()
