"""
Audit routes — mirrors BizChangeLogController.

Endpoints:
  GET /biz-change-logs
  GET /biz-change-logs/{id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.result import success
from app.schemas.audit import BizChangeLogPageRequest
from app.services.audit_service import BizChangeLogService

router = APIRouter(tags=["审计日志"])


@router.get("/biz-change-logs")
async def page_query(
    current: int = 1,
    size: int = 10,
    bizType: str | None = None,
    bizId: str | None = None,
    operationType: str | None = None,
    operatorId: str | None = None,
    operatorName: str | None = None,
    success: bool | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """审计日志分页查询"""
    service = BizChangeLogService(db)
    request = BizChangeLogPageRequest(
        current=current,
        size=size,
        bizType=bizType,
        bizId=bizId,
        operationType=operationType,
        operatorId=operatorId,
        operatorName=operatorName,
        success=success,
    )
    data = await service.page_query(request)
    return success(data)


@router.get("/biz-change-logs/{log_id}")
async def get_by_id(
    log_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """获取审计日志详情"""
    service = BizChangeLogService(db)
    data = await service.get_by_id(log_id)
    return success(data.model_dump())
