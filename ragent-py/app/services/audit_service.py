"""
Audit service — BizChangeLog page query and detail.

Mirrors Java BizChangeLogServiceImpl.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ClientException
from app.models.system import BizChangeLogDO
from app.schemas.audit import BizChangeLogPageRequest, BizChangeLogVO

logger = logging.getLogger(__name__)


class BizChangeLogService:
    """审计日志服务 — 分页查询、详情"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def page_query(self, request: BizChangeLogPageRequest) -> dict[str, Any]:
        """审计日志分页查询"""
        query = select(BizChangeLogDO)
        count_query = select(func.count()).select_from(BizChangeLogDO)

        # Apply filters
        if request.bizType:
            query = query.where(BizChangeLogDO.biz_type == request.bizType)
            count_query = count_query.where(BizChangeLogDO.biz_type == request.bizType)
        if request.bizId:
            like_pattern = f"%{request.bizId}%"
            query = query.where(BizChangeLogDO.biz_id.like(like_pattern))
            count_query = count_query.where(BizChangeLogDO.biz_id.like(like_pattern))
        if request.operationType:
            query = query.where(BizChangeLogDO.operation_type == request.operationType)
            count_query = count_query.where(BizChangeLogDO.operation_type == request.operationType)
        if request.operatorId:
            query = query.where(BizChangeLogDO.operator_id == request.operatorId)
            count_query = count_query.where(BizChangeLogDO.operator_id == request.operatorId)
        if request.operatorName:
            like_pattern = f"%{request.operatorName}%"
            query = query.where(BizChangeLogDO.operator_name.like(like_pattern))
            count_query = count_query.where(BizChangeLogDO.operator_name.like(like_pattern))
        if request.success is not None:
            query = query.where(BizChangeLogDO.success == request.success)
            count_query = count_query.where(BizChangeLogDO.success == request.success)
        if request.beginTime:
            query = query.where(BizChangeLogDO.create_time >= request.beginTime)
            count_query = count_query.where(BizChangeLogDO.create_time >= request.beginTime)
        if request.endTime:
            query = query.where(BizChangeLogDO.create_time <= request.endTime)
            count_query = count_query.where(BizChangeLogDO.create_time <= request.endTime)

        query = query.order_by(BizChangeLogDO.create_time.desc())
        query = query.offset((request.current - 1) * request.size).limit(request.size)

        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        result = await self.db.execute(query)
        records = result.scalars().all()

        return {
            "records": [self._to_vo(r) for r in records],
            "total": total,
            "current": request.current,
            "size": request.size,
        }

    async def get_by_id(self, log_id: str) -> BizChangeLogVO:
        """根据 ID 获取审计日志详情"""
        result = await self.db.execute(
            select(BizChangeLogDO).where(BizChangeLogDO.id == log_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ClientException("变更审计日志不存在")
        return self._to_vo(record)

    @staticmethod
    def _to_vo(record: BizChangeLogDO) -> BizChangeLogVO:
        return BizChangeLogVO(
            id=record.id,
            bizType=record.biz_type or "",
            bizId=record.biz_id or "",
            operationType=record.operation_type or "",
            actionDesc=record.action_desc,
            beforeSnapshot=record.before_snapshot,
            afterSnapshot=record.after_snapshot,
            changeDiff=record.change_diff,
            operatorId=record.operator_id or "",
            operatorName=record.operator_name or "",
            operatorRole=record.operator_role or "",
            success=record.success if record.success is not None else True,
            errorMessage=record.error_message,
            createTime=record.create_time,
        )
