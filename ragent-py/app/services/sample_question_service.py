"""
Sample question service — CRUD, page query, random list.

Mirrors Java SampleQuestionServiceImpl.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ClientException
from app.core.snowflake import get_snowflake_id_str
from app.models.system import SampleQuestionDO
from app.schemas.sample_question import (
    SampleQuestionCreateRequest,
    SampleQuestionPageRequest,
    SampleQuestionUpdateRequest,
    SampleQuestionVO,
)

logger = logging.getLogger(__name__)

MAX_RANDOM_LIMIT = 20


class SampleQuestionService:
    """示例问题服务 — CRUD、分页查询、随机取数"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, request: SampleQuestionCreateRequest) -> str:
        """创建示例问题"""
        question = (request.question or "").strip()
        if not question:
            raise ClientException("示例问题内容不能为空")

        question_id = get_snowflake_id_str()
        record = SampleQuestionDO(
            id=question_id,
            title=(request.title or "").strip() or None,
            description=(request.description or "").strip() or None,
            question=question,
        )
        self.db.add(record)
        await self.db.flush()
        return question_id

    async def update(self, question_id: str, request: SampleQuestionUpdateRequest) -> None:
        """更新示例问题"""
        record = await self._load_by_id(question_id)

        if request.question is not None:
            question = request.question.strip()
            if not question:
                raise ClientException("示例问题内容不能为空")
            record.question = question
        if request.title is not None:
            record.title = request.title.strip() or None
        if request.description is not None:
            record.description = request.description.strip() or None

        await self.db.flush()

    async def delete(self, question_id: str) -> None:
        """删除示例问题（软删除）"""
        record = await self._load_by_id(question_id)
        record.deleted = 1
        await self.db.flush()

    async def query_by_id(self, question_id: str) -> SampleQuestionVO:
        """根据 ID 查询示例问题"""
        record = await self._load_by_id(question_id)
        return self._to_vo(record)

    async def page_query(self, request: SampleQuestionPageRequest) -> dict[str, Any]:
        """示例问题分页查询"""
        keyword = (request.keyword or "").strip() or None

        query = select(SampleQuestionDO).where(SampleQuestionDO.deleted == 0)
        count_query = select(func.count()).select_from(SampleQuestionDO).where(SampleQuestionDO.deleted == 0)

        if keyword:
            like_pattern = f"%{keyword}%"
            filter_cond = or_(
                SampleQuestionDO.title.like(like_pattern),
                SampleQuestionDO.description.like(like_pattern),
                SampleQuestionDO.question.like(like_pattern),
            )
            query = query.where(filter_cond)
            count_query = count_query.where(filter_cond)

        query = query.order_by(SampleQuestionDO.update_time.desc())
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

    async def list_random_questions(self, limit: int) -> list[SampleQuestionVO]:
        """随机获取示例问题"""
        size = min(max(limit, 1), MAX_RANDOM_LIMIT)
        result = await self.db.execute(
            select(SampleQuestionDO)
            .where(SampleQuestionDO.deleted == 0)
            .order_by(text("RANDOM()"))
            .limit(size)
        )
        records = result.scalars().all()
        return [self._to_vo(r) for r in records]

    async def _load_by_id(self, question_id: str) -> SampleQuestionDO:
        result = await self.db.execute(
            select(SampleQuestionDO).where(
                SampleQuestionDO.id == question_id,
                SampleQuestionDO.deleted == 0,
            )
        )
        record = result.scalar_one_or_none()
        if record is None:
            raise ClientException("示例问题不存在")
        return record

    @staticmethod
    def _to_vo(record: SampleQuestionDO) -> SampleQuestionVO:
        return SampleQuestionVO(
            id=record.id,
            title=record.title,
            description=record.description,
            question=record.question or "",
            createTime=record.create_time,
            updateTime=record.update_time,
        )
