"""
Sample question routes — mirrors SampleQuestionController.

Endpoints:
  GET    /sample-questions/random
  GET    /sample-questions
  GET    /sample-questions/{id}
  POST   /sample-questions
  PUT    /sample-questions/{id}
  DELETE /sample-questions/{id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.result import success
from app.schemas.sample_question import (
    SampleQuestionCreateRequest,
    SampleQuestionPageRequest,
    SampleQuestionUpdateRequest,
)
from app.services.sample_question_service import SampleQuestionService

router = APIRouter(prefix="/sample-questions", tags=["示例问题"])


@router.get("/random")
async def list_random(
    limit: int = 3,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """随机获取示例问题"""
    service = SampleQuestionService(db)
    data = await service.list_random_questions(limit)
    return success([item.model_dump() for item in data])


@router.get("")
async def page_query(
    current: int = 1,
    size: int = 10,
    keyword: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """示例问题分页查询"""
    service = SampleQuestionService(db)
    request = SampleQuestionPageRequest(current=current, size=size, keyword=keyword)
    data = await service.page_query(request)
    return success(data)


@router.get("/{sample_id}")
async def query_by_id(
    sample_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """获取示例问题详情"""
    service = SampleQuestionService(db)
    data = await service.query_by_id(sample_id)
    return success(data.model_dump())


@router.post("")
async def create(
    request: SampleQuestionCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """创建示例问题"""
    service = SampleQuestionService(db)
    question_id = await service.create(request)
    return success({"id": question_id})


@router.put("/{sample_id}")
async def update(
    sample_id: str,
    request: SampleQuestionUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """更新示例问题"""
    service = SampleQuestionService(db)
    await service.update(sample_id, request)
    return success()


@router.delete("/{sample_id}")
async def delete(
    sample_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """删除示例问题"""
    service = SampleQuestionService(db)
    await service.delete(sample_id)
    return success()
