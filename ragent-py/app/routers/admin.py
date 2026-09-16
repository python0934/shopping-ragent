"""
Admin dashboard routes — mirrors DashboardController.

Endpoints:
  GET /overview
  GET /performance
  GET /trends
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.result import success

router = APIRouter(tags=["管理后台"])


@router.get("/overview")
async def dashboard_overview(window: str | None = None) -> dict:
    """仪表盘概览 — Phase 3 实现"""
    return success()


@router.get("/performance")
async def dashboard_performance(window: str | None = None) -> dict:
    """性能指标 — Phase 3 实现"""
    return success()


@router.get("/trends")
async def dashboard_trends(metric: str) -> dict:
    """趋势数据 — Phase 3 实现"""
    return success()
