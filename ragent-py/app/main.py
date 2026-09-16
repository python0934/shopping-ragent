"""
FastAPI application entry point — mirrors Spring Boot bootstrap + global middleware.

Key behavioural contracts:
  1. ALL responses (including errors) return HTTP 200 with Result JSON
  2. Auth middleware reads raw token from Authorization header (no Bearer prefix)
  3. Demo mode blocks write operations
  4. CORS configured for frontend
  5. Request ID injected into every response

Implementation note:
  The global exception handler uses pure ASGI middleware (not BaseHTTPMiddleware)
  because Starlette's BaseHTTPMiddleware has a known limitation where exceptions
  raised in inner middleware don't propagate to outer middleware's try/except.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator
from urllib.parse import parse_qs

import orjson
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.core.error_code import BaseErrorCode
from app.core.exceptions import (
    AbstractException,
    DemoModeRejectException,
    NotLoginException,
    NotRoleException,
)
from app.core.result import failure_from_exception
from app.core.user_context import UserContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom JSON response using orjson
# ---------------------------------------------------------------------------

class ORJSONResponse(JSONResponse):
    """JSON response using orjson for performance."""

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifecycle: init Redis on startup, close on shutdown."""
    logger.info("Ragent Python starting up...")
    try:
        from app.core.redis_client import init_redis
        await init_redis()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis connection failed (will retry on demand): %s", e)

    # MCP 连接与意图树快照都是 Agent 工具目录的前置：失败只降级成「无 MCP 工具」
    try:
        from app.mcp import mcp_client_manager
        await mcp_client_manager.startup()
    except Exception as e:
        logger.warning("MCP client startup failed (agent runs without MCP tools): %s", e)

    try:
        from app.core.database import async_session_factory
        from app.services.intent_registry import intent_node_registry
        async with async_session_factory() as session:
            await intent_node_registry.refresh(session)
    except Exception as e:
        logger.warning("Intent tree snapshot failed (MCP intent nodes unavailable): %s", e)

    yield

    from app.core.redis_client import close_redis
    await close_redis()
    try:
        from app.mcp import mcp_client_manager
        await mcp_client_manager.shutdown()
    except Exception as e:
        logger.warning("MCP client shutdown failed: %s", e)
    from app.core.database import close_db
    await close_db()
    logger.info("Ragent Python shut down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ragent AI Platform",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.router.default_response_class = ORJSONResponse


# ---------------------------------------------------------------------------
# Global exception handler — registered LAST so it's the OUTERMOST middleware
# ---------------------------------------------------------------------------
# In Starlette, middleware added last wraps all previous middleware.
# This ensures exceptions from ALL inner middleware are caught here.

async def _global_exception_handler(request: Request, call_next: Any) -> Response:
    """
    Catches ALL exceptions and returns HTTP 200 + Result JSON.
    Mirrors Java GlobalExceptionHandler — the frontend relies on
    HTTP 200 + code field, NOT HTTP status codes.
    """
    try:
        return await call_next(request)
    except AbstractException as exc:
        if exc.__cause__:
            logger.error("[%s] %s [ex] %s [cause] %s", request.method, request.url, exc, exc.__cause__)
        else:
            logger.warning("[%s] %s [ex] %s", request.method, request.url, exc)
        body = failure_from_exception(exc.error_code, exc.errorMessage)
        return ORJSONResponse(status_code=200, content=body)
    except Exception as exc:
        logger.error("[%s] %s", request.method, request.url, exc_info=True)
        body = failure_from_exception(
            BaseErrorCode.SERVICE_ERROR.code,
            BaseErrorCode.SERVICE_ERROR.message,
        )
        return ORJSONResponse(status_code=200, content=body)


# ---------------------------------------------------------------------------
# Request validation handler
# ---------------------------------------------------------------------------
# FastAPI 默认把参数校验失败渲染成 HTTP 422，且响应体是自成一家的 detail 数组。
# Java 侧的 HandlerMethodValidationException 走 GlobalExceptionHandler，出来仍是
# HTTP 200 + Result；前端只认 code 字段，422 会被当成网络故障，故在这里拉齐。

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    message = _format_validation_errors(exc.errors())
    logger.warning("[%s] %s [validation] %s", request.method, request.url, message)
    return ORJSONResponse(
        status_code=200,
        content=failure_from_exception(BaseErrorCode.CLIENT_ERROR.code, message),
    )


def _format_validation_errors(errors: list[Any]) -> str:
    """取首个错误的字段名与原因，对齐 Java 只报第一条 FieldError 的做法"""
    if not errors:
        return BaseErrorCode.CLIENT_ERROR.message
    first = errors[0]
    loc = [str(part) for part in first.get("loc", ()) if part not in ("query", "body", "path")]
    field = ".".join(loc) if loc else "参数"
    return f"{field}: {first.get('msg', '参数校验失败')}"


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# BaseHTTPMiddleware stack (inner layers)
# ---------------------------------------------------------------------------

# --- Request ID + timing ---
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Any) -> Response:
    """Inject request ID and measure timing."""
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    request.state.request_id = request_id
    start = time.perf_counter()

    response: Response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
    return response


# --- Auth middleware — mirrors Sa-Token interceptor ---

AUTH_WHITELIST = {"/auth/login", "/auth/logout", "/health"}
AUTH_PREFIX_WHITELIST = ("/auth/",)


@app.middleware("http")
async def auth_middleware(request: Request, call_next: Any) -> Response:
    """
    Authentication middleware — mirrors Java SaTokenConfig + UserContextInterceptor.
    Returns error JSON directly (never raises) to avoid BaseHTTPMiddleware exception issues.
    """
    path = request.url.path
    rel_path = path[len(settings.api_prefix):] if path.startswith(settings.api_prefix) else path

    # Skip auth for whitelisted paths
    if rel_path in AUTH_WHITELIST or any(rel_path.startswith(p) for p in AUTH_PREFIX_WHITELIST):
        return await call_next(request)

    # Skip auth for OPTIONS (CORS preflight)
    if request.method == "OPTIONS":
        return await call_next(request)

    # Extract token from Authorization header (raw, no Bearer prefix)
    token = request.headers.get("Authorization", "").strip()
    if not token:
        body = failure_from_exception(BaseErrorCode.NOT_LOGIN.code, BaseErrorCode.NOT_LOGIN.message)
        return ORJSONResponse(status_code=200, content=body)

    # Look up token in Redis
    try:
        from app.core.redis_client import get_redis
        redis = get_redis()
        user_id = await redis.get(f"login:token:{token}")
        if not user_id:
            body = failure_from_exception(BaseErrorCode.NOT_LOGIN.code, BaseErrorCode.NOT_LOGIN.message)
            return ORJSONResponse(status_code=200, content=body)

        user_data = await redis.hgetall(f"login:user:{user_id}")
        if not user_data:
            body = failure_from_exception(BaseErrorCode.NOT_LOGIN.code, BaseErrorCode.NOT_LOGIN.message)
            return ORJSONResponse(status_code=200, content=body)

        from app.core.user_context import LoginUser
        user = LoginUser(
            userId=user_id,
            username=user_data.get("username", ""),
            role=user_data.get("role", "user"),
            avatar=user_data.get("avatar", ""),
        )
        UserContext.set(user)
        await redis.expire(f"login:token:{token}", settings.token_timeout)

    except (NotLoginException, NotRoleException):
        body = failure_from_exception(BaseErrorCode.NOT_LOGIN.code, BaseErrorCode.NOT_LOGIN.message)
        return ORJSONResponse(status_code=200, content=body)
    except Exception as e:
        logger.warning("Auth middleware error: %s", e)
        body = failure_from_exception(BaseErrorCode.NOT_LOGIN.code, BaseErrorCode.NOT_LOGIN.message)
        return ORJSONResponse(status_code=200, content=body)

    try:
        response = await call_next(request)
    finally:
        UserContext.clear()

    return response


# --- Demo mode middleware ---

WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


@app.middleware("http")
async def demo_mode_middleware(request: Request, call_next: Any) -> Response:
    """
    Demo mode guard — mirrors Java DemoModeInterceptor.
    Returns error JSON directly (never raises) to avoid BaseHTTPMiddleware exception issues.
    """
    if settings.demo_mode and request.method in WRITE_METHODS:
        body = failure_from_exception(BaseErrorCode.DEMO_MODE_REJECT.code, BaseErrorCode.DEMO_MODE_REJECT.message)
        return ORJSONResponse(status_code=200, content=body)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Register global exception handler LAST (outermost middleware)
# ---------------------------------------------------------------------------
# In Starlette, the last middleware added is the outermost one.
# This MUST come after all @app.middleware("http") registrations.

from starlette.middleware.base import BaseHTTPMiddleware as _BHM

app.add_middleware(_BHM, dispatch=_global_exception_handler)


# ---------------------------------------------------------------------------
# Health check (MUST be registered before catch-all)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Register all API routers
# ---------------------------------------------------------------------------

from app.routers.auth import router as auth_router
from app.routers.user import router as user_router
from app.routers.sample_question import router as sample_question_router
from app.routers.audit import router as audit_router
from app.routers.chat import router as chat_router
from app.routers.rag_chat import router as rag_chat_router
from app.routers.knowledge import router as knowledge_router
from app.routers.agent import (
    profile_router as agent_profile_router,
    intent_router as intent_tree_router,
    mapping_router as mapping_router,
    agent_chat_router as agent_chat_router,
    agent_conv_router as agent_conv_router,
    graph_router as graph_router,
)
from app.routers.ingestion import router as ingestion_router
from app.routers.admin import router as admin_router

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(sample_question_router)
app.include_router(audit_router)
app.include_router(chat_router)
app.include_router(rag_chat_router)
app.include_router(knowledge_router)
app.include_router(agent_profile_router)
app.include_router(intent_tree_router)
app.include_router(mapping_router)
app.include_router(agent_chat_router)
app.include_router(agent_conv_router)
app.include_router(graph_router)
app.include_router(ingestion_router)
app.include_router(admin_router)


# ---------------------------------------------------------------------------
# 404 catch-all — return Result JSON instead of HTML
# ---------------------------------------------------------------------------

@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all_404(path_name: str) -> JSONResponse:
    """Catch-all for undefined routes — returns Result JSON (HTTP 200)."""
    body = failure_from_exception(
        BaseErrorCode.NOT_FOUND.code,
        BaseErrorCode.NOT_FOUND.message,
    )
    return JSONResponse(status_code=200, content=body)


# ---------------------------------------------------------------------------
# Run with: uvicorn app.main:app --host 0.0.0.0 --port 9090
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.server.port, reload=True)
