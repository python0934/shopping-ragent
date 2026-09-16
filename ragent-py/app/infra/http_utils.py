"""
HTTP utilities — ModelUrlResolver, ModelClientErrorType, ModelClientException,
HttpMediaTypes, HttpResponseHelper.

Mirrors Java infra http classes:
  - http.ModelUrlResolver
  - http.ModelClientErrorType
  - http.ModelClientException
  - http.HttpMediaTypes
  - http.HttpResponseHelper
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelClientErrorType — mirrors Java ModelClientErrorType
# ---------------------------------------------------------------------------

class ModelClientErrorType(str, Enum):
    """
    模型客户端错误类型 — 统一错误分类和处理策略。

    Mirrors Java http.ModelClientErrorType.
    """

    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"
    SERVER_ERROR = "SERVER_ERROR"
    CLIENT_ERROR = "CLIENT_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PROVIDER_ERROR = "PROVIDER_ERROR"

    @classmethod
    def from_http_status(cls, status: int) -> ModelClientErrorType:
        """根据 HTTP 状态码推断错误类型"""
        if status in (401, 403):
            return cls.UNAUTHORIZED
        if status == 429:
            return cls.RATE_LIMITED
        if status >= 500:
            return cls.SERVER_ERROR
        return cls.CLIENT_ERROR


# ---------------------------------------------------------------------------
# ModelClientException — mirrors Java ModelClientException
# ---------------------------------------------------------------------------

class ModelClientException(Exception):
    """
    模型客户端异常 — 封装模型调用过程中的各类异常信息。

    Mirrors Java http.ModelClientException.
    """

    def __init__(
        self,
        message: str,
        error_type: ModelClientErrorType = ModelClientErrorType.PROVIDER_ERROR,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        if cause:
            self.__cause__ = cause

    def __str__(self) -> str:
        return f"ModelClientException(type={self.error_type.value}, status={self.status_code}, msg={self.args[0]!r})"


# ---------------------------------------------------------------------------
# HttpMediaTypes — mirrors Java HttpMediaTypes
# ---------------------------------------------------------------------------

class HttpMediaTypes:
    """HTTP 媒体类型常量"""

    JSON = "application/json; charset=utf-8"


# ---------------------------------------------------------------------------
# ModelUrlResolver — mirrors Java ModelUrlResolver
# ---------------------------------------------------------------------------

class ModelUrlResolver:
    """
    模型 URL 解析器 — 解析 AI 模型的完整 URL 地址。

    优先级：候选模型 URL > 提供商基础 URL + 端点路径

    Mirrors Java http.ModelUrlResolver.
    """

    @staticmethod
    def resolve_url(
        provider: dict[str, Any],
        candidate: dict[str, Any] | None,
        capability: Any,
    ) -> str:
        """
        解析模型 URL 地址。

        Args:
            provider: 提供商配置，包含 url 和 endpoints
            candidate: 候选模型配置，可能包含自定义 url
            capability: 模型能力类型（ModelCapability 枚举或字符串）

        Returns:
            完整的模型 URL 地址
        """
        # 候选模型 URL 优先
        if candidate:
            candidate_url = candidate.get("url")
            if candidate_url and candidate_url.strip():
                return candidate_url

        # 提供商基础 URL + 端点路径
        base_url = provider.get("url", "") if provider else ""
        if not base_url or not base_url.strip():
            raise ModelClientException(
                "Provider baseUrl is missing",
                ModelClientErrorType.PROVIDER_ERROR,
            )

        cap_key = capability.value if hasattr(capability, "value") else str(capability)
        endpoints = provider.get("endpoints", {})
        if isinstance(endpoints, dict):
            path = endpoints.get(cap_key.lower(), "")
        else:
            # endpoints might be a Pydantic model
            path = getattr(endpoints, cap_key.lower(), "")

        if not path or not str(path).strip():
            raise ModelClientException(
                f"Provider endpoint is missing: {cap_key.lower()}",
                ModelClientErrorType.PROVIDER_ERROR,
            )

        return ModelUrlResolver._join_url(base_url, str(path))

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        """智能拼接 URL 和路径"""
        if base_url.endswith("/") and path.startswith("/"):
            return base_url + path[1:]
        if not base_url.endswith("/") and not path.startswith("/"):
            return base_url + "/" + path
        return base_url + path


# ---------------------------------------------------------------------------
# HttpResponseHelper — mirrors Java HttpResponseHelper
# ---------------------------------------------------------------------------

class HttpResponseHelper:
    """
    HTTP 响应处理工具类 — 集中管理响应读取、JSON 解析和模型目标校验。

    Mirrors Java http.HttpResponseHelper.
    """

    @staticmethod
    def read_body(body: Any) -> str:
        """读取响应体原始字符串"""
        if body is None:
            return ""
        if isinstance(body, (bytes, bytearray)):
            return body.decode("utf-8")
        return str(body)

    @staticmethod
    def parse_json(body: Any, label: str) -> dict[str, Any]:
        """将响应体解析为 dict"""
        if body is None:
            raise ModelClientException(
                f"{label} 响应为空",
                ModelClientErrorType.INVALID_RESPONSE,
            )
        raw = body if isinstance(body, str) else HttpResponseHelper.read_body(body)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise ModelClientException(
                f"{label} JSON 解析失败: {e}",
                ModelClientErrorType.INVALID_RESPONSE,
                cause=e,
            )

    @staticmethod
    def require_provider(target: Any, label: str) -> dict[str, Any]:
        """校验并返回提供商配置"""
        if target is None or not hasattr(target, "provider") or target.provider is None:
            raise ModelClientException(
                f"{label} 提供商配置缺失",
                ModelClientErrorType.PROVIDER_ERROR,
            )
        return target.provider

    @staticmethod
    def require_api_key(provider: dict[str, Any], label: str) -> str:
        """校验并返回 API 密钥"""
        api_key = provider.get("api_key", "") if isinstance(provider, dict) else getattr(provider, "api_key", "")
        if not api_key or not str(api_key).strip():
            raise ModelClientException(
                f"{label} API密钥缺失",
                ModelClientErrorType.UNAUTHORIZED,
            )
        return str(api_key)

    @staticmethod
    def require_model(target: Any, label: str) -> str:
        """校验并返回模型名称"""
        candidate = target.candidate if hasattr(target, "candidate") else None
        if candidate is None:
            raise ModelClientException(
                f"{label} 模型名称缺失",
                ModelClientErrorType.PROVIDER_ERROR,
            )
        model = candidate.get("model", "") if isinstance(candidate, dict) else getattr(candidate, "model", "")
        if not model:
            raise ModelClientException(
                f"{label} 模型名称缺失",
                ModelClientErrorType.PROVIDER_ERROR,
            )
        return str(model)
