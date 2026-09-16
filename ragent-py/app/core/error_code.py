"""
Error code definitions — mirrors Java BaseErrorCode enum.

Categories:
  A0xxxxx — Client errors (user-side)
  B0xxxxx — Service errors (system-side)
  C0xxxxx — Remote errors (third-party)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    """Immutable error-code descriptor."""
    code: str
    message: str


class BaseErrorCode:
    """Standard error codes shared across all modules."""

    # ===== A-class: Client errors =====
    CLIENT_ERROR = ErrorCode("A000001", "用户端错误")

    # A01 — User registration
    USER_REGISTER_ERROR = ErrorCode("A000100", "用户注册错误")
    USER_NAME_VERIFY_ERROR = ErrorCode("A000110", "用户名校验失败")
    USER_NAME_EXIST_ERROR = ErrorCode("A000111", "用户名已存在")
    USER_NAME_SENSITIVE_ERROR = ErrorCode("A000112", "用户名包含敏感词")
    USER_NAME_SPECIAL_CHARACTER_ERROR = ErrorCode("A000113", "用户名包含特殊字符")
    PASSWORD_VERIFY_ERROR = ErrorCode("A000120", "密码校验失败")
    PASSWORD_SHORT_ERROR = ErrorCode("A000121", "密码长度不够")
    PHONE_VERIFY_ERROR = ErrorCode("A000151", "手机格式校验失败")

    # A02 — Idempotent
    IDEMPOTENT_TOKEN_NULL_ERROR = ErrorCode("A000200", "幂等Token为空")
    IDEMPOTENT_TOKEN_DELETE_ERROR = ErrorCode("A000201", "幂等Token已被使用或失效")

    # A03 — Query limits
    SEARCH_AMOUNT_EXCEEDS_LIMIT = ErrorCode("A000300", "查询数据量超过最大限制")

    # Auth (used by middleware, mirrors Sa-Token error responses)
    NOT_LOGIN = ErrorCode("A000100", "未登录或登录已过期")
    NOT_ROLE = ErrorCode("A000101", "权限不足")

    # Idempotent submit / rate-limit / demo-mode (app-level codes)
    IDEMPOTENT_REJECT = ErrorCode("A000105", "请勿重复提交")
    RATE_LIMIT_REJECT = ErrorCode("A000106", "当前排队人数较多，请稍后再试")
    DEMO_MODE_REJECT = ErrorCode("A000107", "演示模式，请勿修改")

    # 404
    NOT_FOUND = ErrorCode("A000404", "接口不存在")

    # ===== B-class: Service errors =====
    SERVICE_ERROR = ErrorCode("B000001", "系统执行出错")
    SERVICE_TIMEOUT_ERROR = ErrorCode("B000100", "系统执行超时")

    # ===== C-class: Remote errors =====
    REMOTE_ERROR = ErrorCode("C000001", "调用第三方服务出错")
