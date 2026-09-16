"""Unit tests for app.core.exceptions — three-tier exception hierarchy."""

from app.core.error_code import BaseErrorCode, ErrorCode
from app.core.exceptions import (
    AbstractException,
    ClientException,
    DemoModeRejectException,
    IdempotentRejectException,
    NotLoginException,
    NotRoleException,
    RateLimitRejectException,
    RemoteException,
    ServiceException,
)


class TestAbstractException:
    def test_with_error_code_only(self):
        exc = AbstractException(error_code=BaseErrorCode.SERVICE_ERROR)
        assert exc.error_code == "B000001"
        assert exc.errorMessage == "系统执行出错"

    def test_with_message_overrides(self):
        exc = AbstractException(
            message="自定义消息",
            error_code=BaseErrorCode.SERVICE_ERROR,
        )
        assert exc.error_code == "B000001"
        assert exc.errorMessage == "自定义消息"

    def test_with_cause(self):
        cause = ValueError("root cause")
        exc = AbstractException(
            message="wrapped",
            error_code=BaseErrorCode.SERVICE_ERROR,
            cause=cause,
        )
        assert exc.__cause__ is cause

    def test_default_error_code(self):
        exc = AbstractException(message="no code")
        assert exc.error_code == "B000001"

    def test_str_representation(self):
        exc = ClientException("bad input")
        s = str(exc)
        assert "ClientException" in s
        assert "A000001" in s


class TestClientException:
    def test_default_code(self):
        exc = ClientException("参数错误")
        assert exc.error_code == "A000001"
        assert exc.errorMessage == "参数错误"

    def test_custom_error_code(self):
        exc = ClientException(error_code=BaseErrorCode.USER_NAME_EXIST_ERROR)
        assert exc.error_code == "A000111"
        assert exc.errorMessage == "用户名已存在"

    def test_message_with_error_code(self):
        exc = ClientException(
            message="用户名太短",
            error_code=BaseErrorCode.USER_NAME_VERIFY_ERROR,
        )
        assert exc.error_code == "A000110"
        assert exc.errorMessage == "用户名太短"


class TestServiceException:
    def test_default_code(self):
        exc = ServiceException("内部错误")
        assert exc.error_code == "B000001"

    def test_fallback_to_error_code_message(self):
        exc = ServiceException(error_code=BaseErrorCode.SERVICE_TIMEOUT_ERROR)
        assert exc.errorMessage == "系统执行超时"


class TestRemoteException:
    def test_default_code(self):
        exc = RemoteException("第三方超时")
        assert exc.error_code == "C000001"
        assert exc.errorMessage == "第三方超时"


class TestConvenienceExceptions:
    def test_not_login(self):
        exc = NotLoginException()
        assert exc.error_code == "A000100"
        assert "未登录" in exc.errorMessage

    def test_not_role(self):
        exc = NotRoleException()
        assert exc.error_code == "A000101"
        assert "权限" in exc.errorMessage

    def test_idempotent_reject(self):
        exc = IdempotentRejectException()
        assert exc.error_code == "A000105"
        assert "重复" in exc.errorMessage

    def test_rate_limit_reject(self):
        exc = RateLimitRejectException()
        assert exc.error_code == "A000106"

    def test_demo_mode_reject(self):
        exc = DemoModeRejectException()
        assert exc.error_code == "A000107"


class TestErrorCode:
    def test_error_code_immutable(self):
        ec = ErrorCode("X000001", "test")
        assert ec.code == "X000001"
        assert ec.message == "test"

    def test_error_code_frozen(self):
        ec = ErrorCode("X000001", "test")
        try:
            ec.code = "changed"
            assert False, "Should have raised"
        except AttributeError:
            pass
