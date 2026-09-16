"""Unit tests for app.core.result — unified response builder."""

from app.core.error_code import BaseErrorCode
from app.core.result import Result, failure, failure_from_exception, success


class TestSuccess:
    def test_success_no_data(self):
        r = success()
        assert r["code"] == "0"
        assert r["message"] is None
        assert r["data"] is None
        assert r["requestId"].startswith("req_")
        assert len(r["requestId"]) == 20  # "req_" (4) + 16 hex chars

    def test_success_with_data(self):
        r = success({"key": "value"})
        assert r["code"] == "0"
        assert r["data"] == {"key": "value"}

    def test_success_with_list_data(self):
        r = success([1, 2, 3])
        assert r["code"] == "0"
        assert r["data"] == [1, 2, 3]

    def test_success_with_string_data(self):
        r = success("hello")
        assert r["code"] == "0"
        assert r["data"] == "hello"


class TestFailure:
    def test_failure_default(self):
        r = failure()
        assert r["code"] == "B000001"
        assert r["message"] == "系统执行出错"
        assert r["data"] is None

    def test_failure_custom_code(self):
        r = failure("A000001", "用户端错误")
        assert r["code"] == "A000001"
        assert r["message"] == "用户端错误"

    def test_failure_from_exception(self):
        r = failure_from_exception("A000100", "未登录或登录已过期")
        assert r["code"] == "A000100"
        assert r["message"] == "未登录或登录已过期"
        assert r["data"] is None
        assert r["requestId"].startswith("req_")


class TestResultModel:
    def test_result_model_success(self):
        r = Result(code="0", data={"id": 1})
        assert r.is_success is True
        assert r.code == "0"

    def test_result_model_failure(self):
        r = Result(code="B000001", message="系统执行出错")
        assert r.is_success is False

    def test_result_serialization(self):
        r = Result(code="0", data="test", requestId="req_abc123")
        d = r.model_dump()
        assert d["code"] == "0"
        assert d["data"] == "test"
        assert d["requestId"] == "req_abc123"
        assert d["message"] is None
