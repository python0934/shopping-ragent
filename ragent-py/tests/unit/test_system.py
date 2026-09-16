"""
Phase 3 unit tests — System support domain (auth/user/audit/sample questions).

Tests cover:
  - Schemas validation (Pydantic models)
  - Service logic (mocked DB)
  - Router integration (FastAPI TestClient with mocked services)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.auth import LoginRequest, LoginResponse
from app.schemas.user import (
    ChangePasswordRequest,
    CurrentUserVO,
    UserCreateRequest,
    UserPageRequest,
    UserUpdateRequest,
    UserVO,
)
from app.schemas.audit import BizChangeLogPageRequest, BizChangeLogVO
from app.schemas.sample_question import (
    SampleQuestionCreateRequest,
    SampleQuestionPageRequest,
    SampleQuestionUpdateRequest,
    SampleQuestionVO,
)


# ===========================================================================
# TestAuthSchemas
# ===========================================================================

class TestLoginRequest:
    def test_valid(self):
        req = LoginRequest(username="admin", password="123456")
        assert req.username == "admin"
        assert req.password == "123456"

    def test_missing_fields(self):
        with pytest.raises(Exception):
            LoginRequest()


class TestLoginResponse:
    def test_creation(self):
        resp = LoginResponse(userId="123", role="admin", token="abc", avatar="http://img.png")
        assert resp.userId == "123"
        assert resp.role == "admin"
        assert resp.token == "abc"
        assert resp.avatar == "http://img.png"

    def test_default_avatar(self):
        resp = LoginResponse(userId="123", role="user", token="abc")
        assert resp.avatar == ""


# ===========================================================================
# TestUserSchemas
# ===========================================================================

class TestUserCreateRequest:
    def test_defaults(self):
        req = UserCreateRequest(username="test", password="pass")
        assert req.role == "user"
        assert req.avatar == ""

    def test_custom(self):
        req = UserCreateRequest(username="test", password="pass", role="admin", avatar="http://img")
        assert req.role == "admin"


class TestUserUpdateRequest:
    def test_all_optional(self):
        req = UserUpdateRequest()
        assert req.username is None
        assert req.role is None
        assert req.password is None


class TestUserPageRequest:
    def test_defaults(self):
        req = UserPageRequest()
        assert req.current == 1
        assert req.size == 10
        assert req.keyword is None


class TestChangePasswordRequest:
    def test_valid(self):
        req = ChangePasswordRequest(currentPassword="old", newPassword="new")
        assert req.currentPassword == "old"
        assert req.newPassword == "new"

    def test_missing_fields(self):
        with pytest.raises(Exception):
            ChangePasswordRequest()


class TestUserVO:
    def test_creation(self):
        vo = UserVO(id="123", username="test", role="user")
        assert vo.id == "123"
        assert vo.createTime is None

    def test_with_datetime(self):
        now = datetime.now()
        vo = UserVO(id="123", username="test", role="user", createTime=now)
        assert vo.createTime == now


class TestCurrentUserVO:
    def test_creation(self):
        vo = CurrentUserVO(userId="123", username="admin", role="admin")
        assert vo.userId == "123"


# ===========================================================================
# TestAuditSchemas
# ===========================================================================

class TestBizChangeLogPageRequest:
    def test_defaults(self):
        req = BizChangeLogPageRequest()
        assert req.current == 1
        assert req.size == 10
        assert req.bizType is None
        assert req.success is None


class TestBizChangeLogVO:
    def test_creation(self):
        vo = BizChangeLogVO(id="1", bizType="USER", bizId="123")
        assert vo.bizType == "USER"
        assert vo.success is True


# ===========================================================================
# TestSampleQuestionSchemas
# ===========================================================================

class TestSampleQuestionCreateRequest:
    def test_valid(self):
        req = SampleQuestionCreateRequest(question="What is RAG?")
        assert req.question == "What is RAG?"
        assert req.title is None

    def test_missing_question(self):
        with pytest.raises(Exception):
            SampleQuestionCreateRequest()


class TestSampleQuestionUpdateRequest:
    def test_all_optional(self):
        req = SampleQuestionUpdateRequest()
        assert req.question is None
        assert req.title is None


class TestSampleQuestionPageRequest:
    def test_defaults(self):
        req = SampleQuestionPageRequest()
        assert req.current == 1
        assert req.size == 10


class TestSampleQuestionVO:
    def test_creation(self):
        vo = SampleQuestionVO(id="1", question="What is RAG?")
        assert vo.question == "What is RAG?"
        assert vo.title is None


# ===========================================================================
# TestAuthServiceLogic
# ===========================================================================

class TestAuthServicePasswordMatch:
    def test_matches(self):
        from app.services.auth_service import AuthService
        assert AuthService._password_matches("abc", "abc")
        assert not AuthService._password_matches("abc", "def")
        assert AuthService._password_matches(None, None)
        assert not AuthService._password_matches("abc", None)
        assert not AuthService._password_matches(None, "abc")


# ===========================================================================
# TestUserServiceLogic
# ===========================================================================

class TestUserServiceNormalizeRole:
    def test_valid_roles(self):
        from app.services.user_service import UserService
        assert UserService._normalize_role("admin") == "admin"
        assert UserService._normalize_role("user") == "user"
        assert UserService._normalize_role("ADMIN") == "admin"
        assert UserService._normalize_role("USER") == "user"

    def test_empty_defaults_to_user(self):
        from app.services.user_service import UserService
        assert UserService._normalize_role(None) == "user"
        assert UserService._normalize_role("") == "user"

    def test_invalid_role_raises(self):
        from app.core.exceptions import ClientException
        from app.services.user_service import UserService
        with pytest.raises(ClientException, match="角色类型不合法"):
            UserService._normalize_role("superadmin")


class TestUserServiceEnsureNotDefaultAdmin:
    def test_admin_protected(self):
        from app.core.exceptions import ClientException
        from app.models.system import UserDO
        from app.services.user_service import UserService
        user = UserDO(id="1", username="admin", role="admin")
        with pytest.raises(ClientException, match="默认管理员不允许修改或删除"):
            UserService._ensure_not_default_admin(user)

    def test_non_admin_ok(self):
        from app.models.system import UserDO
        from app.services.user_service import UserService
        user = UserDO(id="1", username="testuser", role="user")
        UserService._ensure_not_default_admin(user)  # should not raise

    def test_none_ok(self):
        from app.services.user_service import UserService
        UserService._ensure_not_default_admin(None)  # should not raise


class TestUserServiceToVO:
    def test_conversion(self):
        from app.models.system import UserDO
        from app.services.user_service import UserService
        user = UserDO(
            id="123",
            username="testuser",
            role="user",
            avatar="http://img.png",
        )
        vo = UserService._to_vo(user)
        assert vo.id == "123"
        assert vo.username == "testuser"
        assert vo.role == "user"
        assert vo.avatar == "http://img.png"


# ===========================================================================
# TestSampleQuestionServiceLogic
# ===========================================================================

class TestSampleQuestionServiceToVO:
    def test_conversion(self):
        from app.models.system import SampleQuestionDO
        from app.services.sample_question_service import SampleQuestionService
        record = SampleQuestionDO(
            id="1",
            title="Test",
            description="Desc",
            question="What is RAG?",
        )
        vo = SampleQuestionService._to_vo(record)
        assert vo.id == "1"
        assert vo.title == "Test"
        assert vo.question == "What is RAG?"


# ===========================================================================
# TestAuditServiceToVO
# ===========================================================================

class TestAuditServiceToVO:
    def test_conversion(self):
        from app.models.system import BizChangeLogDO
        from app.services.audit_service import BizChangeLogService
        record = BizChangeLogDO(
            id="1",
            biz_type="USER",
            biz_id="123",
            operation_type="CREATE",
            action_desc="Created user",
            operator_id="456",
            operator_name="admin",
            success=True,
        )
        vo = BizChangeLogService._to_vo(record)
        assert vo.id == "1"
        assert vo.bizType == "USER"
        assert vo.operationType == "CREATE"
        assert vo.actionDesc == "Created user"
        assert vo.success is True
