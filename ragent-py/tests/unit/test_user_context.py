"""Unit tests for app.core.user_context — async-safe user context."""

import pytest

from app.core.exceptions import ClientException
from app.core.user_context import LoginUser, UserContext


class TestLoginUser:
    def test_create_login_user(self):
        user = LoginUser(
            userId="123",
            username="test",
            role="user",
            avatar="https://example.com/a.png",
        )
        assert user.userId == "123"
        assert user.username == "test"
        assert user.role == "user"
        assert user.avatar == "https://example.com/a.png"

    def test_default_avatar_empty(self):
        user = LoginUser(userId="1", username="x", role="user")
        assert user.avatar == ""


class TestUserContext:
    def test_set_and_get(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.get() is mock_login_user

    def test_get_returns_none_when_empty(self):
        assert UserContext.get() is None

    def test_has_user(self, mock_login_user):
        assert UserContext.has_user() is False
        UserContext.set(mock_login_user)
        assert UserContext.has_user() is True

    def test_require_user_success(self, mock_login_user):
        UserContext.set(mock_login_user)
        user = UserContext.require_user()
        assert user is mock_login_user

    def test_require_user_raises(self):
        with pytest.raises(ClientException, match="未获取到当前登录用户"):
            UserContext.require_user()

    def test_get_user_id(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.get_user_id() == "1234567890123456789"

    def test_get_user_id_none(self):
        assert UserContext.get_user_id() is None

    def test_get_username(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.get_username() == "testuser"

    def test_get_role(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.get_role() == "user"

    def test_get_avatar(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.get_avatar() == "https://example.com/avatar.png"

    def test_clear(self, mock_login_user):
        UserContext.set(mock_login_user)
        assert UserContext.has_user() is True
        UserContext.clear()
        assert UserContext.has_user() is False
        assert UserContext.get() is None

    def test_default_avatar_constant(self):
        assert UserContext.DEFAULT_AVATAR == "https://avatars.githubusercontent.com/u/583231?v=4"
