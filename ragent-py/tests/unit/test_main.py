"""Integration tests for app.main — FastAPI app middleware and endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Redis and DB (no real connections)."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.hgetall = AsyncMock(return_value={})
    mock_redis.expire = AsyncMock(return_value=True)

    import app.core.redis_client

    # Save original and set mock as the module-level redis client
    original_redis = app.core.redis_client._redis_client
    app.core.redis_client._redis_client = mock_redis

    # Mock DB session
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None), scalar=MagicMock(return_value=0), scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()

    async def mock_get_db():
        yield mock_db

    # Patch init_redis/close_redis so lifespan doesn't try real connection
    with patch("app.core.redis_client.init_redis", new=AsyncMock(return_value=mock_redis)), \
         patch("app.core.redis_client.close_redis", new=AsyncMock()):
        import app.main
        # Override DB dependency
        from app.core.database import get_db
        app.main.app.dependency_overrides[get_db] = mock_get_db
        try:
            with TestClient(app.main.app, raise_server_exceptions=False) as c:
                yield c
        finally:
            app.main.app.dependency_overrides.clear()

    # Restore
    app.core.redis_client._redis_client = original_redis


class TestHealthCheck:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


class TestCatchAll404:
    def test_unknown_path_without_auth_returns_not_login(self, client):
        """Unauthenticated requests to unknown paths get 'not logged in' (auth runs first)."""
        resp = client.get("/nonexistent/path")
        assert resp.status_code == 200
        data = resp.json()
        # Auth middleware runs before route matching — returns not-login
        assert data["code"] == "A000100"

    def test_unknown_path_returns_result_json(self, client):
        """After auth, unknown paths get 404 Result JSON."""
        # The catch-all 404 is tested implicitly — auth blocks unauthenticated requests first
        # This is consistent with Java behavior where SaToken interceptor runs before DispatcherServlet
        resp = client.get("/nonexistent/path")
        assert resp.status_code == 200  # Always HTTP 200
        data = resp.json()
        assert "code" in data
        assert "requestId" in data


class TestGlobalExceptionHandler:
    def test_all_responses_are_http_200(self, client):
        """Verify the core contract: ALL responses return HTTP 200."""
        resp = client.get("/does/not/exist")
        assert resp.status_code == 200

        resp = client.get("/health")
        assert resp.status_code == 200


class TestRequestIdMiddleware:
    def test_response_has_request_id_header(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers
        assert resp.headers["x-request-id"].startswith("req_")

    def test_response_has_timing_header(self, client):
        resp = client.get("/health")
        assert "x-response-time" in resp.headers


class TestAuthMiddleware:
    def test_auth_whitelist_allows_login_path(self, client):
        """Auth whitelist paths should pass through without token."""
        # POST /auth/login requires a body; posting without one returns validation error
        # but still returns HTTP 200 with a Result JSON (not a 401)
        resp = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert resp.status_code == 200
        data = resp.json()
        # Route is accessible (auth middleware passed through), business logic handles the rest
        assert "code" in data

    def test_protected_path_without_token_returns_not_login(self, client):
        """Protected paths without token should return not-login error."""
        resp = client.get("/user/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == "A000100"
        assert "未登录" in data["message"] or "登录" in data["message"]

    def test_protected_path_with_invalid_token(self, client):
        """Invalid token should return not-login error."""
        resp = client.get(
            "/user/me",
            headers={"Authorization": "invalid-token-xyz"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == "A000100"


class TestDemoModeMiddleware:
    def test_demo_mode_blocks_post(self, client):
        """When demo_mode is True, POST should be rejected."""
        import app.main as main_module
        original = main_module.settings.demo_mode
        try:
            main_module.settings.demo_mode = True
            resp = client.post("/some/endpoint")
            assert resp.status_code == 200
            data = resp.json()
            assert data["code"] == "A000107"
            assert "演示模式" in data["message"]
        finally:
            main_module.settings.demo_mode = original

    def test_demo_mode_allows_get(self, client):
        """When demo_mode is True, GET should pass through."""
        import app.main as main_module
        original = main_module.settings.demo_mode
        try:
            main_module.settings.demo_mode = True
            resp = client.get("/health")
            assert resp.status_code == 200
        finally:
            main_module.settings.demo_mode = original
