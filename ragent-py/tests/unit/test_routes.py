"""Unit tests for API route skeleton — verify all endpoints are registered."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked Redis, DB and authenticated user."""
    mock_redis = AsyncMock()
    # Return a valid user for any token
    mock_redis.get = AsyncMock(return_value="test-user-id")
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.hgetall = AsyncMock(return_value={"username": "admin", "role": "admin", "avatar": ""})
    mock_redis.expire = AsyncMock(return_value=True)

    import app.core.redis_client
    original_redis = app.core.redis_client._redis_client
    app.core.redis_client._redis_client = mock_redis

    # Mock DB session
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None),
        scalar=MagicMock(return_value=0),
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
    ))
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()

    async def mock_get_db():
        yield mock_db

    with patch("app.core.redis_client.init_redis", new=AsyncMock(return_value=mock_redis)), \
         patch("app.core.redis_client.close_redis", new=AsyncMock()):
        import app.main
        from app.core.database import get_db
        app.main.app.dependency_overrides[get_db] = mock_get_db
        try:
            with TestClient(app.main.app, raise_server_exceptions=False) as c:
                yield c
        finally:
            app.main.app.dependency_overrides.clear()

    app.core.redis_client._redis_client = original_redis


AUTH_HEADERS = {"Authorization": "test-token"}


class TestRouteRegistration:
    """Verify all expected API routes are registered by making HTTP requests."""

    def test_total_route_count(self, client):
        """At least 76 API endpoints + health + catch-all + openapi = 79+ routes."""
        import app.main
        # Count all routes including nested router routes
        total = 0
        for route in app.main.app.router.routes:
            if hasattr(route, 'original_router'):
                total += len(route.original_router.routes)
            elif hasattr(route, 'methods'):
                total += 1
        assert total >= 76, f"Expected >= 76 routes, got {total}"

    def test_auth_routes_accessible(self, client):
        """Auth endpoints should be accessible without token."""
        resp = client.post("/auth/login", json={"username": "admin", "password": "test"})
        assert resp.status_code == 200
        assert "code" in resp.json()

        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

        resp = client.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_user_routes_respond(self, client):
        resp = client.get("/user/me", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_sample_question_routes_respond(self, client):
        resp = client.get("/sample-questions/random", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert "code" in resp.json()

    def test_conversation_routes_respond(self, client):
        resp = client.get("/conversations", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_knowledge_routes_respond(self, client):
        resp = client.get("/knowledge-base", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_rag_routes_respond(self, client):
        resp = client.get("/rag/settings", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_agent_profile_routes_respond(self, client):
        resp = client.get("/agents", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_intent_tree_routes_respond(self, client):
        resp = client.get("/intent-tree/trees", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_ingestion_routes_respond(self, client):
        resp = client.get("/ingestion/pipelines", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_admin_routes_respond(self, client):
        resp = client.get("/overview", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_mappings_routes_respond(self, client):
        resp = client.get("/mappings", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_audit_routes_respond(self, client):
        resp = client.get("/biz-change-logs", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert "code" in resp.json()

    def test_graph_route_responds(self, client):
        resp = client.get("/admin/kg/graph", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_agent_chat_routes_respond(self, client):
        resp = client.get("/agent/v1/meta", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_agent_conversation_routes_respond(self, client):
        resp = client.get("/agent/v1/conversations", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_rag_trace_routes_respond(self, client):
        resp = client.get("/rag/traces/runs", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_ingestion_tasks_respond(self, client):
        resp = client.get("/ingestion/tasks", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_performance_route_responds(self, client):
        resp = client.get("/performance", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_trends_route_responds(self, client):
        resp = client.get("/trends?metric=test", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"

    def test_knowledge_docs_search_responds(self, client):
        resp = client.get("/knowledge-base/docs/search", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert "code" in resp.json()

    def test_rag_eval_responds(self, client):
        resp = client.get("/rag/eval?question=test", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["code"] == "0"
