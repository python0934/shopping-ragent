"""
Phase 7 集成测试 — Agent HTTP 接口.

真跑中间件链（认证 / requestId / demo 模式 / 全局异常）与真 SQL，
只把 Redis 与大模型换成替身。校验点落在前端实际依赖的契约上：
HTTP 一律 200、``code`` 字符串、``data`` 字段名 camelCase。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.enums import AgentMessageStatus
from app.core.error_code import BaseErrorCode
from app.services.agent_conversation_service import AgentConversationService

from .conftest import AUTH_HEADERS, TEST_TOKEN, TEST_USER_ID, run_async

CONV_ID = "conv-integration-1"


def _as_client_executor(fake: Any) -> Any:
    """把测试替身包成注册表认的执行器类型"""
    from app.mcp.registry import McpClientToolExecutor

    return McpClientToolExecutor(None, fake.tool_definition(), "test-server")


def _seed_conversation(client: Any, conversation_id: str = CONV_ID, question: str = "你好") -> str:
    """走真服务落一轮问答，返回提问消息 ID"""

    async def _do() -> str:
        async with client.db_factory() as db:
            service = AgentConversationService(db)
            await service.touch_conversation(TEST_USER_ID, conversation_id, question)
            message_id = await service.add_user_message(TEST_USER_ID, conversation_id, question)
            await service.add_assistant_message(
                user_id=TEST_USER_ID,
                conversation_id=conversation_id,
                reply_to_message_id=message_id,
                content="你好，有什么可以帮你",
                thinking_content=None,
                blocks=[{"kind": "answer", "text": "你好，有什么可以帮你"}],
                message_status=AgentMessageStatus.NORMAL,
            )
            await db.commit()
            return message_id

    return run_async(_do())


# ---------------------------------------------------------------------------
# 统一响应格式
# ---------------------------------------------------------------------------

class TestResponseEnvelope:

    def test_success_envelope_shape(self, client):
        body = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()

        assert body["code"] == "0"
        assert set(body) >= {"code", "message", "data", "requestId"}
        assert body["requestId"]

    def test_all_endpoints_return_http_200(self, client):
        """前端只看 code 字段，异常也必须是 200"""
        responses = [
            client.get("/agent/v1/meta", headers=AUTH_HEADERS),
            client.get("/agent/v1/conversations", headers=AUTH_HEADERS),
            client.get("/agent/v1/conversations/nope/messages", headers=AUTH_HEADERS),
            client.delete("/agent/v1/conversations/nope", headers=AUTH_HEADERS),
        ]

        assert [r.status_code for r in responses] == [200] * len(responses)

    def test_missing_token_is_rejected_with_200(self, client):
        body = client.get("/agent/v1/meta").json()

        assert body["code"] == BaseErrorCode.NOT_LOGIN.code
        assert body["data"] is None

    def test_unknown_token_is_rejected(self, client):
        client.fake_redis.logged_in = False

        body = client.get("/agent/v1/meta", headers={"Authorization": "bad"}).json()

        assert body["code"] == BaseErrorCode.NOT_LOGIN.code

    def test_request_id_differs_per_call(self, client):
        first = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["requestId"]
        second = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["requestId"]

        assert first != second


# ---------------------------------------------------------------------------
# meta 契约
# ---------------------------------------------------------------------------

class TestAgentMeta:

    def test_meta_matches_frontend_contract(self, client):
        data = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["data"]

        assert data["framework"] == "AgentScope ReAct"
        assert isinstance(data["model"], str) and data["model"]
        assert isinstance(data["maxIters"], int) and data["maxIters"] > 0
        assert data["capabilities"][:2] == ["react", "knowledge-base"]
        assert data["toolProvider"] == "native"
        assert data["mcpConfigured"] is False

    def test_meta_reports_mcp_when_configured(self, client, monkeypatch: pytest.MonkeyPatch):
        """意图树配了 MCP 节点且注册表里有对应执行器，capabilities 才多一项"""
        import app.routers.agent as agent_router
        from app.agent.provider import ReActAgentProvider
        from app.mcp.registry import McpToolRegistry
        from app.services.intent_registry import DbIntentNodeRegistry

        from .conftest import FakeMcpExecutor, seed_intent_node

        async def _seed() -> None:
            async with client.db_factory() as db:
                await seed_intent_node(db, "weather", mcp_tool_id="get_weather", name="天气查询")

        run_async(_seed())

        mcp_registry = McpToolRegistry()
        mcp_registry.register(_as_client_executor(FakeMcpExecutor("get_weather")))
        monkeypatch.setattr(agent_router, "_provider", ReActAgentProvider(
            intent_node_registry=DbIntentNodeRegistry(),
            mcp_tool_registry=mcp_registry,
        ))

        data = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["data"]

        assert data["mcpConfigured"] is True
        assert data["toolProvider"] == "native + mcp"
        assert "mcp-tools" in data["capabilities"]

    def test_meta_reports_unavailable_when_server_down(self, client, monkeypatch: pytest.MonkeyPatch):
        """意图树配了但 MCP Server 没连上，不算已配置"""
        import app.routers.agent as agent_router
        from app.agent.provider import ReActAgentProvider
        from app.mcp.registry import McpToolRegistry
        from app.services.intent_registry import DbIntentNodeRegistry

        from .conftest import seed_intent_node

        async def _seed() -> None:
            async with client.db_factory() as db:
                await seed_intent_node(db, "weather", mcp_tool_id="get_weather")

        run_async(_seed())
        monkeypatch.setattr(agent_router, "_provider", ReActAgentProvider(
            intent_node_registry=DbIntentNodeRegistry(),
            mcp_tool_registry=McpToolRegistry(),
        ))

        data = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["data"]

        assert data["mcpConfigured"] is False
        assert data["toolProvider"] == "native"


# ---------------------------------------------------------------------------
# 会话与消息
# ---------------------------------------------------------------------------

class TestConversationApi:

    def test_empty_conversation_list(self, client):
        data = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert data == []

    def test_list_returns_seeded_conversation(self, client):
        _seed_conversation(client)

        data = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert len(data) == 1
        assert data[0]["conversationId"] == CONV_ID
        assert data[0]["title"] == "你好"
        assert data[0]["turns"] == 1
        assert data[0]["lastTime"]

    def test_list_is_scoped_to_current_user(self, client):
        _seed_conversation(client, CONV_ID)

        async def _seed_other() -> None:
            async with client.db_factory() as db:
                service = AgentConversationService(db)
                await service.touch_conversation("9999999999999999999", "other-conv", "别人的")
                await db.commit()

        run_async(_seed_other())

        data = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert [c["conversationId"] for c in data] == [CONV_ID]

    def test_messages_come_in_chronological_order(self, client):
        _seed_conversation(client)

        data = client.get(
            f"/agent/v1/conversations/{CONV_ID}/messages", headers=AUTH_HEADERS,
        ).json()["data"]

        assert [m["role"] for m in data] == ["user", "assistant"]
        assert data[1]["content"] == "你好，有什么可以帮你"
        assert data[1]["blocks"][0]["kind"] == "answer"
        assert data[1]["messageStatus"] == "NORMAL"

    def test_messages_of_foreign_conversation_are_empty(self, client):
        _seed_conversation(client)

        data = client.get(
            "/agent/v1/conversations/someone-elses/messages", headers=AUTH_HEADERS,
        ).json()["data"]

        assert data == []

    def test_rename_updates_title(self, client):
        _seed_conversation(client)

        body = client.put(
            f"/agent/v1/conversations/{CONV_ID}/title",
            headers=AUTH_HEADERS, json={"title": "改过的标题"},
        ).json()
        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert body["code"] == "0"
        assert listed[0]["title"] == "改过的标题"

    def test_rename_rejects_blank_title(self, client):
        _seed_conversation(client)

        body = client.put(
            f"/agent/v1/conversations/{CONV_ID}/title",
            headers=AUTH_HEADERS, json={"title": "   "},
        ).json()

        assert body["code"] == BaseErrorCode.CLIENT_ERROR.code

    def test_rename_clips_overlong_title(self, client):
        _seed_conversation(client)

        client.put(
            f"/agent/v1/conversations/{CONV_ID}/title",
            headers=AUTH_HEADERS, json={"title": "标" * 300},
        )
        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert len(listed[0]["title"]) == 128

    def test_delete_hides_conversation_and_messages(self, client):
        _seed_conversation(client)

        body = client.delete(f"/agent/v1/conversations/{CONV_ID}", headers=AUTH_HEADERS).json()
        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]
        messages = client.get(
            f"/agent/v1/conversations/{CONV_ID}/messages", headers=AUTH_HEADERS,
        ).json()["data"]

        assert body["code"] == "0"
        assert listed == []
        assert messages == []

    def test_delete_unknown_conversation_still_succeeds(self, client):
        """幂等：前端重试删除不该报错"""
        body = client.delete("/agent/v1/conversations/ghost", headers=AUTH_HEADERS).json()

        assert body["code"] == "0"

    def test_batch_delete_removes_all_listed(self, client):
        _seed_conversation(client, "c1", "第一个")
        _seed_conversation(client, "c2", "第二个")
        _seed_conversation(client, "c3", "第三个")

        body = client.post(
            "/agent/v1/conversations/batch-delete",
            headers=AUTH_HEADERS, json={"ids": ["c1", "c3"]},
        ).json()
        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert body["code"] == "0"
        assert [c["conversationId"] for c in listed] == ["c2"]

    def test_batch_delete_with_empty_ids_is_noop(self, client):
        _seed_conversation(client, "c1")

        body = client.post(
            "/agent/v1/conversations/batch-delete", headers=AUTH_HEADERS, json={"ids": []},
        ).json()
        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]

        assert body["code"] == "0"
        assert len(listed) == 1


# ---------------------------------------------------------------------------
# 对话入参校验
# ---------------------------------------------------------------------------

class TestChatValidation:

    def test_blank_question_is_rejected(self, client):
        body = client.get(
            "/agent/v1/chat", headers=AUTH_HEADERS, params={"question": "   "},
        ).json()

        assert body["code"] == BaseErrorCode.CLIENT_ERROR.code
        assert "问题不能为空" in body["message"]

    def test_missing_question_is_validation_error(self, client):
        response = client.get("/agent/v1/chat", headers=AUTH_HEADERS)

        assert response.status_code == 200
        assert response.json()["code"] != "0"

    def test_overlong_question_is_rejected(self, client):
        """8000 汉字装不进 GET 查询串（httpx 有 URL 长度上限），直接打服务层"""
        from app.core.exceptions import ClientException
        from app.services.agent_chat_service import AgentChatService

        async def _call() -> Any:
            async with client.db_factory() as db:
                service = AgentChatService(db, redis=client.fake_redis)
                return await service.stream_chat(TEST_USER_ID, "字" * 8_001)

        with pytest.raises(ClientException) as exc_info:
            run_async(_call())

        assert "8000" in str(exc_info.value.errorMessage)

    def test_stop_requires_task_id(self, client):
        body = client.post("/agent/v1/stop", headers=AUTH_HEADERS, params={}).json()

        assert body["code"] == BaseErrorCode.CLIENT_ERROR.code
        assert "taskId" in body["message"]

    def test_stop_unknown_task_returns_success(self, client):
        """Java 侧签名是 Result<Void>，停不停得掉不往 data 里放东西"""
        body = client.post(
            "/agent/v1/stop", headers=AUTH_HEADERS, params={"taskId": "ghost-task"},
        ).json()

        assert body["code"] == "0"
        assert body["data"] is None


# ---------------------------------------------------------------------------
# 认证与租户隔离
# ---------------------------------------------------------------------------

class TestAuthBoundary:

    def test_every_agent_endpoint_needs_token(self, client):
        calls = [
            lambda: client.get("/agent/v1/meta"),
            lambda: client.get("/agent/v1/conversations"),
            lambda: client.get(f"/agent/v1/conversations/{CONV_ID}/messages"),
            lambda: client.put(f"/agent/v1/conversations/{CONV_ID}/title", json={"title": "x"}),
            lambda: client.delete(f"/agent/v1/conversations/{CONV_ID}"),
            lambda: client.post("/agent/v1/conversations/batch-delete", json={"ids": ["x"]}),
            lambda: client.post("/agent/v1/stop", params={"taskId": "t"}),
        ]

        codes = [call().json()["code"] for call in calls]

        assert codes == [BaseErrorCode.NOT_LOGIN.code] * len(calls)

    def test_token_is_read_without_bearer_prefix(self, client):
        """Java 侧 Authorization 直接放裸 token，加了 Bearer 反而认不出"""
        with_bearer = client.get(
            "/agent/v1/meta", headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        ).json()

        assert with_bearer["code"] == BaseErrorCode.NOT_LOGIN.code
