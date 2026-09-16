"""
端到端测试夹具.

复用集成层的 HTTP / DB / Redis 替身，只额外补两件 E2E 才需要的东西：
一个能按脚本回放的大模型替身，以及一个能把 SSE 字节流还原成事件帧的解析器。

TestClient 的 ``handle_request`` 会把整个 ASGI 调用跑完再返回响应体，
所以这里的「流式」是事后按帧断言，不是边推边收；真要观察中途状态
（如停止）得另起线程并发发请求，见 ``test_agent_stream.py``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pytest

from app.agent.provider import ReActAgentProvider
from app.agent.react_agent import ModelCompletion, ModelToolCall
from app.mcp.registry import McpToolRegistry
from app.services.intent_registry import DbIntentNodeRegistry

# 集成层的夹具直接复用：pytest 按 conftest 命名空间收集，导入即注册
from tests.integration.conftest import (  # noqa: F401
    AUTH_HEADERS,
    TEST_TOKEN,
    TEST_USER_ID,
    FakeMcpExecutor,
    FakeRedis,
    client,
    db_factory,
    run_async,
    seed_builtin_agent,
    seed_intent_node,
)

E2E_PROMPTS = {
    "AGENT_MAIN": "你是端到端测试用的智能体，直接回答用户问题。",
    "KNOWLEDGE_TOOL_DESCRIPTION": "检索企业知识库，覆盖产品与售后类问题。",
}


# ---------------------------------------------------------------------------
# SSE 帧解析
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SseFrame:
    """一条 SSE 事件：事件名即前端 EventSource 的监听名"""

    event: str
    data: Any


def parse_sse(lines: Iterable[str]) -> list[SseFrame]:
    """把 ``iter_lines()`` 的行流还原成事件帧，空行即帧边界"""
    frames: list[SseFrame] = []
    event: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        if event is None and not data_lines:
            return
        payload = "\n".join(data_lines)
        try:
            parsed: Any = json.loads(payload)
        except ValueError:
            parsed = payload
        frames.append(SseFrame(event or "message", parsed))

    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            flush()
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue  # 注释行（心跳），前端也会忽略
        name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            event = value
        elif name == "data":
            data_lines.append(value)

    flush()
    return frames


def stream_frames(
    client: Any,
    question: str,
    conversation_id: str | None = None,
) -> list[SseFrame]:
    """跑完整一轮 SSE 对话，返回全部事件帧"""
    params: dict[str, str] = {"question": question}
    if conversation_id:
        params["conversationId"] = conversation_id

    with client.stream("GET", "/agent/v1/chat", params=params, headers=AUTH_HEADERS) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-accel-buffering"] == "no"
        return parse_sse(resp.iter_lines())


def frames_of(frames: list[SseFrame], event: str) -> list[SseFrame]:
    return [f for f in frames if f.event == event]


def event_names(frames: list[SseFrame]) -> list[str]:
    return [f.event for f in frames]


# ---------------------------------------------------------------------------
# 大模型替身
# ---------------------------------------------------------------------------

class StubModelClient:
    """
    按脚本逐轮回放的大模型替身。

    脚本项可为：
      - ``ModelCompletion``：自动把 thinking / text 切成增量吐给事件桥
      - ``async (on_delta) -> ModelCompletion``：自己控制节奏，取消与故障用例要用

    脚本走完后重复最后一项，避免多轮对话用例越界。
    """

    CHUNK = 4

    def __init__(self, *steps: Any) -> None:
        self._steps = list(steps)
        self.calls: list[dict[str, Any]] = []

    @property
    def rounds(self) -> int:
        return len(self.calls)

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Any | None = None,
    ) -> ModelCompletion:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._steps:
            return ModelCompletion()

        step = self._steps[min(len(self.calls) - 1, len(self._steps) - 1)]
        if callable(step):
            return await step(on_delta)

        completion = step if isinstance(step, ModelCompletion) else ModelCompletion()
        if on_delta is not None:
            if completion.thinking:
                on_delta("think", completion.thinking)
            for start in range(0, len(completion.text), self.CHUNK):
                on_delta("response", completion.text[start:start + self.CHUNK])
        return completion


def answer_step(text: str, thinking: str = "") -> ModelCompletion:
    """一轮直接给终答"""
    return ModelCompletion(text=text, thinking=thinking)


def tool_step(*calls: tuple[str, str, dict[str, Any]]) -> ModelCompletion:
    """一轮要求调工具：``(callId, toolName, arguments)``"""
    return ModelCompletion(tool_calls=[
        ModelToolCall(id=call_id, name=name, arguments=dict(args or {}))
        for call_id, name, args in calls
    ])


# ---------------------------------------------------------------------------
# Agent 装配替身
# ---------------------------------------------------------------------------

@dataclass
class AgentStub:
    """一次装配的可观测句柄"""

    model: StubModelClient
    provider: ReActAgentProvider
    intent_registry: DbIntentNodeRegistry = field(default_factory=DbIntentNodeRegistry)
    mcp_registry: McpToolRegistry = field(default_factory=McpToolRegistry)
    search_calls: list[str] = field(default_factory=list)


@pytest.fixture
def install_agent(client: Any, monkeypatch: pytest.MonkeyPatch) -> Callable[..., AgentStub]:
    """
    把路由里的进程级 provider 换成带替身模型的装配器。

    意图树与 MCP 注册表一律新建，不用进程单例：单例跨用例会把上一个
    用例的工具带进来，meta 的 ``mcpConfigured`` 就飘了。
    """
    import app.routers.agent as agent_routes

    def _install(
        *steps: Any,
        search_reply: str | None = "知识库检索结果",
        intent_registry: DbIntentNodeRegistry | None = None,
        mcp_registry: McpToolRegistry | None = None,
    ) -> AgentStub:
        intents = intent_registry or DbIntentNodeRegistry()
        mcps = mcp_registry or McpToolRegistry()
        model = StubModelClient(*steps)
        stub = AgentStub(
            model=model,
            provider=ReActAgentProvider(
                model_client=model,
                intent_node_registry=intents,
                mcp_tool_registry=mcps,
            ),
            intent_registry=intents,
            mcp_registry=mcps,
        )

        # 知识检索门面换掉：真门面会去查向量库，E2E 关心的是工具帧而不是检索质量
        if search_reply is not None:
            calls = stub.search_calls

            class _StubFacade:
                def __init__(self, **_: Any) -> None:
                    pass

                async def search(self, query: str, history: Any = None) -> str:
                    calls.append(query)
                    return search_reply

            monkeypatch.setattr(
                "app.services.knowledge_search_facade.KnowledgeSearchFacade", _StubFacade,
            )

        monkeypatch.setattr(agent_routes, "_provider", stub.provider)
        return stub

    return _install


@pytest.fixture
def seeded_agent(client: Any) -> str:
    """铺一个带全部必需槽位的内置智能体，返回 agentId"""

    async def _seed() -> str:
        async with client.db_factory() as db:
            return await seed_builtin_agent(db, E2E_PROMPTS)

    return run_async(_seed())
