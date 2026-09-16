"""
Phase 5 单元测试 — Agent 运行时.

覆盖 tools / react_agent / event_bridge / run_gate / schemas / prompt_service。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.agent.enums import AgentEventType, AgentMessageStatus, AgentSSEEventType, ToolResultState
from app.agent.event_bridge import (
    FALLBACK_CALL_KEY,
    TOOL_RESULT_MAX_CHARS,
    AgentRunHandle,
    AgentStreamEventBridge,
)
from app.agent.messages import Msg, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from app.agent.enums import MsgRole
from app.agent.react_agent import (
    AgentEvent,
    ModelCompletion,
    ModelToolCall,
    ReActAgent,
    msg_to_openai,
)
from app.agent.run_gate import RUNNING_KEY_PREFIX, SLOT_SEPARATOR, AgentRunGate
from app.agent.tools import (
    AgentTool,
    AgentToolCatalog,
    KnowledgeSearchTool,
    McpToolBridge,
    ToolCallParam,
    RuntimeContext,
)
from app.core.exceptions import ClientException
from app.schemas.agent import (
    AgentBlock,
    AgentCompletionPayload,
    AgentMessageVO,
    AgentMetaPayload,
    AgentToolProgress,
)
from app.services.prompt_service import (
    AgentPromptSlot,
    OrchestrationMode,
    PromptSlotGroup,
    PromptTemplateUtils,
)


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class _FakeStateStore:
    """内存状态存储，替掉 PgAgentStateStore 免得拖数据库进来"""

    def __init__(self, initial: list[Any] | None = None) -> None:
        self.store: dict[tuple[str, str, str], Any] = {}
        self.saved = 0
        if initial:
            self.store[("u1", "s1", "context")] = initial

    async def get_list(self, user_id: str, session_id: str, key: str) -> list[Any]:
        value = self.store.get((user_id, session_id, key))
        return list(value) if isinstance(value, list) else []

    async def save_list(self, user_id: str, session_id: str, key: str, values: list[Any]) -> None:
        self.store[(user_id, session_id, key)] = list(values)
        self.saved += 1


class _ScriptedModel:
    """按脚本回放补全结果，并把增量逐字符吐给回调以模拟流式"""

    def __init__(self, completions: list[ModelCompletion]) -> None:
        self.completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Any = None,
    ) -> ModelCompletion:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.completions:
            return ModelCompletion(text="(脚本耗尽)")
        completion = self.completions.pop(0)
        if on_delta:
            for char in completion.thinking:
                on_delta("think", char)
            for char in completion.text:
                on_delta("response", char)
        return completion


class _EchoTool(AgentTool):
    """回声工具：把入参 query 原样返回，可注入失败"""

    def __init__(self, fail: bool = False, payload: str | None = None) -> None:
        self.fail = fail
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "回声"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    async def call(self, param: ToolCallParam) -> ToolResultBlock:
        self.calls.append(dict(param.input or {}))
        text = self.payload if self.payload is not None else str((param.input or {}).get("query", ""))
        return AgentTool.build_result(param.tool_call_id, self.name, text, self.fail)


class _RecordingSender:
    """记录 SSE 事件名与载荷，形态对齐 SseSender"""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self._closed = False
        self.completed = False
        self.failed: Exception | None = None

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def send_event(self, event_name: str, data: Any) -> None:
        if self._closed:
            return
        self.events.append((event_name, data))

    async def complete(self) -> None:
        self._closed = True
        self.completed = True

    async def fail(self, error: Exception | None = None) -> None:
        self._closed = True
        self.failed = error

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def payloads(self, event_name: str) -> list[Any]:
        return [data for name, data in self.events if name == event_name]


class _FakeRedis:
    """最小 Redis 桩：SET NX PX / GET / EVAL(CAS DEL)"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.eval_calls = 0

    async def set(self, key: str, value: str, nx: bool = False, px: int | None = None) -> bool:
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def eval(self, script: str, numkeys: int, key: str, expected: str) -> int:
        self.eval_calls += 1
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0


async def _collect(agen: Any) -> list[AgentEvent]:
    return [event async for event in agen]


def _tool_completion(call_id: str = "call-1", name: str = "echo_tool", query: str = "hi") -> ModelCompletion:
    return ModelCompletion(tool_calls=[ModelToolCall(id=call_id, name=name, arguments={"query": query})])


# ---------------------------------------------------------------------------
# tools — KnowledgeSearchTool
# ---------------------------------------------------------------------------

class _FakeFacade:
    def __init__(self, result: str = "检索结果", raises: bool = False) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def search(self, query: str, history: list[dict[str, str]]) -> str:
        self.calls.append((query, history))
        if self.raises:
            raise RuntimeError("kb down")
        return self.result


class TestKnowledgeSearchTool:

    def _tool(self, facade: Any = None, loader: Any = None) -> KnowledgeSearchTool:
        return KnowledgeSearchTool("工具描述", facade or _FakeFacade(), loader)

    def test_contract_constants_match_java(self):
        assert KnowledgeSearchTool.TOOL_NAME == "search_knowledge"
        assert KnowledgeSearchTool.DISPLAY_NAME == "知识库检索"
        assert KnowledgeSearchTool.REWRITE_CONTEXT_TURNS == 2

    def test_name_and_description(self):
        tool = self._tool()
        assert tool.name == "search_knowledge"
        assert tool.description == "工具描述"

    def test_parameters_schema_is_closed(self):
        schema = self._tool().parameters()
        assert schema["type"] == "object"
        assert schema["required"] == ["query"]
        # 关掉额外属性，模型乱塞参数时供应商会直接拒
        assert schema["additionalProperties"] is False
        assert schema["properties"]["query"]["type"] == "string"

    def test_is_read_only(self):
        assert self._tool().is_read_only() is True

    def test_function_schema_shape(self):
        schema = self._tool().to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_knowledge"
        assert schema["function"]["description"] == "工具描述"

    @pytest.mark.asyncio
    async def test_call_returns_facade_result(self):
        facade = _FakeFacade("答案文本")
        tool = self._tool(facade)

        result = await tool.call(ToolCallParam(
            tool_call_id="c1", input={"query": "  问题  "},
            runtime_context=RuntimeContext(user_id="u1", session_id="s1"),
        ))

        assert result.state == ToolResultState.SUCCESS
        assert result.output_text() == "答案文本"
        assert result.id == "c1"
        # 入参两侧空白必须削掉，否则改写模型会把它当内容
        assert facade.calls[0][0] == "问题"

    @pytest.mark.asyncio
    async def test_call_rejects_blank_query(self):
        tool = self._tool()
        result = await tool.call(ToolCallParam(tool_call_id="c1", input={"query": "   "}))

        assert result.state == ToolResultState.ERROR
        assert "query" in result.output_text()

    @pytest.mark.asyncio
    async def test_call_rejects_missing_param(self):
        result = await self._tool().call(ToolCallParam(tool_call_id="c1", input={}))
        assert result.state == ToolResultState.ERROR

    @pytest.mark.asyncio
    async def test_call_rejects_none_param(self):
        result = await self._tool().call(None)  # type: ignore[arg-type]
        assert result.state == ToolResultState.ERROR

    @pytest.mark.asyncio
    async def test_call_wraps_facade_error(self):
        """检索异常必须转成工具错误结果，抛出会掀翻整轮 ReAct"""
        result = await self._tool(_FakeFacade(raises=True)).call(
            ToolCallParam(tool_call_id="c1", input={"query": "q"})
        )
        assert result.state == ToolResultState.ERROR
        assert "异常" in result.output_text()

    @pytest.mark.asyncio
    async def test_call_passes_recent_turns_to_facade(self):
        history = [{"role": "user", "content": "上文"}, {"role": "assistant", "content": "答复"}]

        async def loader(session_id: str, user_id: str, turns: int) -> list[dict[str, str]]:
            assert turns == KnowledgeSearchTool.REWRITE_CONTEXT_TURNS
            assert session_id == "s1"
            return history

        facade = _FakeFacade()
        tool = self._tool(facade, loader)
        await tool.call(ToolCallParam(
            tool_call_id="c1", input={"query": "q"},
            runtime_context=RuntimeContext(user_id="u1", session_id="s1"),
        ))

        assert facade.calls[0][1] == history

    @pytest.mark.asyncio
    async def test_call_degrades_to_no_history_without_runtime_context(self):
        facade = _FakeFacade()
        await self._tool(facade).call(ToolCallParam(tool_call_id="c1", input={"query": "q"}))
        assert facade.calls[0][1] == []

    @pytest.mark.asyncio
    async def test_call_degrades_when_history_loader_fails(self):
        """加载历史失败不该连带检索失败，退化成无历史改写继续跑"""
        async def boom(*args: Any) -> list[dict[str, str]]:
            raise RuntimeError("db down")

        facade = _FakeFacade("ok")
        tool = self._tool(facade, boom)
        result = await tool.call(ToolCallParam(
            tool_call_id="c1", input={"query": "q"},
            runtime_context=RuntimeContext(user_id="u1", session_id="s1"),
        ))

        assert result.state == ToolResultState.SUCCESS
        assert facade.calls[0][1] == []


# ---------------------------------------------------------------------------
# tools — McpToolBridge
# ---------------------------------------------------------------------------

class _FakeExecutor:
    def __init__(
        self,
        tool_id: str = "mcp.weather",
        definition: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        raises: bool = False,
    ) -> None:
        self._tool_id = tool_id
        self._definition = definition if definition is not None else {
            "name": tool_id,
            "description": "服务端描述",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "annotations": {"readOnlyHint": True},
        }
        self._result = result if result is not None else {
            "content": [{"type": "text", "text": "晴 25 度"}],
            "isError": False,
        }
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    @property
    def tool_id(self) -> str:
        return self._tool_id

    def tool_definition(self) -> dict[str, Any]:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        if self.raises:
            raise RuntimeError("mcp down")
        return self._result


class TestMcpToolBridge:

    def test_name_comes_from_executor_tool_id(self):
        assert McpToolBridge(_FakeExecutor()).name == "mcp.weather"

    def test_description_override_wins(self):
        bridge = McpToolBridge(_FakeExecutor(), "意图树里的描述")
        assert bridge.description == "意图树里的描述"

    def test_description_falls_back_to_server_side(self):
        assert McpToolBridge(_FakeExecutor(), "  ").description == "服务端描述"

    def test_parameters_pass_through_input_schema(self):
        params = McpToolBridge(_FakeExecutor()).parameters()
        assert params["type"] == "object"
        assert params["properties"] == {"city": {"type": "string"}}
        assert params["required"] == ["city"]

    def test_parameters_tolerate_missing_schema(self):
        bridge = McpToolBridge(_FakeExecutor(definition={"description": "d"}))
        params = bridge.parameters()
        assert params == {"type": "object", "properties": {}}

    def test_read_only_hint_passes_through(self):
        assert McpToolBridge(_FakeExecutor()).is_read_only() is True

    def test_missing_read_only_hint_defaults_to_write(self):
        """猜错的两个方向不对等：把写工具当只读会放过重复副作用"""
        bridge = McpToolBridge(_FakeExecutor(definition={
            "description": "d", "inputSchema": {"type": "object", "properties": {}},
        }))
        assert bridge.is_read_only() is False

    @pytest.mark.asyncio
    async def test_call_extracts_text_content(self):
        executor = _FakeExecutor()
        result = await McpToolBridge(executor).call(
            ToolCallParam(tool_call_id="c1", input={"city": "北京"})
        )

        assert result.state == ToolResultState.SUCCESS
        assert result.output_text() == "晴 25 度"
        assert executor.calls == [{"city": "北京"}]

    @pytest.mark.asyncio
    async def test_call_joins_multiple_text_chunks(self):
        executor = _FakeExecutor(result={
            "content": [{"type": "text", "text": "甲"}, {"type": "text", "text": "乙"}],
            "isError": False,
        })
        result = await McpToolBridge(executor).call(ToolCallParam(tool_call_id="c1", input={}))
        assert result.output_text() == "甲\n乙"

    @pytest.mark.asyncio
    async def test_call_reports_empty_content(self):
        executor = _FakeExecutor(result={"content": [], "isError": False})
        result = await McpToolBridge(executor).call(ToolCallParam(tool_call_id="c1", input={}))
        assert result.output_text() == "（工具无返回内容）"

    @pytest.mark.asyncio
    async def test_call_propagates_is_error_flag(self):
        executor = _FakeExecutor(result={
            "content": [{"type": "text", "text": "失败了"}], "isError": True,
        })
        result = await McpToolBridge(executor).call(ToolCallParam(tool_call_id="c1", input={}))
        assert result.state == ToolResultState.ERROR

    @pytest.mark.asyncio
    async def test_call_wraps_execution_error(self):
        result = await McpToolBridge(_FakeExecutor(raises=True)).call(
            ToolCallParam(tool_call_id="c1", input={})
        )
        assert result.state == ToolResultState.ERROR
        assert "异常" in result.output_text()

    @pytest.mark.asyncio
    async def test_call_tolerates_none_input(self):
        executor = _FakeExecutor()
        await McpToolBridge(executor).call(ToolCallParam(tool_call_id="c1"))
        assert executor.calls == [{}]


# ---------------------------------------------------------------------------
# tools — AgentToolCatalog
# ---------------------------------------------------------------------------

class _CatalogPromptResolver:
    def __init__(self, description: str = "知识库工具描述") -> None:
        self.description = description

    async def resolve_knowledge_tool_description(self) -> str:
        return self.description


class _FakeIntentRegistry:
    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self._nodes = nodes

    def list_mcp_tool_nodes(self) -> list[dict[str, Any]]:
        return self._nodes


class _FakeMcpRegistry:
    def __init__(self, executors: list[Any]) -> None:
        self._executors = executors

    def list_all_executors(self) -> list[Any]:
        return self._executors


class TestAgentToolCatalog:

    def _catalog(self, **kwargs: Any) -> AgentToolCatalog:
        base: dict[str, Any] = {
            "prompt_resolver": _CatalogPromptResolver(),
            "search_facade": _FakeFacade(),
        }
        base.update(kwargs)
        return AgentToolCatalog(**base)

    @pytest.mark.asyncio
    async def test_resolve_always_includes_knowledge_tool(self):
        catalog = await self._catalog().resolve()

        assert catalog.knowledge_tool_description == "知识库工具描述"
        assert catalog.display_name_of("search_knowledge") == "知识库检索"
        assert catalog.bindings == ()

    @pytest.mark.asyncio
    async def test_resolve_rejects_blank_tool_description(self):
        """描述为空等于让模型盲猜工具用途，宁可装配失败"""
        with pytest.raises(RuntimeError):
            await self._catalog(prompt_resolver=_CatalogPromptResolver("   ")).resolve()

    @pytest.mark.asyncio
    async def test_resolve_binds_configured_and_available_tools(self):
        catalog = await self._catalog(
            intent_node_registry=_FakeIntentRegistry([
                {"mcpToolId": "mcp.weather", "name": "天气", "description": "查天气"},
            ]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        ).resolve()

        assert len(catalog.bindings) == 1
        assert catalog.bindings[0].tool_id == "mcp.weather"
        assert catalog.bindings[0].display_name == "天气"
        assert catalog.unavailable_tool_ids == ()
        assert catalog.display_name_of("mcp.weather") == "天气"

    @pytest.mark.asyncio
    async def test_resolve_collects_unavailable_tool_ids(self):
        """配了但当前没执行器的工具要显式收进不可用名单，不能静默丢掉"""
        catalog = await self._catalog(
            intent_node_registry=_FakeIntentRegistry([
                {"mcpToolId": "mcp.gone", "name": "已下线", "description": "d"},
            ]),
            mcp_tool_registry=_FakeMcpRegistry([]),
        ).resolve()

        assert catalog.unavailable_tool_ids == ("mcp.gone",)
        assert catalog.bindings == ()

    @pytest.mark.asyncio
    async def test_resolve_falls_back_to_tool_id_as_display_name(self):
        catalog = await self._catalog(
            intent_node_registry=_FakeIntentRegistry([{"mcpToolId": "mcp.weather", "name": "  "}]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        ).resolve()

        assert catalog.bindings[0].display_name == "mcp.weather"

    @pytest.mark.asyncio
    async def test_resolve_merges_descriptions_from_duplicate_nodes(self):
        catalog = await self._catalog(
            intent_node_registry=_FakeIntentRegistry([
                {"mcpToolId": "mcp.weather", "name": "天气", "description": "甲"},
                {"mcpToolId": "mcp.weather", "name": "天气别名", "description": "乙"},
                {"mcpToolId": "mcp.weather", "name": "天气", "description": "甲"},
            ]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        ).resolve()

        assert len(catalog.bindings) == 1
        # 重复描述去重，首个非空名胜出
        assert catalog.bindings[0].description == "甲\n乙"
        assert catalog.bindings[0].display_name == "天气"

    @pytest.mark.asyncio
    async def test_resolve_skips_nodes_without_tool_id(self):
        catalog = await self._catalog(
            intent_node_registry=_FakeIntentRegistry([{"mcpToolId": "  ", "name": "无 ID"}]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        ).resolve()

        assert catalog.bindings == ()
        assert catalog.unavailable_tool_ids == ()

    @pytest.mark.asyncio
    async def test_display_name_of_unknown_tool_returns_raw_name(self):
        catalog = await self._catalog().resolve()
        assert catalog.display_name_of("not_registered") == "not_registered"

    @pytest.mark.asyncio
    async def test_build_tools_orders_knowledge_first(self):
        catalog_builder = self._catalog(
            intent_node_registry=_FakeIntentRegistry([
                {"mcpToolId": "mcp.weather", "name": "天气", "description": "d"},
            ]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        )
        catalog = await catalog_builder.resolve()
        tools = catalog_builder.build_tools(catalog)

        assert [t.name for t in tools] == ["search_knowledge", "mcp.weather"]
        assert isinstance(tools[1], McpToolBridge)

    def test_mcp_tool_count_zero_without_registries(self):
        assert self._catalog().mcp_tool_count() == 0

    def test_mcp_tool_count_counts_intersection_only(self):
        catalog = self._catalog(
            intent_node_registry=_FakeIntentRegistry([
                {"mcpToolId": "mcp.weather", "name": "天气"},
                {"mcpToolId": "mcp.gone", "name": "已下线"},
            ]),
            mcp_tool_registry=_FakeMcpRegistry([_FakeExecutor()]),
        )
        assert catalog.mcp_tool_count() == 1


# ---------------------------------------------------------------------------
# react_agent — msg_to_openai
# ---------------------------------------------------------------------------

class TestMsgToOpenAi:

    def test_user_message(self):
        assert msg_to_openai(Msg.user("q")) == [{"role": "user", "content": "q"}]

    def test_system_message(self):
        assert msg_to_openai(Msg.system("p")) == [{"role": "system", "content": "p"}]

    def test_plain_assistant_message(self):
        assert msg_to_openai(Msg.assistant(text="a")) == [{"role": "assistant", "content": "a"}]

    def test_assistant_thinking_maps_to_reasoning_content(self):
        payload = msg_to_openai(Msg.assistant(text="a", thinking="t"))[0]
        assert payload["reasoning_content"] == "t"
        assert payload["content"] == "a"

    def test_tool_use_maps_to_tool_calls(self):
        msg = Msg(role=MsgRole.ASSISTANT, content=[
            ToolUseBlock(id="c1", name="echo_tool", input={"query": "q"}),
        ])
        payload = msg_to_openai(msg)[0]

        assert payload["content"] is None
        assert payload["tool_calls"][0]["id"] == "c1"
        assert payload["tool_calls"][0]["type"] == "function"
        assert payload["tool_calls"][0]["function"]["name"] == "echo_tool"
        assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"query": "q"}

    def test_tool_result_maps_to_role_tool(self):
        msg = Msg(role=MsgRole.TOOL, content=[
            ToolResultBlock(id="c1", name="echo_tool", output=[TextBlock(text="r")]),
        ])
        assert msg_to_openai(msg) == [{"role": "tool", "tool_call_id": "c1", "content": "r"}]

    def test_mixed_msg_splits_into_assistant_and_tool_messages(self):
        """一条 Msg 同时带发起与结果时，OpenAI 协议里属两种消息，必须拆开"""
        msg = Msg(role=MsgRole.ASSISTANT, content=[
            ToolUseBlock(id="c1", name="t", input={}),
            ToolResultBlock(id="c1", name="t", output=[TextBlock(text="r")]),
        ])
        payloads = msg_to_openai(msg)

        assert len(payloads) == 2
        assert payloads[0]["role"] == "assistant"
        assert payloads[1]["role"] == "tool"

    def test_empty_assistant_content_is_empty_string(self):
        """空 assistant 消息给 None 会被部分供应商判 400"""
        assert msg_to_openai(Msg(role=MsgRole.ASSISTANT, content=[]))[0]["content"] == ""


# ---------------------------------------------------------------------------
# react_agent — 循环
# ---------------------------------------------------------------------------

class TestReActAgent:

    def _agent(
        self,
        model: Any,
        tools: list[AgentTool] | None = None,
        state_store: Any = None,
        max_iters: int = 5,
        max_retries: int = 0,
    ) -> ReActAgent:
        return ReActAgent(
            model_client=model,
            tools=tools if tools is not None else [_EchoTool()],
            system_prompt="你是助手",
            state_store=state_store or _FakeStateStore(),
            max_iters=max_iters,
            max_retries=max_retries,
        )

    @pytest.mark.asyncio
    async def test_direct_answer_without_tools(self):
        model = _ScriptedModel([ModelCompletion(text="答案")])
        agent = self._agent(model)

        events = await _collect(agent.stream_events("问题", RuntimeContext("u1", "s1")))
        types = [e.type for e in events]

        assert AgentEventType.TEXT_BLOCK_DELTA in types
        assert AgentEventType.AGENT_RESULT in types
        assert AgentEventType.TOOL_CALL_START not in types
        assert events[-1].result.get_text_content() == "答案"

    @pytest.mark.asyncio
    async def test_deltas_are_emitted_incrementally(self):
        model = _ScriptedModel([ModelCompletion(text="abc")])
        events = await _collect(self._agent(model).stream_events("q", RuntimeContext("u1", "s1")))

        deltas = [e.delta for e in events if e.type == AgentEventType.TEXT_BLOCK_DELTA]
        assert deltas == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_thinking_deltas_use_their_own_event_type(self):
        model = _ScriptedModel([ModelCompletion(text="a", thinking="xy")])
        events = await _collect(self._agent(model).stream_events("q", RuntimeContext("u1", "s1")))

        thinking = [e.delta for e in events if e.type == AgentEventType.THINKING_BLOCK_DELTA]
        assert thinking == ["x", "y"]

    @pytest.mark.asyncio
    async def test_system_prompt_is_first_upstream_message(self):
        model = _ScriptedModel([ModelCompletion(text="a")])
        await _collect(self._agent(model).stream_events("q", RuntimeContext("u1", "s1")))

        assert model.calls[0]["messages"][0] == {"role": "system", "content": "你是助手"}

    @pytest.mark.asyncio
    async def test_tool_definitions_are_sent_to_model(self):
        model = _ScriptedModel([ModelCompletion(text="a")])
        await _collect(self._agent(model).stream_events("q", RuntimeContext("u1", "s1")))

        names = [t["function"]["name"] for t in model.calls[0]["tools"]]
        assert names == ["echo_tool"]

    @pytest.mark.asyncio
    async def test_tool_call_round_trip(self):
        tool = _EchoTool(payload="工具产物")
        model = _ScriptedModel([
            _tool_completion(query="北京"),
            ModelCompletion(text="最终答案"),
        ])
        agent = self._agent(model, [tool])

        events = await _collect(agent.stream_events("q", RuntimeContext("u1", "s1")))
        types = [e.type for e in events]

        assert types.count(AgentEventType.TOOL_CALL_START) == 1
        assert types.count(AgentEventType.TOOL_RESULT_END) == 1
        assert events[-1].result.get_text_content() == "最终答案"
        assert tool.calls == [{"query": "北京"}]

        end = next(e for e in events if e.type == AgentEventType.TOOL_RESULT_END)
        assert end.result_text == "工具产物"
        assert end.state == ToolResultState.SUCCESS

    @pytest.mark.asyncio
    async def test_second_round_sees_tool_result(self):
        model = _ScriptedModel([_tool_completion(query="北京"), ModelCompletion(text="答")])
        await _collect(self._agent(model, [_EchoTool(payload="产物")]).stream_events(
            "q", RuntimeContext("u1", "s1"),
        ))

        second_round = model.calls[1]["messages"]
        assert {"role": "tool", "tool_call_id": "call-1", "content": "产物"} in second_round

    @pytest.mark.asyncio
    async def test_tool_error_is_reported_not_raised(self):
        """工具失败要作为错误结果回给模型，让它自己决定怎么补救"""
        model = _ScriptedModel([_tool_completion(), ModelCompletion(text="兜底")])
        events = await _collect(self._agent(model, [_EchoTool(fail=True)]).stream_events(
            "q", RuntimeContext("u1", "s1"),
        ))

        end = next(e for e in events if e.type == AgentEventType.TOOL_RESULT_END)
        assert end.state == ToolResultState.ERROR
        assert events[-1].result.get_text_content() == "兜底"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self):
        model = _ScriptedModel([
            _tool_completion(name="not_registered"),
            ModelCompletion(text="答"),
        ])
        events = await _collect(self._agent(model).stream_events("q", RuntimeContext("u1", "s1")))

        end = next(e for e in events if e.type == AgentEventType.TOOL_RESULT_END)
        assert end.state == ToolResultState.ERROR
        assert "未知工具" in (end.result_text or "")

    @pytest.mark.asyncio
    async def test_exceed_max_iters_emits_hint_then_summary(self):
        """跑满迭代仍要给结论，只提示不判失败"""
        model = _ScriptedModel([
            _tool_completion(call_id="c1"),
            _tool_completion(call_id="c2"),
            ModelCompletion(text="被迫总结"),
        ])
        events = await _collect(self._agent(model, max_iters=2).stream_events(
            "q", RuntimeContext("u1", "s1"),
        ))
        types = [e.type for e in events]

        assert AgentEventType.EXCEED_MAX_ITERS in types
        assert types.index(AgentEventType.EXCEED_MAX_ITERS) < types.index(AgentEventType.AGENT_RESULT)
        assert events[-1].result.get_text_content() == "被迫总结"
        # 总结轮不带工具，逼模型收口
        assert model.calls[-1]["tools"] == []

    @pytest.mark.asyncio
    async def test_context_is_persisted_after_run(self):
        store = _FakeStateStore()
        model = _ScriptedModel([ModelCompletion(text="答")])
        await _collect(self._agent(model, state_store=store).stream_events("q", RuntimeContext("u1", "s1")))

        assert store.saved == 1
        saved = store.store[("u1", "s1", "context")]
        assert saved[0]["role"] == "user"
        assert saved[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_context_is_reloaded_across_runs(self):
        store = _FakeStateStore()
        agent = self._agent(_ScriptedModel([ModelCompletion(text="一")]), state_store=store)
        await _collect(agent.stream_events("第一问", RuntimeContext("u1", "s1")))

        model2 = _ScriptedModel([ModelCompletion(text="二")])
        agent2 = self._agent(model2, state_store=store)
        await _collect(agent2.stream_events("第二问", RuntimeContext("u1", "s1")))

        contents = [m["content"] for m in model2.calls[0]["messages"] if m["role"] == "user"]
        assert "第一问" in contents
        assert "第二问" in contents

    @pytest.mark.asyncio
    async def test_retry_on_model_failure(self):
        class _FlakyModel:
            def __init__(self) -> None:
                self.attempts = 0

            async def stream_completion(self, messages: Any, tools: Any, on_delta: Any = None) -> ModelCompletion:
                self.attempts += 1
                if self.attempts < 3:
                    raise RuntimeError("超时")
                return ModelCompletion(text="第三次成功")

        model = _FlakyModel()
        events = await _collect(self._agent(model, max_retries=2).stream_events("q", RuntimeContext("u1", "s1")))

        assert model.attempts == 3
        assert events[-1].result.get_text_content() == "第三次成功"

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        class _DeadModel:
            async def stream_completion(self, *args: Any) -> ModelCompletion:
                raise RuntimeError("永久失败")

        with pytest.raises(Exception):
            await _collect(self._agent(_DeadModel(), max_retries=1).stream_events(
                "q", RuntimeContext("u1", "s1"),
            ))

    @pytest.mark.asyncio
    async def test_interrupt_stops_loop(self):
        tool = _EchoTool()
        model = _ScriptedModel([
            _tool_completion(call_id="c1"),
            _tool_completion(call_id="c2"),
            ModelCompletion(text="不该到这"),
        ])
        agent = self._agent(model, [tool])

        # 第一次工具执行时打断
        original_call = tool.call

        async def call_then_interrupt(param: ToolCallParam) -> ToolResultBlock:
            result = await original_call(param)
            agent.interrupt("u1", "s1")
            return result

        tool.call = call_then_interrupt  # type: ignore[method-assign]

        events = await _collect(agent.stream_events("q", RuntimeContext("u1", "s1")))
        assert all(e.result is None or e.result.get_text_content() != "不该到这" for e in events)
        assert not agent.is_interrupted("u1", "s1"), "收尾必须清掉打断旗标"

    @pytest.mark.asyncio
    async def test_interrupt_flag_is_scoped_to_session(self):
        agent = self._agent(_ScriptedModel([ModelCompletion(text="a")]))
        agent.interrupt("u1", "s1")

        assert agent.is_interrupted("u1", "s1")
        assert not agent.is_interrupted("u1", "s2")

        agent.clear_interrupt("u1", "s1")
        assert not agent.is_interrupted("u1", "s1")

    @pytest.mark.asyncio
    async def test_memory_middleware_runs_before_each_reasoning(self):
        calls: list[int] = []

        class _SpyMemory:
            async def before_reasoning(self, context: Any, user_id: str, session_id: str) -> bool:
                calls.append(len(context))
                return False

        agent = self._agent(_ScriptedModel([_tool_completion(), ModelCompletion(text="a")]))
        agent.memory = _SpyMemory()  # type: ignore[assignment]
        await _collect(agent.stream_events("q", RuntimeContext("u1", "s1")))

        assert len(calls) == 2
        # 第二轮上下文里已经多了工具调用与结果
        assert calls[1] > calls[0]


# ---------------------------------------------------------------------------
# event_bridge
# ---------------------------------------------------------------------------

class TestAgentStreamEventBridge:

    def _bridge(self, sender: Any = None, persist: Any = None) -> AgentStreamEventBridge:
        async def default_persist(content: str, thinking: str, blocks: Any, status: Any) -> str:
            return "msg-1"

        return AgentStreamEventBridge(
            sender=sender or _RecordingSender(),
            conversation_id="conv-1",
            task_id="task-1",
            persist_callback=persist or default_persist,
        )

    @staticmethod
    async def _events(*events: AgentEvent) -> Any:
        async def gen() -> Any:
            for event in events:
                yield event
        return gen()

    @pytest.mark.asyncio
    async def test_meta_frame_contract(self):
        sender = _RecordingSender()
        await self._bridge(sender).send_meta()

        assert sender.names() == [AgentSSEEventType.META.value]
        assert sender.payloads("meta")[0] == {"conversationId": "conv-1", "taskId": "task-1"}

    @pytest.mark.asyncio
    async def test_text_delta_becomes_message_frame(self):
        sender = _RecordingSender()
        bridge = self._bridge(sender)

        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="你"),
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="好"),
        ))

        assert sender.payloads("message") == [
            {"type": "response", "delta": "你"},
            {"type": "response", "delta": "好"},
        ]
        assert bridge.content == "你好"

    @pytest.mark.asyncio
    async def test_thinking_delta_uses_think_type(self):
        sender = _RecordingSender()
        bridge = self._bridge(sender)

        await bridge.run(await self._events(
            AgentEvent(AgentEventType.THINKING_BLOCK_DELTA, delta="想"),
        ))

        assert sender.payloads("message") == [{"type": "think", "delta": "想"}]
        assert bridge.thinking_content == "想"
        # 思考不算正文
        assert bridge.content == ""

    @pytest.mark.asyncio
    async def test_empty_delta_is_not_forwarded(self):
        sender = _RecordingSender()
        await self._bridge(sender).run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta=""),
        ))
        assert sender.payloads("message") == []

    @pytest.mark.asyncio
    async def test_tool_progress_frames(self):
        sender = _RecordingSender()
        bridge = self._bridge(sender)
        bridge.set_display_name_resolver(lambda n: "回声工具")

        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="echo_tool"),
            AgentEvent(
                AgentEventType.TOOL_RESULT_END, tool_call_id="c1",
                tool_call_name="echo_tool", result_text="产物",
            ),
        ))

        frames = sender.payloads("tool")
        assert frames[0] == {"name": "echo_tool", "displayName": "回声工具", "status": "start"}
        assert frames[1] == {
            "name": "echo_tool", "displayName": "回声工具",
            "status": "end", "result": "产物", "ok": True,
        }

    @pytest.mark.asyncio
    async def test_failed_tool_reports_ok_false(self):
        sender = _RecordingSender()
        await self._bridge(sender).run(await self._events(
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="t"),
            AgentEvent(
                AgentEventType.TOOL_RESULT_END, tool_call_id="c1", tool_call_name="t",
                state=ToolResultState.ERROR, result_text="炸了",
            ),
        ))

        assert sender.payloads("tool")[1]["ok"] is False

    @pytest.mark.asyncio
    async def test_tool_result_truncated_over_limit(self):
        sender = _RecordingSender()
        await self._bridge(sender).run(await self._events(
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="t"),
            AgentEvent(
                AgentEventType.TOOL_RESULT_END, tool_call_id="c1", tool_call_name="t",
                result_text="x" * (TOOL_RESULT_MAX_CHARS + 100),
            ),
        ))

        frame = sender.payloads("tool")[1]
        assert len(frame["result"]) < TOOL_RESULT_MAX_CHARS + 100

    @pytest.mark.asyncio
    async def test_missing_tool_call_id_uses_fallback_key(self):
        """供应商漏发 toolCallId 时进度仍要能配对，不能整条轨迹丢掉"""
        sender = _RecordingSender()
        bridge = self._bridge(sender)

        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_name="t"),
            AgentEvent(AgentEventType.TOOL_RESULT_END, tool_call_name="t", result_text="r"),
        ))

        assert FALLBACK_CALL_KEY in bridge._tool_slots
        assert sender.payloads("tool")[1]["result"] == "r"

    @pytest.mark.asyncio
    async def test_hint_frames(self):
        sender = _RecordingSender()
        await self._bridge(sender).run(await self._events(
            AgentEvent(AgentEventType.HINT_BLOCK, hint="提示文本"),
            AgentEvent(AgentEventType.EXCEED_MAX_ITERS),
        ))

        hints = sender.payloads("hint")
        assert hints[0] == {"code": "AGENT_HINT", "text": "提示文本"}
        assert hints[1]["code"] == "MAX_ITERATIONS"

    @pytest.mark.asyncio
    async def test_finish_frame_carries_message_id_and_status(self):
        sender = _RecordingSender()
        bridge = self._bridge(sender)

        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="答"),
            AgentEvent(AgentEventType.AGENT_RESULT, result=Msg.assistant(text="答")),
        ))

        assert sender.payloads("finish") == [{"messageId": "msg-1", "messageStatus": "NORMAL"}]
        assert sender.names()[-1] == AgentSSEEventType.DONE.value
        assert sender.completed

    @pytest.mark.asyncio
    async def test_final_answer_used_when_no_streamed_delta(self):
        """流式增量一个字没收到时回落框架终答，不能给用户一条空消息"""
        sender = _RecordingSender()
        persisted: dict[str, Any] = {}

        async def persist(content: str, thinking: str, blocks: Any, status: Any) -> str:
            persisted["content"] = content
            return "msg-2"

        await self._bridge(sender, persist).run(await self._events(
            AgentEvent(AgentEventType.AGENT_RESULT, result=Msg.assistant(text="终答")),
        ))

        assert persisted["content"] == "终答"

    @pytest.mark.asyncio
    async def test_persist_receives_blocks_and_status(self):
        captured: dict[str, Any] = {}

        async def persist(content: str, thinking: str, blocks: Any, status: Any) -> str:
            captured.update({"content": content, "thinking": thinking, "blocks": blocks, "status": status})
            return "msg-3"

        bridge = self._bridge(_RecordingSender(), persist)
        await bridge.run(await self._events(
            AgentEvent(AgentEventType.THINKING_BLOCK_DELTA, delta="想"),
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="答"),
        ))

        assert captured["content"] == "答"
        assert captured["thinking"] == "想"
        assert captured["status"] == AgentMessageStatus.NORMAL
        kinds = [b["kind"] for b in captured["blocks"]]
        assert kinds == ["reasoning", "answer"]

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_break_stream(self):
        """答案已经吐给用户了，落库失败不该把整条流判死"""
        async def boom(*args: Any) -> str:
            raise RuntimeError("db down")

        sender = _RecordingSender()
        await self._bridge(sender, boom).run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="答"),
        ))

        assert sender.failed is None
        assert sender.completed
        assert sender.payloads("finish")[0]["messageStatus"] == "NORMAL"

    @pytest.mark.asyncio
    async def test_blocks_merge_consecutive_same_kind(self):
        bridge = self._bridge()
        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="a"),
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="b"),
        ))

        blocks = bridge.settled_blocks()
        assert blocks == [{"kind": "answer", "at": blocks[0]["at"], "text": "ab"}]

    @pytest.mark.asyncio
    async def test_blocks_split_when_tool_interrupts_text(self):
        """工具插在两段回答之间时轨迹要分成两块，否则渲染顺序错乱"""
        bridge = self._bridge()
        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="先"),
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="t"),
            AgentEvent(AgentEventType.TOOL_RESULT_END, tool_call_id="c1", tool_call_name="t", result_text="r"),
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="后"),
        ))

        kinds = [b["kind"] for b in bridge.settled_blocks()]
        assert kinds == ["answer", "tool", "answer"]

    @pytest.mark.asyncio
    async def test_unfinished_tool_marked_interrupted(self):
        bridge = self._bridge()
        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="t"),
        ))

        blocks = bridge.settled_blocks()
        assert blocks[0]["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_empty_blocks_are_dropped(self):
        bridge = self._bridge()
        await bridge.run(await self._events(
            AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="   "),
        ))

        assert bridge.settled_blocks() is None

    @pytest.mark.asyncio
    async def test_cancel_persists_interrupted_when_content_exists(self):
        sender = _RecordingSender()
        captured: dict[str, Any] = {}

        async def persist(content: str, thinking: str, blocks: Any, status: Any) -> str:
            captured["status"] = status
            return "msg-9"

        bridge = self._bridge(sender, persist)

        async def gen() -> Any:
            yield AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="半截")
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await bridge.run(gen())

        assert captured["status"] == AgentMessageStatus.INTERRUPTED
        assert sender.payloads("cancel") == [{"messageId": "msg-9", "messageStatus": "INTERRUPTED"}]

    @pytest.mark.asyncio
    async def test_cancel_skips_persist_when_nothing_produced(self):
        """一个字都没吐就断了，落库只会留一条空消息"""
        sender = _RecordingSender()
        persisted = []

        async def persist(*args: Any) -> str:
            persisted.append(args)
            return "msg-x"

        bridge = self._bridge(sender, persist)

        async def gen() -> Any:
            raise asyncio.CancelledError
            yield  # pragma: no cover

        with pytest.raises(asyncio.CancelledError):
            await bridge.run(gen())

        assert persisted == []
        assert sender.payloads("cancel") == [{}]

    @pytest.mark.asyncio
    async def test_cancel_persists_when_only_tool_tracked(self):
        sender = _RecordingSender()
        persisted: list[Any] = []

        async def persist(*args: Any) -> str:
            persisted.append(args)
            return "msg-t"

        bridge = self._bridge(sender, persist)

        async def gen() -> Any:
            yield AgentEvent(AgentEventType.TOOL_CALL_START, tool_call_id="c1", tool_call_name="t")
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await bridge.run(gen())

        assert len(persisted) == 1

    @pytest.mark.asyncio
    async def test_error_persists_partial_content_then_fails(self):
        sender = _RecordingSender()
        captured: dict[str, Any] = {}

        async def persist(content: str, thinking: str, blocks: Any, status: Any) -> str:
            captured.update({"content": content, "status": status})
            return "msg-e"

        bridge = self._bridge(sender, persist)

        async def gen() -> Any:
            yield AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="半截")
            raise RuntimeError("模型炸了")

        await bridge.run(gen())

        assert captured["content"] == "半截"
        assert captured["status"] == AgentMessageStatus.INTERRUPTED
        assert sender.failed is not None

    @pytest.mark.asyncio
    async def test_settle_is_idempotent(self):
        sender = _RecordingSender()
        bridge = self._bridge(sender)

        await bridge.run(await self._events(AgentEvent(AgentEventType.TEXT_BLOCK_DELTA, delta="a")))
        await bridge._on_complete()
        await bridge._on_cancelled()

        assert sender.names().count("finish") == 1
        assert sender.payloads("cancel") == []


class TestAgentRunHandle:

    @pytest.mark.asyncio
    async def test_stop_invokes_cancel_and_on_stop(self):
        order: list[str] = []

        async def on_stop() -> None:
            order.append("on_stop")

        handle = AgentRunHandle(
            task_id="t1", conversation_id="c1", user_id="u1",
            cancel=lambda: order.append("cancel"),
            on_stop=on_stop,
        )
        await handle.stop()

        assert order == ["cancel", "on_stop"]

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        count = 0

        async def on_stop() -> None:
            nonlocal count
            count += 1

        handle = AgentRunHandle(task_id="t1", conversation_id="c1", user_id="u1", on_stop=on_stop)
        await handle.stop()
        await handle.stop()

        assert count == 1

    @pytest.mark.asyncio
    async def test_stop_without_on_stop_only_cancels(self):
        cancelled = []
        handle = AgentRunHandle(
            task_id="t1", conversation_id="c1", user_id="u1",
            cancel=lambda: cancelled.append(True),
        )
        await handle.stop()
        assert cancelled == [True]


# ---------------------------------------------------------------------------
# run_gate
# ---------------------------------------------------------------------------

class TestAgentRunGate:

    def _gate(self, redis: Any = None) -> tuple[AgentRunGate, Any]:
        fake = redis or _FakeRedis()
        return AgentRunGate(fake, sse_timeout_ms=1_000), fake

    def test_key_prefix_and_separator_match_java(self):
        assert RUNNING_KEY_PREFIX == "ragent:agent:running:"
        assert SLOT_SEPARATOR == "|"

    def test_ttl_is_twice_sse_timeout(self):
        """进程崩溃时没人来释放，TTL 是唯一出路"""
        gate, _ = self._gate()
        assert gate.ttl_ms == 2_000

    @pytest.mark.asyncio
    async def test_acquire_writes_slot_value(self):
        gate, redis = self._gate()
        await gate.acquire("u1", "task-1", "conv-1")

        assert redis.data["ragent:agent:running:u1"] == "task-1|conv-1"

    @pytest.mark.asyncio
    async def test_acquire_rejects_when_busy(self):
        gate, _ = self._gate()
        await gate.acquire("u1", "task-1", "conv-1")

        with pytest.raises(ClientException) as info:
            await gate.acquire("u1", "task-2", "conv-2")
        assert "处理中" in info.value.errorMessage

    @pytest.mark.asyncio
    async def test_acquire_allows_different_users(self):
        gate, _ = self._gate()
        await gate.acquire("u1", "t1", "c1")
        await gate.acquire("u2", "t2", "c2")

    @pytest.mark.asyncio
    async def test_release_removes_own_slot(self):
        gate, redis = self._gate()
        release = await gate.acquire("u1", "task-1", "conv-1")

        await release()
        assert "ragent:agent:running:u1" not in redis.data
        assert redis.eval_calls == 1

    @pytest.mark.asyncio
    async def test_release_does_not_touch_another_run_slot(self):
        """运行位被 TTL 挤掉又被下一轮抢走时，无条件删会放掉别人的闸门"""
        gate, redis = self._gate()
        release = await gate.acquire("u1", "task-1", "conv-1")

        redis.data["ragent:agent:running:u1"] = "task-2|conv-2"
        await release()

        assert redis.data["ragent:agent:running:u1"] == "task-2|conv-2"

    @pytest.mark.asyncio
    async def test_release_is_idempotent(self):
        gate, _ = self._gate()
        release = await gate.acquire("u1", "task-1", "conv-1")

        await release()
        await release()

    @pytest.mark.asyncio
    async def test_release_swallows_redis_errors(self):
        """释放失败只报警：TTL 兜底，不该把已经答完的流判成失败"""
        class _BoomRedis(_FakeRedis):
            async def eval(self, *args: Any) -> int:
                raise RuntimeError("redis down")

        gate = AgentRunGate(_BoomRedis(), sse_timeout_ms=1_000)
        release = await gate.acquire("u1", "t1", "c1")
        await release()

    @pytest.mark.asyncio
    async def test_running_task_id_matches_conversation(self):
        gate, _ = self._gate()
        await gate.acquire("u1", "task-1", "conv-1")

        assert await gate.running_task_id("u1", "conv-1") == "task-1"

    @pytest.mark.asyncio
    async def test_running_task_id_ignores_other_conversation(self):
        """删会话时只能停属于它的那条流，别人的流不能连坐"""
        gate, _ = self._gate()
        await gate.acquire("u1", "task-1", "conv-1")

        assert await gate.running_task_id("u1", "conv-other") is None

    @pytest.mark.asyncio
    async def test_running_task_id_none_when_idle(self):
        gate, _ = self._gate()
        assert await gate.running_task_id("u1", "conv-1") is None

    @pytest.mark.asyncio
    async def test_running_task_id_handles_bytes_value(self):
        gate, redis = self._gate()
        redis.data["ragent:agent:running:u1"] = "task-1|conv-1".encode("utf-8")

        assert await gate.running_task_id("u1", "conv-1") == "task-1"

    @pytest.mark.asyncio
    async def test_running_task_id_ignores_malformed_slot(self):
        gate, redis = self._gate()
        redis.data["ragent:agent:running:u1"] = "没有分隔符"

        assert await gate.running_task_id("u1", "conv-1") is None

    def test_ttl_defaults_to_settings(self):
        from app.config import settings

        gate = AgentRunGate(_FakeRedis())
        assert gate.ttl_ms == settings.agent.sse_timeout_ms * 2


# ---------------------------------------------------------------------------
# schemas — 前端契约
# ---------------------------------------------------------------------------

class TestAgentSchemas:

    def test_block_drops_none_fields(self):
        """Java 侧 @JsonInclude(NON_NULL)，多吐 null 会让前端判空逻辑失效"""
        wire = AgentBlock(kind="answer", at="2026-01-01T00:00:00", text="a").to_wire()
        assert wire == {"kind": "answer", "at": "2026-01-01T00:00:00", "text": "a"}

    def test_block_keeps_camel_case_field_names(self):
        wire = AgentBlock(kind="tool", name="t", displayName="展示名", toolCallId="c1").to_wire()
        assert "displayName" in wire
        assert "toolCallId" in wire
        assert "display_name" not in wire

    def test_meta_payload_field_names(self):
        payload = AgentMetaPayload(conversationId="c1", taskId="t1").model_dump()
        assert payload == {"conversationId": "c1", "taskId": "t1"}

    def test_tool_progress_drops_absent_result(self):
        wire = AgentToolProgress(name="t", displayName="d", status="start").to_wire()
        assert "result" not in wire
        assert "ok" not in wire

    def test_tool_progress_keeps_false_ok_flag(self):
        """ok=false 是有意义的信号，不能因为 falsy 就被剔掉"""
        wire = AgentToolProgress(name="t", displayName="d", status="end", result="r", ok=False).to_wire()
        assert wire["ok"] is False

    def test_completion_payload_shape(self):
        wire = AgentCompletionPayload(messageId="m1", messageStatus="NORMAL").to_wire()
        assert wire == {"messageId": "m1", "messageStatus": "NORMAL"}

    def test_message_vo_serializes_datetime(self):
        from datetime import datetime

        vo = AgentMessageVO(
            id="m1", role="assistant", content="a",
            createTime=datetime(2026, 1, 1, 12, 0, 0),
        )
        dumped = vo.model_dump(mode="json")
        assert dumped["createTime"].startswith("2026-01-01T12:00:00")

    def test_message_vo_accepts_blocks(self):
        vo = AgentMessageVO(id="m1", blocks=[{"kind": "answer", "text": "a"}])
        assert vo.blocks[0].kind == "answer"


# ---------------------------------------------------------------------------
# prompt_service
# ---------------------------------------------------------------------------

class TestPromptTemplateUtils:

    def test_cleanup_collapses_blank_lines(self):
        assert PromptTemplateUtils.cleanup_prompt("a\n\n\n\nb") == "a\n\nb"

    def test_cleanup_handles_none(self):
        assert PromptTemplateUtils.cleanup_prompt(None) == ""

    def test_cleanup_strips_edges(self):
        assert PromptTemplateUtils.cleanup_prompt("  a  ") == "a"

    def test_fill_slots_replaces_placeholders(self):
        assert PromptTemplateUtils.fill_slots("你好 {name}", {"name": "世界"}) == "你好 世界"

    def test_fill_slots_leaves_unknown_placeholders(self):
        """没配的槽位原样留着，静默删掉会让提示词读不通"""
        assert PromptTemplateUtils.fill_slots("a {x} b", {}) == "a {x} b"

    def test_fill_slots_handles_none(self):
        assert PromptTemplateUtils.fill_slots(None, None) == ""

    def test_parse_sections_splits_by_header(self):
        content = "--- section: 甲 ---\n内容甲\n--- section: 乙 ---\n内容乙"
        sections = PromptTemplateUtils.parse_sections(content)

        assert set(sections) == {"甲", "乙"}
        assert sections["甲"] == "内容甲"
        assert sections["乙"] == "内容乙"

    def test_parse_sections_returns_empty_for_none(self):
        assert PromptTemplateUtils.parse_sections(None) == {}


class TestAgentPromptSlot:

    def test_slot_keys_match_java_names(self):
        assert AgentPromptSlot.AGENT_MAIN.value == "AGENT_MAIN"
        assert AgentPromptSlot.KNOWLEDGE_TOOL_DESCRIPTION.value == "KNOWLEDGE_TOOL_DESCRIPTION"
        assert AgentPromptSlot.AGENT_CONTEXT_COMPACTION.value == "AGENT_CONTEXT_COMPACTION"

    def test_all_slots_have_display_name_and_group(self):
        for slot in AgentPromptSlot:
            assert slot.display_name
            assert isinstance(slot.group, PromptSlotGroup)

    def test_find_by_key(self):
        assert AgentPromptSlot.find("KB_ANSWER") == AgentPromptSlot.KB_ANSWER
        assert AgentPromptSlot.find("不存在") is None
        assert AgentPromptSlot.find(None) is None

    def test_effective_modes_are_non_empty(self):
        for slot in AgentPromptSlot:
            assert slot.effective_modes
            assert all(isinstance(m, OrchestrationMode) for m in slot.effective_modes)

    def test_effective_in_filters_by_mode(self):
        agent_slots = AgentPromptSlot.effective_in(OrchestrationMode.AGENT)
        assert AgentPromptSlot.AGENT_MAIN in agent_slots
        assert all(OrchestrationMode.AGENT in s.effective_modes for s in agent_slots)

    def test_workflow_only_slot_is_inactive_in_agent_mode(self):
        workflow_only = [
            s for s in AgentPromptSlot
            if s.effective_modes == frozenset({OrchestrationMode.WORKFLOW})
        ]
        for slot in workflow_only:
            assert not slot.is_effective_in(OrchestrationMode.AGENT)
            assert slot.inactive_reason
