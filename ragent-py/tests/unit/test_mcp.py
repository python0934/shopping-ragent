"""
Phase 6 单元测试 — MCP 工具注册表、远程执行器与意图节点注册表.

MCP Server 不真连：``McpClientManager._open_and_list`` 被打桩，
只验证连接编排、超时降级与注册结果。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agent.tools import AgentToolCatalog, McpToolBridge, ToolCallParam
from app.config import McpServerEntry
from app.core.database import Base
from app.core.snowflake import get_snowflake_id_str
from app.mcp.registry import (
    REMOTE_FAILURE_PREFIX,
    McpClientManager,
    McpClientToolExecutor,
    McpToolRegistry,
    _content_to_dicts,
    tool_to_definition,
)
from app.models.rag import IntentNodeDO
from app.services.intent_registry import DbIntentNodeRegistry, IntentNode


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class _FakeMcpTool:
    """形状对齐 mcp.types.Tool：annotations 是 pydantic 模型"""

    def __init__(self, name: str, description: str = "", schema: dict | None = None,
                 read_only: bool | None = None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}
        self.annotations = _FakeAnnotations(read_only) if read_only is not None else None


class _FakeAnnotations:
    def __init__(self, read_only: bool | None) -> None:
        self.readOnlyHint = read_only

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"readOnlyHint": self.readOnlyHint}


class _FakeTextContent:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


class _FakeCallResult:
    def __init__(self, content: list[Any], is_error: bool = False) -> None:
        self.content = content
        self.isError = is_error


class _FakeSession:
    """记录 call_tool 调用，按脚本回值或抛异常"""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        return self.result


def _executor(name: str = "weather", session: Any = None, **kwargs: Any) -> McpClientToolExecutor:
    return McpClientToolExecutor(
        session if session is not None else _FakeSession(),
        tool_to_definition(_FakeMcpTool(name, **kwargs)),
        "default",
    )


# ---------------------------------------------------------------------------
# McpToolRegistry
# ---------------------------------------------------------------------------

class TestMcpToolRegistry:

    def test_register_and_lookup(self):
        registry = McpToolRegistry()
        executor = _executor("weather")

        registry.register(executor)

        assert registry.contains("weather")
        assert registry.size() == 1
        assert registry.get_executor("weather") is executor

    def test_register_none_ignored(self):
        registry = McpToolRegistry()

        registry.register(None)

        assert registry.size() == 0

    def test_register_blank_tool_id_ignored(self):
        registry = McpToolRegistry()

        registry.register(McpClientToolExecutor(_FakeSession(), {"name": "  "}))

        assert registry.size() == 0

    def test_register_same_id_overwrites(self):
        registry = McpToolRegistry()
        first, second = _executor("weather"), _executor("weather")

        registry.register(first)
        registry.register(second)

        assert registry.size() == 1
        assert registry.get_executor("weather") is second

    def test_unregister_removes_and_is_idempotent(self):
        registry = McpToolRegistry()
        registry.register(_executor("weather"))

        registry.unregister("weather")
        registry.unregister("weather")

        assert registry.size() == 0
        assert registry.get_executor("weather") is None

    def test_list_all_keeps_registration_order(self):
        registry = McpToolRegistry()
        registry.register(_executor("b"))
        registry.register(_executor("a"))

        assert [e.tool_id for e in registry.list_all_executors()] == ["b", "a"]
        assert [t["name"] for t in registry.list_all_tools()] == ["b", "a"]

    def test_clear_empties_registry(self):
        registry = McpToolRegistry()
        registry.register(_executor("weather"))

        registry.clear()

        assert registry.size() == 0


# ---------------------------------------------------------------------------
# tool_to_definition / _content_to_dicts
# ---------------------------------------------------------------------------

class TestDefinitionMapping:

    def test_definition_uses_camel_case_keys(self):
        definition = tool_to_definition(_FakeMcpTool(
            "weather", "查天气", {"type": "object", "properties": {"city": {"type": "string"}}},
        ))

        assert definition["name"] == "weather"
        assert definition["description"] == "查天气"
        assert definition["inputSchema"]["properties"]["city"]["type"] == "string"
        assert definition["annotations"] == {}

    def test_definition_passes_through_read_only_hint(self):
        definition = tool_to_definition(_FakeMcpTool("weather", read_only=True))

        assert definition["annotations"]["readOnlyHint"] is True

    def test_definition_tolerates_missing_schema(self):
        tool = _FakeMcpTool("weather")
        tool.inputSchema = None

        assert tool_to_definition(tool)["inputSchema"] == {}

    def test_content_to_dicts_dumps_pydantic_blocks(self):
        assert _content_to_dicts([_FakeTextContent("hi")]) == [{"type": "text", "text": "hi"}]

    def test_content_to_dicts_falls_back_to_str(self):
        assert _content_to_dicts(["raw"]) == [{"type": "text", "text": "raw"}]

    def test_content_to_dicts_handles_empty(self):
        assert _content_to_dicts(None) == []
        assert _content_to_dicts([]) == []


# ---------------------------------------------------------------------------
# McpClientToolExecutor
# ---------------------------------------------------------------------------

class TestMcpClientToolExecutor:

    def test_tool_id_comes_from_definition(self):
        assert _executor("weather").tool_id == "weather"

    async def test_execute_returns_content_and_error_flag(self):
        session = _FakeSession(_FakeCallResult([_FakeTextContent("晴 25℃")]))

        result = await _executor("weather", session).execute({"city": "北京"})

        assert result == {"content": [{"type": "text", "text": "晴 25℃"}], "isError": False}
        assert session.calls == [("weather", {"city": "北京"})]

    async def test_execute_propagates_server_error_flag(self):
        session = _FakeSession(_FakeCallResult([_FakeTextContent("参数非法")], is_error=True))

        result = await _executor("weather", session).execute({})

        assert result["isError"] is True

    async def test_execute_none_arguments_becomes_empty_dict(self):
        session = _FakeSession(_FakeCallResult([]))

        await _executor("weather", session).execute(None)

        assert session.calls == [("weather", {})]

    async def test_execute_degrades_exception_to_error_result(self):
        """异常不往上抛：ReAct 循环要把失败当工具结果回灌给模型自纠"""
        session = _FakeSession(error=RuntimeError("连接中断"))

        result = await _executor("weather", session).execute({})

        assert result["isError"] is True
        assert result["content"][0]["text"] == f"{REMOTE_FAILURE_PREFIX}连接中断"

    async def test_execute_uses_class_name_when_message_blank(self):
        session = _FakeSession(error=RuntimeError())

        result = await _executor("weather", session).execute({})

        assert result["content"][0]["text"] == f"{REMOTE_FAILURE_PREFIX}RuntimeError"

    async def test_execute_does_not_mutate_caller_arguments(self):
        session = _FakeSession(_FakeCallResult([]))
        args = {"city": "北京"}

        await _executor("weather", session).execute(args)

        assert args == {"city": "北京"}
        assert session.calls[0][1] is not args


# ---------------------------------------------------------------------------
# McpToolBridge 与执行器的衔接
# ---------------------------------------------------------------------------

class TestBridgeIntegration:

    def test_bridge_reads_mcp_definition(self):
        executor = _executor("weather", description="服务端描述", read_only=True)
        executor._definition["inputSchema"] = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        bridge = McpToolBridge(executor)

        assert bridge.name == "weather"
        assert bridge.description == "服务端描述"
        assert bridge.is_read_only() is True
        assert bridge.parameters()["required"] == ["city"]

    def test_bridge_description_override_wins(self):
        bridge = McpToolBridge(_executor("weather", description="服务端描述"), "意图树描述")

        assert bridge.description == "意图树描述"

    def test_bridge_blank_override_falls_back(self):
        bridge = McpToolBridge(_executor("weather", description="服务端描述"), "   ")

        assert bridge.description == "服务端描述"

    async def test_bridge_call_returns_tool_result_block(self):
        session = _FakeSession(_FakeCallResult([_FakeTextContent("晴")]))
        bridge = McpToolBridge(_executor("weather", session))

        block = await bridge.call(ToolCallParam(tool_call_id="c1", input={"city": "北京"}))

        assert block.id == "c1"
        assert block.output_text() == "晴"
        assert block.state.value == "SUCCESS"

    async def test_bridge_call_marks_error_state(self):
        session = _FakeSession(error=RuntimeError("炸了"))
        bridge = McpToolBridge(_executor("weather", session))

        block = await bridge.call(ToolCallParam(tool_call_id="c1", input={}))

        assert block.state.value == "ERROR"
        assert "炸了" in block.output_text()

    async def test_bridge_empty_content_gets_placeholder(self):
        session = _FakeSession(_FakeCallResult([]))
        bridge = McpToolBridge(_executor("weather", session))

        block = await bridge.call(ToolCallParam(tool_call_id="c1", input={}))

        assert block.output_text() == "（工具无返回内容）"


# ---------------------------------------------------------------------------
# McpClientManager
# ---------------------------------------------------------------------------

class TestMcpClientManager:

    async def test_startup_without_servers_is_noop(self):
        manager = McpClientManager(McpToolRegistry())

        await manager.startup([])

        assert manager.registry.size() == 0
        assert manager.connected_servers == []

    async def test_startup_skips_entries_without_url(self):
        manager = McpClientManager(McpToolRegistry())

        await manager.startup([McpServerEntry(name="blank", url="  ")])

        assert manager.connected_servers == []

    async def test_startup_registers_discovered_tools(self):
        manager = McpClientManager(McpToolRegistry())
        session = _FakeSession()

        async def fake_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            return session, [_FakeMcpTool("weather"), _FakeMcpTool("leave")]

        manager._open_and_list = fake_open  # type: ignore[method-assign]

        await manager.startup([McpServerEntry(name="default", url="http://mcp.local")])

        assert manager.registry.size() == 2
        assert manager.registry.contains("weather")
        assert manager.connected_servers == ["default"]
        await manager.shutdown()

    async def test_startup_survives_server_without_tools(self):
        manager = McpClientManager(McpToolRegistry())

        async def fake_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            return _FakeSession(), []

        manager._open_and_list = fake_open  # type: ignore[method-assign]

        await manager.startup([McpServerEntry(name="empty", url="http://mcp.local")])

        assert manager.registry.size() == 0
        assert manager.connected_servers == ["empty"]
        await manager.shutdown()

    async def test_startup_skips_unreachable_server_without_raising(self):
        manager = McpClientManager(McpToolRegistry())

        async def fake_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            raise ConnectionError("拒绝连接")

        manager._open_and_list = fake_open  # type: ignore[method-assign]

        await manager.startup([
            McpServerEntry(name="dead", url="http://dead.local"),
            McpServerEntry(name="alive", url="http://alive.local"),
        ])

        assert manager.connected_servers == []
        assert manager.registry.size() == 0

    async def test_startup_abandons_slow_server_at_timeout(self):
        """连不上的 Server 不能拖住应用启动 —— 超时后当场拆栈并跳过"""
        manager = McpClientManager(McpToolRegistry(), connect_timeout_seconds=0.05)

        async def slow_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            await asyncio.sleep(5)
            return _FakeSession(), [_FakeMcpTool("weather")]

        manager._open_and_list = slow_open  # type: ignore[method-assign]

        await manager.startup([McpServerEntry(name="slow", url="http://slow.local")])

        assert manager.registry.size() == 0
        assert manager.connected_servers == []

    async def test_startup_is_repeatable(self):
        manager = McpClientManager(McpToolRegistry())

        async def fake_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            return _FakeSession(), [_FakeMcpTool("weather")]

        manager._open_and_list = fake_open  # type: ignore[method-assign]

        await manager.startup([McpServerEntry(name="s", url="http://mcp.local")])
        await manager.startup([McpServerEntry(name="s", url="http://mcp.local")])

        assert manager.registry.size() == 1
        assert manager.connected_servers == ["s"]
        await manager.shutdown()

    async def test_shutdown_clears_registry_and_servers(self):
        manager = McpClientManager(McpToolRegistry())

        async def fake_open(stack: Any, server: McpServerEntry) -> tuple[Any, list[Any]]:
            return _FakeSession(), [_FakeMcpTool("weather")]

        manager._open_and_list = fake_open  # type: ignore[method-assign]
        await manager.startup([McpServerEntry(name="s", url="http://mcp.local")])

        await manager.shutdown()

        assert manager.registry.size() == 0
        assert manager.connected_servers == []

    async def test_shutdown_tolerates_close_failure(self):
        manager = McpClientManager(McpToolRegistry())
        manager._stacks.append(_BrokenStack())

        await manager.shutdown()  # 不抛

        assert manager._stacks == []


class _BrokenStack:
    async def aclose(self) -> None:
        raise RuntimeError("卡死")


# ---------------------------------------------------------------------------
# DbIntentNodeRegistry
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def session():
    """SQLite 内存库会话，只建意图节点表"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[IntentNodeDO.__table__]))

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db:
        yield db

    await engine.dispose()


async def _seed_node(
    db: AsyncSession,
    intent_code: str,
    *,
    name: str = "节点",
    parent_code: str | None = None,
    kind: int = 0,
    mcp_tool_id: str | None = None,
    description: str | None = None,
    enabled: int = 1,
    deleted: int = 0,
    level: int = 2,
) -> None:
    db.add(IntentNodeDO(
        id=get_snowflake_id_str(),
        intent_code=intent_code,
        name=name,
        level=level,
        parent_code=parent_code,
        description=description,
        kind=kind,
        mcp_tool_id=mcp_tool_id,
        enabled=enabled,
        deleted=deleted,
        collection_names=[],
        # SQLite 会把 server_default="CURRENT_TIMESTAMP" 存成字面串，只能显式给值
        create_time=datetime.now(),
        update_time=datetime.now(),
    ))
    await db.commit()


class TestDbIntentNodeRegistry:

    async def test_empty_db_yields_no_nodes(self, session: AsyncSession):
        registry = DbIntentNodeRegistry()

        assert await registry.refresh(session) == 0
        assert registry.list_mcp_tool_nodes() == []

    async def test_picks_enabled_mcp_leaves(self, session: AsyncSession):
        await _seed_node(session, "weather", name="天气查询", kind=2, mcp_tool_id="get_weather",
                         description="查天气")
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)
        nodes = registry.list_mcp_tool_nodes()

        assert len(nodes) == 1
        assert nodes[0]["mcpToolId"] == "get_weather"
        assert nodes[0]["name"] == "天气查询"
        assert nodes[0]["description"] == "查天气"

    async def test_excludes_non_mcp_kinds(self, session: AsyncSession):
        await _seed_node(session, "rag-node", kind=0, mcp_tool_id="get_weather")
        await _seed_node(session, "sys-node", kind=1, mcp_tool_id="get_weather")
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert registry.list_mcp_tool_nodes() == []

    async def test_excludes_disabled_and_deleted(self, session: AsyncSession):
        await _seed_node(session, "off", kind=2, mcp_tool_id="t1", enabled=0)
        await _seed_node(session, "gone", kind=2, mcp_tool_id="t2", deleted=1)
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert registry.list_mcp_tool_nodes() == []

    async def test_excludes_blank_tool_id(self, session: AsyncSession):
        await _seed_node(session, "blank", kind=2, mcp_tool_id="   ")
        await _seed_node(session, "null", kind=2, mcp_tool_id=None)
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert registry.list_mcp_tool_nodes() == []

    async def test_parent_node_is_not_a_leaf(self, session: AsyncSession):
        await _seed_node(session, "root", kind=2, mcp_tool_id="t_root", level=0)
        await _seed_node(session, "child", kind=2, mcp_tool_id="t_child", parent_code="root")
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert [n["mcpToolId"] for n in registry.list_mcp_tool_nodes()] == ["t_child"]

    async def test_orphaned_child_counts_as_leaf(self, session: AsyncSession):
        """父节点被停用后不在快照里，子节点重新成为叶子"""
        await _seed_node(session, "root", kind=2, mcp_tool_id="t_root", enabled=0)
        await _seed_node(session, "child", kind=2, mcp_tool_id="t_child", parent_code="root")
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert [n["mcpToolId"] for n in registry.list_mcp_tool_nodes()] == ["t_child"]

    async def test_nodes_sorted_by_id(self, session: AsyncSession):
        for code in ("c", "a", "b"):
            await _seed_node(session, code, kind=2, mcp_tool_id=f"t_{code}")
        registry = DbIntentNodeRegistry()

        await registry.refresh(session)

        assert [n["id"] for n in registry.list_mcp_tool_nodes()] == ["a", "b", "c"]

    async def test_get_node_by_id(self, session: AsyncSession):
        await _seed_node(session, "weather", kind=2, mcp_tool_id="t1")
        registry = DbIntentNodeRegistry()
        await registry.refresh(session)

        assert registry.get_node_by_id("weather").is_mcp is True
        assert registry.get_node_by_id("  weather  ").is_mcp is True
        assert registry.get_node_by_id("missing") is None
        assert registry.get_node_by_id(None) is None
        assert registry.get_node_by_id("  ") is None

    async def test_refresh_replaces_previous_snapshot(self, session: AsyncSession):
        await _seed_node(session, "weather", kind=2, mcp_tool_id="t1")
        registry = DbIntentNodeRegistry()
        await registry.refresh(session)

        await session.execute(IntentNodeDO.__table__.update().values(enabled=0))
        await session.commit()
        await registry.refresh(session)

        assert registry.list_mcp_tool_nodes() == []

    async def test_clear_empties_snapshot(self, session: AsyncSession):
        await _seed_node(session, "weather", kind=2, mcp_tool_id="t1")
        registry = DbIntentNodeRegistry()
        await registry.refresh(session)

        registry.clear()

        assert registry.list_all_nodes() == []

    def test_intent_node_predicates(self):
        leaf = IntentNode(id="a", kind=2, mcp_tool_id="t")
        branch = IntentNode(id="b", kind=2, mcp_tool_id="t", children=("c",))

        assert leaf.is_leaf and leaf.is_mcp
        assert not branch.is_leaf
        assert not IntentNode(id="c", kind=0).is_mcp


# ---------------------------------------------------------------------------
# AgentToolCatalog × 真实注册表
# ---------------------------------------------------------------------------

class _StubPromptResolver:
    async def resolve_knowledge_tool_description(self) -> str:
        return "检索知识库"


class _StubSearchFacade:
    async def search(self, query: str, history: list[dict[str, str]]) -> str:
        return "证据"


class TestCatalogWithRegistries:

    def _registries(self, intent_nodes: list[dict[str, Any]], executors: list[McpClientToolExecutor]):
        class _Intent:
            def list_mcp_tool_nodes(self) -> list[dict[str, Any]]:
                return intent_nodes

        registry = McpToolRegistry()
        for executor in executors:
            registry.register(executor)
        return _Intent(), registry

    async def test_intersection_binds_configured_and_available(self):
        intent, mcp = self._registries(
            [{"mcpToolId": "weather", "name": "天气查询", "description": "查天气"}],
            [_executor("weather")],
        )
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        resolved = await catalog.resolve()

        assert len(resolved.bindings) == 1
        assert resolved.bindings[0].tool_id == "weather"
        assert resolved.display_name_of("weather") == "天气查询"
        assert resolved.unavailable_tool_ids == ()

    async def test_configured_but_unavailable_is_collected(self):
        intent, mcp = self._registries(
            [{"mcpToolId": "ghost", "name": "幽灵工具", "description": ""}], [],
        )
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        resolved = await catalog.resolve()

        assert resolved.bindings == ()
        assert resolved.unavailable_tool_ids == ("ghost",)

    async def test_available_but_unconfigured_is_not_bound(self):
        intent, mcp = self._registries([], [_executor("weather")])
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        resolved = await catalog.resolve()

        assert resolved.bindings == ()
        assert resolved.unavailable_tool_ids == ()

    async def test_build_tools_includes_knowledge_and_mcp(self):
        intent, mcp = self._registries(
            [{"mcpToolId": "weather", "name": "天气查询", "description": "查天气"}],
            [_executor("weather")],
        )
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        tools = catalog.build_tools(await catalog.resolve())

        assert [t.name for t in tools] == ["search_knowledge", "weather"]

    async def test_duplicate_nodes_merge_descriptions(self):
        intent, mcp = self._registries(
            [
                {"mcpToolId": "weather", "name": "", "description": "查天气"},
                {"mcpToolId": "weather", "name": "天气查询", "description": "查天气"},
                {"mcpToolId": "weather", "name": "别名", "description": "支持城市名"},
            ],
            [_executor("weather")],
        )
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        resolved = await catalog.resolve()
        binding = resolved.bindings[0]

        assert binding.display_name == "天气查询"
        assert binding.description == "查天气\n支持城市名"

    def test_mcp_tool_count_skips_prompt_resolution(self):
        intent, mcp = self._registries(
            [{"mcpToolId": "weather", "name": "天气", "description": ""}], [_executor("weather")],
        )
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade(),
                                   intent_node_registry=intent, mcp_tool_registry=mcp)

        assert catalog.mcp_tool_count() == 1

    async def test_no_registries_yields_no_mcp_tools(self):
        catalog = AgentToolCatalog(_StubPromptResolver(), _StubSearchFacade())

        resolved = await catalog.resolve()

        assert resolved.bindings == ()
        assert catalog.mcp_tool_count() == 0
        assert resolved.display_name_of("search_knowledge") == "知识库检索"


# ---------------------------------------------------------------------------
# ReActAgentProvider 的快照刷新
# ---------------------------------------------------------------------------

class TestProviderIntentRefresh:

    @staticmethod
    def _provider(registry: Any, tool_ids: tuple[str, ...] = ("t1",)) -> Any:
        from app.agent.provider import ReActAgentProvider

        mcp = McpToolRegistry()
        for tool_id in tool_ids:
            mcp.register(_executor(tool_id))
        return ReActAgentProvider(
            model_client=_StubModelClient(),
            intent_node_registry=registry,
            mcp_tool_registry=mcp,
        )

    async def test_refresh_delegates_to_registry(self, session: AsyncSession):
        await _seed_node(session, "weather", kind=2, mcp_tool_id="t1")
        provider = self._provider(DbIntentNodeRegistry())

        await provider.refresh_intent_snapshot(session)

        assert provider.mcp_tool_count() == 1

    async def test_refresh_swallows_db_failure(self):
        """快照不是硬依赖：刷失败只沿用上一份，不能把整轮对话带倒"""
        registry = DbIntentNodeRegistry()
        provider = self._provider(registry)

        await provider.refresh_intent_snapshot(_BoomSession())

        assert provider.mcp_tool_count() == 0

    async def test_refresh_skips_registry_without_refresh(self):
        class _SyncOnly:
            def list_mcp_tool_nodes(self) -> list[dict[str, Any]]:
                return []

        await self._provider(_SyncOnly()).refresh_intent_snapshot(_BoomSession())  # 不抛

    def test_provider_defaults_to_process_singletons(self):
        from app.agent.provider import ReActAgentProvider
        from app.mcp import mcp_tool_registry
        from app.services.intent_registry import intent_node_registry

        provider = ReActAgentProvider(model_client=_StubModelClient())

        assert provider.intent_node_registry is intent_node_registry
        assert provider.mcp_tool_registry is mcp_tool_registry

    def test_provider_honours_explicit_none(self):
        """显式传 None 是「关掉 MCP」，不能被单例默认值抢回去"""
        from app.agent.provider import ReActAgentProvider

        provider = ReActAgentProvider(
            model_client=_StubModelClient(),
            intent_node_registry=None,
            mcp_tool_registry=None,
        )

        assert provider.intent_node_registry is None
        assert provider.mcp_tool_count() == 0


class _BoomSession:
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("DB 不可达")


class _StubModelClient:
    """provider 构造需要一个模型客户端，这些用例不跑循环"""

    async def stream_completion(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("不应被调用")
