"""
MCP 工具注册表与远程执行器 — mirrors rag.core.mcp.DefaultMcpToolRegistry /
McpClientToolExecutor / McpClientAutoConfiguration.

注册表是进程级单例：启动时把所有配置的 MCP Server 连上、把发现的工具逐个注册，
Agent 每次装配工具目录时只读快照，不在请求路径上碰网络。
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from typing import Any

from app.config import McpServerEntry, settings

logger = logging.getLogger(__name__)

CLIENT_NAME = "ragent-bootstrap"
CLIENT_VERSION = "1.0.0"
MCP_PATH_SUFFIX = "/mcp"
REMOTE_FAILURE_PREFIX = "远程调用失败: "
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
CLOSE_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# 执行器 — mirrors McpClientToolExecutor
# ---------------------------------------------------------------------------

class McpClientToolExecutor:
    """
    通过 MCP 会话调用远端 Server 暴露的工具。

    工具定义统一摊平成 dict（``name`` / ``description`` / ``inputSchema`` /
    ``annotations``），agent.tool.McpToolBridge 按这几个键读取。
    """

    def __init__(self, session: Any, tool_definition: dict[str, Any], server_name: str = "") -> None:
        self._session = session
        self._definition = tool_definition
        self._server_name = server_name

    @property
    def tool_id(self) -> str:
        return str(self._definition.get("name") or "")

    @property
    def server_name(self) -> str:
        return self._server_name

    def tool_definition(self) -> dict[str, Any]:
        return self._definition

    async def execute(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        调用远端工具；异常一律折成 isError 结果。

        不把异常抛给调用方，是因为 ReAct 循环要把失败当工具结果回灌给模型自纠，
        抛出去会直接终止整轮流。
        """
        args = dict(arguments or {})
        start_ms = time.monotonic()
        try:
            result = await self._session.call_tool(self.tool_id, args)
            content = _content_to_dicts(getattr(result, "content", None))
            logger.info(
                "MCP 远程工具调用完成, toolId=%s, server=%s, contentSize=%d, elapsed=%.0fms",
                self.tool_id, self._server_name, len(content), (time.monotonic() - start_ms) * 1000,
            )
            return {"content": content, "isError": bool(getattr(result, "isError", False))}
        except Exception as e:  # noqa: BLE001 - 远端不可控，统一降级
            reason = str(e) or e.__class__.__name__
            logger.warning(
                "MCP 远程工具调用异常, toolId=%s, server=%s, elapsed=%.0fms, reason=%s",
                self.tool_id, self._server_name, (time.monotonic() - start_ms) * 1000, reason,
            )
            return {
                "content": [{"type": "text", "text": f"{REMOTE_FAILURE_PREFIX}{reason}"}],
                "isError": True,
            }


def _content_to_dicts(content: Any) -> list[dict[str, Any]]:
    """MCP 内容块 -> dict；非 pydantic 的原样透传"""
    if not content:
        return []
    out: list[dict[str, Any]] = []
    for item in content:
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            try:
                out.append(dump(by_alias=True, exclude_none=True, mode="json"))
                continue
            except Exception:  # noqa: BLE001 - 序列化失败退化成文本
                pass
        out.append({"type": "text", "text": str(item)})
    return out


def tool_to_definition(tool: Any) -> dict[str, Any]:
    """MCP ``Tool`` -> 注册表内部统一形状"""
    annotations = getattr(tool, "annotations", None)
    dumped: dict[str, Any] = {}
    if annotations is not None:
        try:
            dumped = annotations.model_dump(by_alias=True, exclude_none=True, mode="json")
        except Exception:  # noqa: BLE001
            dumped = {}

    schema = getattr(tool, "inputSchema", None) or {}
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump(by_alias=True, exclude_none=True, mode="json")

    return {
        "name": getattr(tool, "name", "") or "",
        "description": getattr(tool, "description", "") or "",
        "inputSchema": schema,
        "annotations": dumped,
    }


# ---------------------------------------------------------------------------
# 注册表 — mirrors DefaultMcpToolRegistry
# ---------------------------------------------------------------------------

class McpToolRegistry:
    """toolId -> executor 的进程级注册表"""

    def __init__(self) -> None:
        self._executors: dict[str, McpClientToolExecutor] = {}

    def register(self, executor: McpClientToolExecutor | None) -> None:
        if executor is None or not executor.tool_definition():
            logger.warning("尝试注册空的执行器，已忽略")
            return

        tool_id = executor.tool_id
        if not tool_id or not tool_id.strip():
            logger.warning("工具 ID 为空，已忽略")
            return

        if tool_id in self._executors:
            logger.warning("工具 %s 已存在，已覆盖", tool_id)
        else:
            logger.info("MCP 工具注册成功, toolId: %s", tool_id)
        self._executors[tool_id] = executor

    def unregister(self, tool_id: str) -> None:
        if self._executors.pop(tool_id, None) is not None:
            logger.info("MCP 工具注销成功, toolId: %s", tool_id)

    def get_executor(self, tool_id: str) -> McpClientToolExecutor | None:
        return self._executors.get(tool_id)

    def list_all_tools(self) -> list[dict[str, Any]]:
        return [e.tool_definition() for e in self._executors.values()]

    def list_all_executors(self) -> list[McpClientToolExecutor]:
        return list(self._executors.values())

    def contains(self, tool_id: str) -> bool:
        return tool_id in self._executors

    def size(self) -> int:
        return len(self._executors)

    def clear(self) -> None:
        self._executors.clear()


# ---------------------------------------------------------------------------
# 连接管理 — mirrors McpClientAutoConfiguration
# ---------------------------------------------------------------------------

class McpClientManager:
    """
    按配置连接所有 MCP Server 并把发现的工具注册进注册表。

    每个 Server 一个独立的 ``AsyncExitStack``：连接失败或超时就当场拆掉，
    不会把半截传输层泄到进程里；连上的才留下来，随应用关停统一释放。
    单个 Server 不可用只记 error 跳过，不影响其余 Server 与应用启动。
    """

    def __init__(
        self,
        registry: McpToolRegistry | None = None,
        connect_timeout_seconds: float | None = None,
    ) -> None:
        self.registry = registry if registry is not None else mcp_tool_registry
        # 显式传入的超时优先于配置，否则读不到 settings 时退回模块常量
        self._explicit_timeout = connect_timeout_seconds
        self._stacks: list[AsyncExitStack] = []
        self._connected_servers: list[str] = []

    @property
    def connect_timeout_seconds(self) -> float:
        if self._explicit_timeout is not None:
            return float(self._explicit_timeout)
        configured = getattr(settings.rag.mcp, "connect_timeout_seconds", None)
        return float(configured) if configured else DEFAULT_CONNECT_TIMEOUT_SECONDS

    @property
    def connected_servers(self) -> list[str]:
        return list(self._connected_servers)

    async def startup(self, servers: list[McpServerEntry] | None = None) -> None:
        """连接配置里的全部 Server；可重复调用，每次先清场"""
        await self.shutdown()

        entries = servers if servers is not None else list(settings.rag.mcp.servers or [])
        entries = [s for s in entries if (s.url or "").strip()]
        if not entries:
            logger.info("未配置 MCP Server，跳过远程工具注册")
            return

        timeout = self.connect_timeout_seconds
        for server in entries:
            try:
                await self._register_remote_tools(server, timeout)
            except Exception as e:  # noqa: BLE001 - 单 Server 失败不连坐
                logger.error("连接 MCP Server [%s] 失败，跳过工具注册，reason=%s", server.name, e)

    async def _register_remote_tools(self, server: McpServerEntry, timeout: float) -> None:
        stack = AsyncExitStack()
        try:
            session, tools = await asyncio.wait_for(
                self._open_and_list(stack, server), timeout=timeout,
            )
        except Exception:
            await self._close_stack(stack)
            raise

        if not tools:
            logger.info("MCP Server [%s] 未发现可用工具，跳过工具注册", server.name)
        else:
            logger.info("MCP Server [%s] 返回 %d 个工具", server.name, len(tools))
            for tool in tools:
                self.registry.register(
                    McpClientToolExecutor(session, tool_to_definition(tool), server.name)
                )

        self._stacks.append(stack)
        self._connected_servers.append(server.name)

    @staticmethod
    async def _open_and_list(stack: AsyncExitStack, server: McpServerEntry) -> tuple[Any, list[Any]]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        url = server.url if server.url.endswith(MCP_PATH_SUFFIX) else server.url + MCP_PATH_SUFFIX
        logger.info("连接 MCP Server: name=%s, url=%s", server.name, url)

        read_stream, write_stream, _ = await stack.enter_async_context(
            streamablehttp_client(url)
        )
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        listed = await session.list_tools()
        return session, list(getattr(listed, "tools", None) or [])

    async def shutdown(self) -> None:
        """释放全部会话并清空注册表"""
        stacks, self._stacks = self._stacks, []
        self._connected_servers.clear()
        self.registry.clear()
        for stack in stacks:
            await self._close_stack(stack)

    @staticmethod
    async def _close_stack(stack: AsyncExitStack) -> None:
        """关停也要限时：卡死的传输层不能拖住整个应用退出"""
        try:
            await asyncio.wait_for(stack.aclose(), timeout=CLOSE_TIMEOUT_SECONDS)
        except Exception as e:  # noqa: BLE001 - 关停期异常只告警
            logger.warning("关闭 MCP 客户端失败: %s", e)


mcp_tool_registry = McpToolRegistry()
mcp_client_manager = McpClientManager(mcp_tool_registry)
