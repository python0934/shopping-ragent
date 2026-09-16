"""
Agent 工具体系 — mirrors agent.tool.*.

  - tool.AgentToolCatalog    -> AgentToolCatalog / ResolvedCatalog
  - tool.KnowledgeSearchTool -> KnowledgeSearchTool
  - tool.McpToolBridge       -> McpToolBridge

Java 侧实现 AgentScope 的 ``AgentTool`` 接口；这里自定义同形状协议，
ReAct 循环（react_agent）按 OpenAI 兼容的 function-calling schema 下发。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.agent.enums import ToolResultState
from app.agent.messages import TextBlock, ToolResultBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 调用参数与结果 — mirrors AgentScope ToolCallParam / ToolResultBlock
# ---------------------------------------------------------------------------

@dataclass
class RuntimeContext:
    """一次运行的身份：会话 ID 即状态存储的 sessionId"""

    user_id: str = ""
    session_id: str = ""


@dataclass
class ToolCallParam:
    tool_call_id: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    runtime_context: RuntimeContext | None = None


class AgentTool(ABC):
    """Agent 原生工具接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema，随工具定义下发给模型"""
        ...

    def is_read_only(self) -> bool:
        return False

    @abstractmethod
    async def call(self, param: ToolCallParam) -> ToolResultBlock:
        ...

    # -- 下发给模型的 function 定义 -----------------------------------------

    def to_function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }

    @staticmethod
    def build_result(tool_call_id: str | None, name: str, text: str, is_error: bool) -> ToolResultBlock:
        return ToolResultBlock(
            id=tool_call_id or "",
            name=name,
            output=[TextBlock(text=text or "")],
            state=ToolResultState.ERROR if is_error else ToolResultState.SUCCESS,
        )


# ---------------------------------------------------------------------------
# 外部依赖协议 — 由 RAG / MCP 模块提供实现
# ---------------------------------------------------------------------------

class KnowledgeSearchFacade(Protocol):
    """知识库检索门面：RAG 管线在 Agent 模式下的唯一入口"""

    async def search(self, query: str, history: list[dict[str, str]]) -> str:
        ...


class McpToolExecutor(Protocol):
    """MCP 工具执行器 — mirrors rag.core.mcp.McpToolExecutor"""

    @property
    def tool_id(self) -> str:
        ...

    def tool_definition(self) -> dict[str, Any]:
        """含 name / description / inputSchema / annotations 的 MCP Tool 定义"""
        ...

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """返回 CallToolResult 形态：{"content": [...], "isError": bool}"""
        ...


class IntentNodeRegistry(Protocol):
    """意图树注册表 — mirrors rag.core.intent.IntentNodeRegistry"""

    def list_mcp_tool_nodes(self) -> list[dict[str, Any]]:
        """返回 [{"mcpToolId": ..., "name": ..., "description": ...}]"""
        ...


class McpToolRegistry(Protocol):
    """MCP 工具注册表 — mirrors rag.core.mcp.McpToolRegistry"""

    def list_all_executors(self) -> list[McpToolExecutor]:
        ...


class PromptResolver(Protocol):
    """提示词解析口 — 实现在 services.prompt_service.AgentPromptResolver"""

    async def resolve_knowledge_tool_description(self) -> str:
        ...


# ---------------------------------------------------------------------------
# KnowledgeSearchTool — mirrors tool.KnowledgeSearchTool
# ---------------------------------------------------------------------------

class KnowledgeSearchTool(AgentTool):
    """知识库检索工具：描述由当前 Agent 的提示词槽位提供"""

    TOOL_NAME = "search_knowledge"
    DISPLAY_NAME = "知识库检索"

    QUERY_PARAM = "query"
    QUERY_DESCRIPTION = "用于检索知识库的完整独立问题"

    REWRITE_CONTEXT_TURNS = 2
    """改写只用近期轮次消解指代，取多了也会被 buildRewriteRequest 截到 4 条"""

    def __init__(
        self,
        description: str,
        search_facade: KnowledgeSearchFacade,
        history_loader: Any | None = None,
    ) -> None:
        self._description = description
        self._search_facade = search_facade
        # 主 Agent 未消解干净的指代由改写兜底；loader 签名 (session_id, user_id, turns) -> list[ChatMessage]
        self._history_loader = history_loader

    @property
    def name(self) -> str:
        return self.TOOL_NAME

    @property
    def description(self) -> str:
        return self._description

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                self.QUERY_PARAM: {
                    "type": "string",
                    "description": self.QUERY_DESCRIPTION,
                },
            },
            "required": [self.QUERY_PARAM],
            "additionalProperties": False,
        }

    def is_read_only(self) -> bool:
        return True

    async def call(self, param: ToolCallParam | None) -> ToolResultBlock:
        if param is None:
            return self.build_result(None, self.TOOL_NAME, "工具调用参数不能为空", True)

        raw_query = (param.input or {}).get(self.QUERY_PARAM)
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            return self.build_result(param.tool_call_id, self.TOOL_NAME, "工具参数 query 不能为空", True)

        try:
            history = await self._recent_turns(param.runtime_context)
            result = await self._search_facade.search(query, history)
            return self.build_result(param.tool_call_id, self.TOOL_NAME, result, False)
        except Exception as e:
            logger.error("知识库检索工具调用异常: %s", e)
            return self.build_result(
                param.tool_call_id, self.TOOL_NAME, f"知识库检索异常: {e}", True,
            )

    async def _recent_turns(self, runtime_context: RuntimeContext | None) -> list[dict[str, str]]:
        """会话身份取不到时退化为无历史改写"""
        if runtime_context is None or self._history_loader is None:
            return []
        try:
            return await self._history_loader(
                runtime_context.session_id, runtime_context.user_id, self.REWRITE_CONTEXT_TURNS,
            )
        except Exception as e:
            logger.warning("加载近期轮次失败，退化为无历史改写: %s", e)
            return []


# ---------------------------------------------------------------------------
# McpToolBridge — mirrors tool.McpToolBridge
# ---------------------------------------------------------------------------

class McpToolBridge(AgentTool):
    """
    MCP 工具桥：把已连接的 MCP 执行器适配为 Agent 原生工具。

    路由描述优先取意图树配置，空则回落 MCP 服务端原始描述。
    """

    def __init__(self, executor: McpToolExecutor, description_override: str = "") -> None:
        self._executor = executor
        self._description_override = description_override or ""

    @property
    def name(self) -> str:
        return self._executor.tool_id

    @property
    def description(self) -> str:
        if self._description_override.strip():
            return self._description_override
        return str(self._executor.tool_definition().get("description") or "")

    def parameters(self) -> dict[str, Any]:
        schema = self._executor.tool_definition().get("inputSchema") or {}
        parameters: dict[str, Any] = {
            "type": schema.get("type") or "object",
            "properties": schema.get("properties") or {},
        }
        required = schema.get("required")
        if required:
            parameters["required"] = required
        return parameters

    def is_read_only(self) -> bool:
        """
        透传 MCP 的 readOnlyHint，服务端没声明就按写工具处理。

        缺省取 False 是因为猜错的两个方向不对等：把写工具当只读会放过重复副作用。
        """
        annotations = self._executor.tool_definition().get("annotations") or {}
        return annotations.get("readOnlyHint") is True

    async def call(self, param: ToolCallParam) -> ToolResultBlock:
        try:
            result = await self._executor.execute(dict(param.input or {}))
            is_error = bool(result and result.get("isError") is True)
            return self.build_result(
                param.tool_call_id, self.name, self._extract_text(result), is_error,
            )
        except Exception as e:
            logger.error("MCP 工具调用异常, toolId: %s, error: %s", self.name, e)
            return self.build_result(param.tool_call_id, self.name, f"工具调用异常: {e}", True)

    @staticmethod
    def _extract_text(result: dict[str, Any] | None) -> str:
        if not result or not result.get("content"):
            return "（工具无返回内容）"
        chunks = [
            str(item.get("text") or "").strip()
            for item in result["content"]
            if isinstance(item, dict) and item.get("type") in (None, "text")
        ]
        return "\n".join(c for c in chunks if c)


# ---------------------------------------------------------------------------
# AgentToolCatalog — mirrors tool.AgentToolCatalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class McpToolBinding:
    tool_id: str
    display_name: str
    description: str
    executor: McpToolExecutor


@dataclass(frozen=True)
class ResolvedCatalog:
    """
    一次解析定格的工具目录，展示名在构造时算好。

    快照随 Agent 实例一起缓存，重建之前所有请求看到的都是同一份。
    """

    knowledge_tool_description: str
    bindings: tuple[McpToolBinding, ...] = ()
    unavailable_tool_ids: tuple[str, ...] = ()
    display_names: dict[str, str] = field(default_factory=dict)

    def display_name_of(self, tool_name: str) -> str:
        """SSE 工具进度事件的展示名，未收录的工具回落原始名"""
        return self.display_names.get(tool_name, tool_name)


class AgentToolCatalog:
    """
    主 Agent 工具目录：固定注册 search_knowledge，并按意图树配置挂载当前可用的 MCP 工具。
    """

    def __init__(
        self,
        prompt_resolver: PromptResolver,
        search_facade: KnowledgeSearchFacade,
        history_loader: Any | None = None,
        intent_node_registry: IntentNodeRegistry | None = None,
        mcp_tool_registry: McpToolRegistry | None = None,
    ) -> None:
        self._prompt_resolver = prompt_resolver
        self._search_facade = search_facade
        self._history_loader = history_loader
        self._intent_registry = intent_node_registry
        self._mcp_registry = mcp_tool_registry

    async def resolve(self) -> ResolvedCatalog:
        """把注册表与提示词解析一次并定格：同一次请求的 Toolkit 从这份快照派生"""
        unavailable: list[str] = []
        bindings = self._resolve_mcp_bindings(unavailable)
        description = await self._resolve_knowledge_tool_description()

        names: dict[str, str] = {
            KnowledgeSearchTool.TOOL_NAME: KnowledgeSearchTool.DISPLAY_NAME,
        }
        for binding in bindings:
            names[binding.tool_id] = binding.display_name

        return ResolvedCatalog(
            knowledge_tool_description=description,
            bindings=tuple(bindings),
            unavailable_tool_ids=tuple(unavailable),
            display_names=names,
        )

    def build_tools(self, catalog: ResolvedCatalog) -> list[AgentTool]:
        """按快照构建全新工具集：过程中不再回读注册表"""
        tools: list[AgentTool] = [
            KnowledgeSearchTool(
                catalog.knowledge_tool_description,
                self._search_facade,
                self._history_loader,
            )
        ]
        tools.extend(McpToolBridge(b.executor, b.description) for b in catalog.bindings)

        # 不可用只在重建这一刻报：解析每请求都走，放解析里会刷屏
        for tool_id in catalog.unavailable_tool_ids:
            logger.warning("意图树配置的 MCP 工具当前不可用, toolId: %s", tool_id)
        return tools

    def mcp_tool_count(self) -> int:
        """
        意图树已配置且 MCP 注册表当前可用的工具数，meta 探活据此报告 MCP 配置状态。

        不走整份解析：探活不该被知识库工具声明缺失连坐。
        """
        return len(self._resolve_mcp_bindings([]))

    async def _resolve_knowledge_tool_description(self) -> str:
        description = await self._prompt_resolver.resolve_knowledge_tool_description()
        if not description or not description.strip():
            raise RuntimeError("KNOWLEDGE_TOOL_DESCRIPTION 提示词不允许为空")
        return description

    def _resolve_mcp_bindings(self, unavailable: list[str]) -> list[McpToolBinding]:
        """意图树配置与 MCP 注册表求交集，配了但当前没执行器的工具 ID 收进 unavailable"""
        if self._intent_registry is None or self._mcp_registry is None:
            return []

        nodes_by_tool_id: dict[str, list[dict[str, Any]]] = {}
        for node in self._intent_registry.list_mcp_tool_nodes():
            tool_id = str(node.get("mcpToolId") or "").strip()
            if tool_id:
                nodes_by_tool_id.setdefault(tool_id, []).append(node)

        executors: dict[str, McpToolExecutor] = {}
        for executor in self._mcp_registry.list_all_executors():
            executors[executor.tool_id] = executor  # 后者覆盖前者，与 Java toMap 一致

        bindings: list[McpToolBinding] = []
        for tool_id, nodes in nodes_by_tool_id.items():
            executor = executors.get(tool_id)
            if executor is None:
                unavailable.append(tool_id)
                continue
            bindings.append(self._to_binding(tool_id, nodes, executor))
        return bindings

    @staticmethod
    def _to_binding(
        tool_id: str,
        nodes: list[dict[str, Any]],
        executor: McpToolExecutor,
    ) -> McpToolBinding:
        display_name = tool_id
        for node in nodes:
            name = str(node.get("name") or "").strip()
            if name:
                display_name = name
                break

        descriptions: list[str] = []
        for node in nodes:
            desc = str(node.get("description") or "").strip()
            if desc and desc not in descriptions:
                descriptions.append(desc)

        return McpToolBinding(
            tool_id=tool_id,
            display_name=display_name,
            description="\n".join(descriptions),
            executor=executor,
        )
