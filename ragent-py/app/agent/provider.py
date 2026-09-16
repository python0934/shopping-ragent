"""
ReAct Agent 装配 — mirrors agent.service.handler.ReActAgentProvider.

按一次请求的上下文把模型客户端、工具目录、记忆中间件、状态存储拼成一个 ReActAgent。
每次请求现装：Agent 持有 AsyncSession，跨请求复用会把已关闭的会话带进循环。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.compaction import (
    AgentContextCompactionMiddleware,
    AgentContextCompactor,
    AgentConversationSummarizer,
)
from app.agent.memory import AgentContextTrimmer, AgentMemoryBudget
from app.agent.react_agent import (
    AgentModelClient,
    ModelCompletion,
    OpenAiCompatModelClient,
    ReActAgent,
)
from app.agent.state_store import PgAgentStateStore
from app.agent.tools import AgentToolCatalog, ResolvedCatalog
from app.config import settings
from app.services.prompt_service import AgentPromptResolver, AgentPromptSlot

logger = logging.getLogger(__name__)


class _Unset:
    """区分「没传」与「显式传 None 关掉 MCP」"""


_UNSET = _Unset()


def _default_intent_registry() -> Any:
    from app.services.intent_registry import intent_node_registry

    return intent_node_registry


def _default_mcp_registry() -> Any:
    from app.mcp import mcp_tool_registry

    return mcp_tool_registry


class _ModelClientLlm:
    """把流式模型客户端收成一个「拿一段文本」的同步口，供摘要与检索合成使用"""

    def __init__(self, client: AgentModelClient) -> None:
        self.client = client

    async def complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt or ""})
        completion: ModelCompletion = await self.client.stream_completion(messages, [], None)
        return completion.text


class _PromptResolverAdapter:
    """工具目录只需要一个槽位，收窄成单方法口，避免整份 Resolver 渗进 agent 包"""

    def __init__(self, resolver: AgentPromptResolver) -> None:
        self.resolver = resolver

    async def resolve_knowledge_tool_description(self) -> str:
        return await self.resolver.resolve(AgentPromptSlot.KNOWLEDGE_TOOL_DESCRIPTION)


class ReActAgentProvider:
    """ReAct Agent 工厂"""

    def __init__(
        self,
        model_client: AgentModelClient | None = None,
        embedding_service: Any | None = None,
        rerank_service: Any | None = None,
        intent_node_registry: Any = _UNSET,
        mcp_tool_registry: Any = _UNSET,
        cache_manager: Any | None = None,
    ) -> None:
        self.model_client = model_client or OpenAiCompatModelClient()
        self.embedding_service = embedding_service
        self.rerank_service = rerank_service
        # 默认挂进程级单例：MCP 工具与意图树快照在 lifespan 里已就绪，
        # 测试想关掉就显式传 None
        self.intent_node_registry = (
            _default_intent_registry() if isinstance(intent_node_registry, _Unset)
            else intent_node_registry
        )
        self.mcp_tool_registry = (
            _default_mcp_registry() if isinstance(mcp_tool_registry, _Unset)
            else mcp_tool_registry
        )
        self.cache_manager = cache_manager

    async def build(
        self,
        db: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> tuple[ReActAgent, ResolvedCatalog]:
        """
        装配一次运行的 Agent。

        返回 ``(agent, catalog)``：catalog 一并给出，SSE 桥要用它的展示名映射。
        """
        from app.services.knowledge_search_facade import KnowledgeSearchFacade

        await self.refresh_intent_snapshot(db)

        prompt_resolver = AgentPromptResolver(db, self.cache_manager)
        llm = _ModelClientLlm(self.model_client)

        facade = KnowledgeSearchFacade(
            db=db,
            llm=llm,
            embedding_service=self.embedding_service,
            rerank_service=self.rerank_service,
            prompt_resolver=prompt_resolver,
        )

        conversation_service = _HistoryLoader(db, user_id)
        catalog_builder = AgentToolCatalog(
            prompt_resolver=_PromptResolverAdapter(prompt_resolver),
            search_facade=facade,
            history_loader=conversation_service.load,
            intent_node_registry=self.intent_node_registry,
            mcp_tool_registry=self.mcp_tool_registry,
        )
        catalog = await catalog_builder.resolve()
        tools = catalog_builder.build_tools(catalog)

        budget = AgentMemoryBudget.from_settings()
        memory = self._build_memory(db, prompt_resolver, llm, budget, user_id, session_id)

        system_prompt = await prompt_resolver.resolve(AgentPromptSlot.AGENT_MAIN)

        agent = ReActAgent(
            model_client=self.model_client,
            tools=tools,
            system_prompt=system_prompt,
            state_store=PgAgentStateStore(db),
            max_iters=settings.agent.max_iters,
            max_retries=settings.agent.max_retries,
            memory=memory,
            catalog=catalog,
        )
        return agent, catalog

    @staticmethod
    def _build_memory(
        db: AsyncSession,
        prompt_resolver: AgentPromptResolver,
        llm: _ModelClientLlm,
        budget: AgentMemoryBudget,
        user_id: str,
        session_id: str,
    ) -> AgentContextCompactionMiddleware | None:
        """记忆门关掉时返回 None：循环里连裁剪都不做，与 Java 侧语义一致"""
        if not budget.enabled:
            return None

        summarizer = AgentConversationSummarizer(llm, prompt_resolver, budget)
        compactor = AgentContextCompactor(summarizer, budget, db if budget.summary_enabled else None)
        return AgentContextCompactionMiddleware(AgentContextTrimmer(budget), compactor, budget)

    async def refresh_intent_snapshot(self, db: AsyncSession) -> None:
        """
        按请求刷新意图树快照。

        Java 侧每次读 Redis，这里读侧是同步内存快照，只能靠调用方在异步上下文里
        推一把；不刷的话启动时 DB 不可达就永久绑不上 MCP 工具。刷失败沿用上一份。
        """
        refresh = getattr(self.intent_node_registry, "refresh", None)
        if refresh is None:
            return
        try:
            await refresh(db)
        except Exception as e:  # noqa: BLE001 - 快照不是硬依赖
            logger.warning("意图树快照刷新失败，沿用上一份: %s", e)

    def mcp_tool_count(self) -> int:
        """meta 探活用：意图树已配置且 MCP 注册表当前可用的工具数"""
        if self.intent_node_registry is None or self.mcp_tool_registry is None:
            return 0
        catalog = AgentToolCatalog(
            prompt_resolver=_NullPromptResolver(),
            search_facade=_NullSearchFacade(),
            intent_node_registry=self.intent_node_registry,
            mcp_tool_registry=self.mcp_tool_registry,
        )
        return catalog.mcp_tool_count()


class _NullPromptResolver:
    """探活路径不解析提示词，给个空口占位"""

    async def resolve_knowledge_tool_description(self) -> str:
        return ""


class _NullSearchFacade:
    """探活路径不检索"""

    async def search(self, query: str, history: list[dict[str, str]]) -> str:
        return ""


class _HistoryLoader:
    """
    近期轮次加载口 — 把 AgentConversationService 的查询收成一个可调用。

    延迟导入避免 agent 包与 services 包互相依赖。
    """

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        self.db = db
        self.user_id = user_id

    async def load(self, session_id: str, user_id: str, turns: int) -> list[dict[str, str]]:
        from app.services.agent_conversation_service import AgentConversationService

        service = AgentConversationService(self.db)
        return await service.load_recent_turns(user_id or self.user_id, session_id, turns)
