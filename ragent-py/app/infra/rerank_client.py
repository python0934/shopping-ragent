"""
Rerank client interfaces — RerankClient, RerankService, RoutingRerankService.

Mirrors Java infra rerank classes:
  - rerank.RerankClient
  - rerank.RerankService
  - rerank.RoutingRerankService
  - rerank.NoopRerankClient
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.infra.enums import ModelCapability
from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelSelector, ModelTarget

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """重排序结果"""
    index: int
    score: float
    document: str


class RerankClient(ABC):
    """
    重排序客户端接口 — 对文档列表按查询相关性重排。

    Mirrors Java rerank.RerankClient.
    """

    @abstractmethod
    def provider(self) -> str:
        """获取服务提供商名称"""
        ...

    @abstractmethod
    def rerank(self, query: str, documents: list[str], target: ModelTarget) -> list[RerankResult]:
        """
        对文档列表按查询相关性重排。

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            target: 目标模型配置

        Returns:
            按相关性排序的结果列表
        """
        ...


class RerankService(ABC):
    """
    重排序服务接口。

    Mirrors Java rerank.RerankService.
    """

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        """对文档列表按查询相关性重排"""
        ...


class RoutingRerankService(RerankService):
    """
    路由式重排序服务 — 多模型候选 + 断路器。

    Mirrors Java rerank.RoutingRerankService.
    """

    def __init__(
        self,
        selector: ModelSelector,
        health_store: ModelHealthStore,
        executor: ModelRoutingExecutor,
        clients: list[RerankClient],
    ) -> None:
        self.selector = selector
        self.health_store = health_store
        self.executor = executor
        self.clients_by_provider: dict[str, RerankClient] = {
            c.provider(): c for c in clients
        }

    def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        candidates = self.selector.select_chat_candidates(False, None, None)
        return self.executor.execute_with_fallback(
            ModelCapability.RERANK,
            candidates,
            lambda target: self.clients_by_provider.get(target.candidate.get("provider", "")),
            lambda client, target: client.rerank(query, documents, target),
        )


class NoopRerankClient(RerankClient):
    """
    空操作重排序客户端 — 按原始顺序返回，用于未配置重排序模型时。

    Mirrors Java rerank.NoopRerankClient.
    """

    def provider(self) -> str:
        return "noop"

    def rerank(self, query: str, documents: list[str], target: ModelTarget) -> list[RerankResult]:
        return [RerankResult(index=i, score=1.0 - i * 0.01, document=doc) for i, doc in enumerate(documents)]
