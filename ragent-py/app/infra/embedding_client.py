"""
Embedding client interfaces — EmbeddingClient, EmbeddingService, RoutingEmbeddingService.

Mirrors Java infra embedding classes:
  - embedding.EmbeddingClient
  - embedding.EmbeddingService
  - embedding.RoutingEmbeddingService
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Protocol

from app.core.exceptions import RemoteException
from app.infra.enums import ModelCapability
from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelSelector, ModelTarget

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """
    嵌入客户端接口 — 将文本转换为向量。

    Mirrors Java embedding.EmbeddingClient.
    """

    @abstractmethod
    def provider(self) -> str:
        """获取服务提供商名称"""
        ...

    @abstractmethod
    def embed(self, texts: list[str], target: ModelTarget) -> list[list[float]]:
        """
        批量嵌入文本。

        Args:
            texts: 待嵌入的文本列表
            target: 目标模型配置

        Returns:
            向量列表，每个向量是一个 float 列表
        """
        ...


class EmbeddingService(ABC):
    """
    嵌入服务接口 — 高层嵌入访问能力。

    Mirrors Java embedding.EmbeddingService.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本"""
        ...


class RoutingEmbeddingService(EmbeddingService):
    """
    路由式嵌入服务 — 多模型候选 + 断路器。

    Mirrors Java embedding.RoutingEmbeddingService.
    """

    def __init__(
        self,
        selector: ModelSelector,
        health_store: ModelHealthStore,
        executor: ModelRoutingExecutor,
        clients: list[EmbeddingClient],
    ) -> None:
        self.selector = selector
        self.health_store = health_store
        self.executor = executor
        self.clients_by_provider: dict[str, EmbeddingClient] = {
            c.provider(): c for c in clients
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        # For embedding, use STANDARD tier
        candidates = self.selector.select_chat_candidates(False, None, None)
        # Filter to embedding-capable candidates (simplified — in real impl, separate selection)
        return self.executor.execute_with_fallback(
            ModelCapability.EMBEDDING,
            candidates,
            lambda target: self.clients_by_provider.get(target.candidate.get("provider", "")),
            lambda client, target: client.embed(texts, target),
        )
