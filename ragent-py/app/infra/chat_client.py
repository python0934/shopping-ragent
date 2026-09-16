"""
Chat client interfaces — ChatClient, StreamCallback, LLMService, RoutingLLMService.

Mirrors Java infra chat classes:
  - chat.ChatClient
  - chat.StreamCallback
  - chat.StreamCancellationHandle
  - chat.LLMService
  - chat.RoutingLLMService
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.exceptions import RemoteException
from app.infra.enums import ModelCapability, Tier
from app.infra.model_routing import (
    CallPermit,
    ModelHealthStore,
    ModelRoutingExecutor,
    ModelSelector,
    ModelTarget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChatRequest — mirrors framework ChatRequest
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    """
    聊天请求对象 — 封装一次 LLM 调用的全部参数。

    Mirrors Java framework.convention.ChatRequest.
    """
    messages: list[dict[str, str]] = field(default_factory=list)
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    thinking: bool = False
    context: str | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# StreamCallback — async callback for streaming responses
# ---------------------------------------------------------------------------

class StreamCallback(Protocol):
    """流式回调接口 — 接收流式响应片段。"""

    def on_thinking(self, content: str) -> None:
        """深度思考内容片段"""
        ...

    def on_content(self, content: str) -> None:
        """正文内容片段"""
        ...

    def on_complete(self) -> None:
        """流式完成"""
        ...

    def on_error(self, error: Exception) -> None:
        """流式出错"""
        ...


class SimpleStreamCallback:
    """简单流式回调 — 收集全部内容"""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self.completed = False
        self.error: Exception | None = None

    def on_thinking(self, content: str) -> None:
        self.thinking_parts.append(content)

    def on_content(self, content: str) -> None:
        self.content_parts.append(content)

    def on_complete(self) -> None:
        self.completed = True

    def on_error(self, error: Exception) -> None:
        self.error = error

    @property
    def full_content(self) -> str:
        return "".join(self.content_parts)

    @property
    def full_thinking(self) -> str:
        return "".join(self.thinking_parts)


# ---------------------------------------------------------------------------
# StreamCancellationHandle
# ---------------------------------------------------------------------------

class StreamCancellationHandle(Protocol):
    """流取消处理器 — 用于中断正在进行的流式响应。"""

    def cancel(self) -> None:
        """取消流式请求"""
        ...


class SimpleCancellationHandle:
    """简单取消处理器"""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


# ---------------------------------------------------------------------------
# ChatClient — interface for model providers
# ---------------------------------------------------------------------------

class ChatClient(ABC):
    """
    聊天客户端接口 — 定义与 AI 模型进行对话的核心方法。

    Mirrors Java chat.ChatClient.
    """

    @abstractmethod
    def provider(self) -> str:
        """获取服务提供商名称"""
        ...

    @abstractmethod
    def chat(self, request: ChatRequest, target: ModelTarget) -> str:
        """同步聊天方法 — 发送请求并等待完整响应返回"""
        ...

    @abstractmethod
    def stream_chat(
        self,
        request: ChatRequest,
        callback: StreamCallback,
        target: ModelTarget,
    ) -> StreamCancellationHandle:
        """流式聊天方法 — 以流式方式接收模型响应"""
        ...


# ---------------------------------------------------------------------------
# LLMService — high-level LLM access interface
# ---------------------------------------------------------------------------

class LLMService(ABC):
    """
    通用大语言模型（LLM）访问接口。

    Mirrors Java chat.LLMService.
    """

    @abstractmethod
    def chat(self, request: ChatRequest, tier: Tier | None = None, preferred_model_id: str | None = None) -> str:
        """同步调用"""
        ...

    @abstractmethod
    def stream_chat(self, request: ChatRequest, callback: StreamCallback) -> StreamCancellationHandle:
        """流式调用"""
        ...


# ---------------------------------------------------------------------------
# RoutingLLMService — routing implementation with fallback
# ---------------------------------------------------------------------------

class RoutingLLMService(LLMService):
    """
    路由式 LLM 服务实现 — 多模型候选 + 断路器 + 首包探测。

    Mirrors Java chat.RoutingLLMService.
    """

    def __init__(
        self,
        selector: ModelSelector,
        health_store: ModelHealthStore,
        executor: ModelRoutingExecutor,
        clients: list[ChatClient],
    ) -> None:
        self.selector = selector
        self.health_store = health_store
        self.executor = executor
        self.clients_by_provider: dict[str, ChatClient] = {
            c.provider(): c for c in clients
        }

    def chat(self, request: ChatRequest, tier: Tier | None = None, preferred_model_id: str | None = None) -> str:
        thinking = request.thinking
        effective_tier = tier or (Tier.DEEP if thinking else Tier.STANDARD)
        candidates = self.selector.select_chat_candidates(thinking, effective_tier, preferred_model_id)

        return self.executor.execute_with_fallback(
            ModelCapability.CHAT,
            candidates,
            lambda target: self.clients_by_provider.get(target.candidate.get("provider", "")),
            lambda client, target: client.chat(request, target),
        )

    def stream_chat(self, request: ChatRequest, callback: StreamCallback) -> StreamCancellationHandle:
        thinking = request.thinking
        targets = self.selector.select_chat_candidates(thinking)
        if not targets:
            raise RemoteException("无可用大模型提供者")

        label = ModelCapability.CHAT.display_name
        last_error: Exception | None = None

        for target in targets:
            client = self.clients_by_provider.get(target.candidate.get("provider", ""))
            if client is None:
                logger.warning("%s 提供商客户端缺失: provider=%s, modelId=%s",
                               label, target.candidate.get("provider"), target.id)
                continue

            permit = self.health_store.allow_call(target.id)
            if permit is None:
                continue

            try:
                handle = client.stream_chat(request, callback, target)
            except Exception as e:
                self.health_store.mark_failure(target.id)
                last_error = e
                logger.warning("%s 流式请求启动失败，切换下一个模型。modelId=%s", label, target.id)
                continue

            if handle is None:
                self.health_store.mark_failure(target.id)
                last_error = RemoteException("流式请求未返回取消句柄")
                continue

            # Success — mark health and return
            self.health_store.mark_success(target.id)
            return handle

        # All models failed
        final_error = RemoteException("大模型调用失败，请稍后再试...", cause=last_error)
        callback.on_error(final_error)
        raise final_error
