"""
Model routing layer — ModelTarget, ModelHealthStore (circuit breaker), ModelSelector.

Mirrors Java infra model classes:
  - model.ModelTarget
  - model.ModelHealthStore
  - model.ModelSelector
  - model.ModelRoutingExecutor
  - model.ModelCaller
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from app.config import settings
from app.core.exceptions import RemoteException
from app.infra.enums import ModelCapability, ModelProvider, Tier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ModelTarget — model call configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTarget:
    """
    模型目标配置 — 封装一次模型调用所需的全部信息。

    Mirrors Java record ModelTarget(String id, ModelCandidate candidate, ProviderConfig provider, Long timeoutMs)
    """
    id: str
    candidate: dict[str, Any]  # model candidate config
    provider: dict[str, Any]   # provider config
    timeout_ms: int | None = None


# ---------------------------------------------------------------------------
# ModelHealthStore — three-state circuit breaker
# ---------------------------------------------------------------------------

class _State(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class CallPermit:
    """调用许可 — half_open_token > 0 表示持有半开探测名额"""
    model_id: str
    half_open_token: int = 0


class ModelHealthStore:
    """
    模型健康状态存储器 — 三态断路器模式。

    Mirrors Java ModelHealthStore — ConcurrentHashMap + AtomicLong.
    Python version uses threading.Lock for thread safety.
    """

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._health: dict[str, _ModelHealth] = {}
        self._probe_token_seq = 0
        self._failure_threshold = settings.ai.selection.failure_threshold
        self._open_duration_ms = settings.ai.selection.open_duration_ms

    def is_unavailable(self, model_id: str) -> bool:
        with self._lock:
            h = self._health.get(model_id)
            if h is None:
                return False
            if h.state == _State.OPEN and h.open_until > time.time() * 1000:
                return True
            if h.state == _State.HALF_OPEN and h.half_open_in_flight:
                return True
            return False

    def allow_call(self, model_id: str) -> CallPermit | None:
        if model_id is None:
            return None
        now = time.time() * 1000
        with self._lock:
            self._probe_token_seq += 1
            token = self._probe_token_seq

            h = self._health.get(model_id, _ModelHealth())

            if h.state == _State.OPEN:
                if h.open_until > now:
                    return None  # still in cooldown
                # transition to HALF_OPEN
                h.state = _State.HALF_OPEN
                h.half_open_in_flight = True
                h.half_open_token = token
                self._health[model_id] = h
                return CallPermit(model_id, token)

            if h.state == _State.HALF_OPEN:
                if h.half_open_in_flight:
                    return None  # another probe in flight
                h.half_open_in_flight = True
                h.half_open_token = token
                self._health[model_id] = h
                return CallPermit(model_id, token)

            # CLOSED — allow freely
            self._health[model_id] = h
            return CallPermit(model_id, 0)

    def mark_success(self, model_id: str) -> None:
        if model_id is None:
            return
        with self._lock:
            h = self._health.get(model_id, _ModelHealth())
            h.state = _State.CLOSED
            h.consecutive_failures = 0
            h.open_until = 0
            h.half_open_in_flight = False
            self._health[model_id] = h

    def mark_failure(self, model_id: str) -> None:
        if model_id is None:
            return
        now = time.time() * 1000
        with self._lock:
            h = self._health.get(model_id, _ModelHealth())
            if h.state == _State.HALF_OPEN:
                h.state = _State.OPEN
                h.open_until = now + self._open_duration_ms
                h.consecutive_failures = 0
                h.half_open_in_flight = False
                self._health[model_id] = h
                return
            h.consecutive_failures += 1
            if h.consecutive_failures >= self._failure_threshold:
                h.state = _State.OPEN
                h.open_until = now + self._open_duration_ms
                h.consecutive_failures = 0
            self._health[model_id] = h

    def release_half_open_permit(self, permit: CallPermit | None) -> None:
        if permit is None or permit.half_open_token <= 0:
            return
        with self._lock:
            h = self._health.get(permit.model_id)
            if h and h.state == _State.HALF_OPEN and h.half_open_in_flight and h.half_open_token == permit.half_open_token:
                h.half_open_in_flight = False


class _ModelHealth:
    __slots__ = ("consecutive_failures", "open_until", "half_open_in_flight", "half_open_token", "state")

    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        self.open_until: float = 0
        self.half_open_in_flight: bool = False
        self.half_open_token: int = 0
        self.state: _State = _State.CLOSED


# ---------------------------------------------------------------------------
# ModelSelector — select candidate models by capability & tier
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    模型选择器 — 根据能力和档位选择候选模型列表。

    Mirrors Java ModelSelector — reads from AIModelProperties.
    """

    def select_chat_candidates(
        self,
        thinking: bool = False,
        tier: Tier | None = None,
        preferred_model_id: str | None = None,
    ) -> list[ModelTarget]:
        """Select chat model candidates based on tier and thinking mode."""
        effective_tier = tier or Tier.DEEP if thinking else (tier or Tier.STANDARD)
        ai_cfg = settings.ai
        tier_cfg = ai_cfg.chat.tiers.get(effective_tier.value, {})
        candidates = tier_cfg.get("candidates", [])
        providers = ai_cfg.providers

        targets: list[ModelTarget] = []
        for c in candidates:
            model_id = c.get("id", "")
            provider_name = c.get("provider", "")
            provider_cfg = providers.get(provider_name, {})
            timeout_ms = tier_cfg.get("timeout_ms")
            targets.append(ModelTarget(
                id=model_id,
                candidate=c,
                provider=provider_cfg,
                timeout_ms=timeout_ms,
            ))

        # Move preferred model to front if specified
        if preferred_model_id:
            for i, t in enumerate(targets):
                if t.id == preferred_model_id:
                    targets.insert(0, targets.pop(i))
                    break

        return targets

    def select_vlm_candidates(self) -> list[ModelTarget]:
        """Select VLM model candidates from ai.vlm config."""
        ai_cfg = settings.ai
        vlm_cfg = ai_cfg.vlm
        candidates = vlm_cfg.candidates
        providers = ai_cfg.providers

        targets: list[ModelTarget] = []
        for c in candidates:
            c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
            model_id = c_dict.get("id", "")
            provider_name = c_dict.get("provider", "")
            provider_obj = getattr(providers, provider_name, None)
            provider_cfg = provider_obj.model_dump() if provider_obj and hasattr(provider_obj, "model_dump") else {}
            targets.append(ModelTarget(
                id=model_id,
                candidate=c_dict,
                provider=provider_cfg,
            ))
        return targets

    def select_embedding_candidates(self) -> list[ModelTarget]:
        """Select embedding model candidates from ai.embedding config."""
        ai_cfg = settings.ai
        emb_cfg = ai_cfg.embedding
        candidates = emb_cfg.candidates
        providers = ai_cfg.providers

        targets: list[ModelTarget] = []
        for c in candidates:
            c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
            model_id = c_dict.get("id", "")
            provider_name = c_dict.get("provider", "")
            provider_obj = getattr(providers, provider_name, None)
            provider_cfg = provider_obj.model_dump() if provider_obj and hasattr(provider_obj, "model_dump") else {}
            targets.append(ModelTarget(
                id=model_id,
                candidate=c_dict,
                provider=provider_cfg,
            ))
        return targets

    def select_rerank_candidates(self) -> list[ModelTarget]:
        """Select rerank model candidates from ai.rerank config."""
        ai_cfg = settings.ai
        rerank_cfg = ai_cfg.rerank
        candidates = rerank_cfg.candidates
        providers = ai_cfg.providers

        targets: list[ModelTarget] = []
        for c in candidates:
            c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
            model_id = c_dict.get("id", "")
            provider_name = c_dict.get("provider", "")
            provider_obj = getattr(providers, provider_name, None)
            provider_cfg = provider_obj.model_dump() if provider_obj and hasattr(provider_obj, "model_dump") else {}
            targets.append(ModelTarget(
                id=model_id,
                candidate=c_dict,
                provider=provider_cfg,
            ))
        return targets


# ---------------------------------------------------------------------------
# ModelRoutingExecutor — execute with fallback across candidates
# ---------------------------------------------------------------------------

ModelCaller = Callable[[Any, ModelTarget], Any]


class ModelRoutingExecutor:
    """
    模型路由执行器 — 遍历候选模型，失败自动切换下一个。

    Mirrors Java ModelRoutingExecutor.executeWithFallback().
    """

    def __init__(self, health_store: ModelHealthStore) -> None:
        self.health_store = health_store

    def execute_with_fallback(
        self,
        capability: ModelCapability,
        targets: list[ModelTarget],
        client_resolver: Callable[[ModelTarget], Any],
        caller: ModelCaller,
    ) -> Any:
        """Execute a model call with automatic fallback on failure."""
        last_error: Exception | None = None
        label = capability.display_name

        for target in targets:
            if self.health_store.is_unavailable(target.id):
                continue

            client = client_resolver(target)
            if client is None:
                logger.warning("%s 提供商客户端缺失: provider=%s, modelId=%s",
                               label, target.candidate.get("provider"), target.id)
                continue

            try:
                return caller(client, target)
            except Exception as e:
                self.health_store.mark_failure(target.id)
                last_error = e
                logger.warning("%s 调用失败，切换下一个模型。modelId=%s, provider=%s, error=%s",
                               label, target.id, target.candidate.get("provider"), e)

        raise RemoteException(
            f"{label} 所有模型调用失败",
            cause=last_error,
        )
