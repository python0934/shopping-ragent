"""
app.infra — AI model infrastructure layer.

Re-exports all public symbols for convenient access.
"""

from app.infra.enums import ModelCapability, ModelProvider, Tier
from app.infra.model_routing import (
    CallPermit,
    ModelHealthStore,
    ModelRoutingExecutor,
    ModelSelector,
    ModelTarget,
)
from app.infra.chat_client import (
    ChatClient,
    ChatRequest,
    LLMService,
    RoutingLLMService,
    SimpleCancellationHandle,
    SimpleStreamCallback,
    StreamCallback,
    StreamCancellationHandle,
)
from app.infra.embedding_client import (
    EmbeddingClient,
    EmbeddingService,
    RoutingEmbeddingService,
)
from app.infra.rerank_client import (
    NoopRerankClient,
    RerankClient,
    RerankResult,
    RerankService,
    RoutingRerankService,
)
from app.infra.vlm_client import (
    RoutingVlmService,
    VlmService,
)
from app.infra.token_counter import (
    HeuristicTokenCounterService,
    TokenCounterService,
)
from app.infra.http_utils import (
    HttpMediaTypes,
    HttpResponseHelper,
    ModelClientErrorType,
    ModelClientException,
    ModelUrlResolver,
)

__all__ = [
    # enums
    "ModelCapability",
    "ModelProvider",
    "Tier",
    # model routing
    "CallPermit",
    "ModelHealthStore",
    "ModelRoutingExecutor",
    "ModelSelector",
    "ModelTarget",
    # chat
    "ChatClient",
    "ChatRequest",
    "LLMService",
    "RoutingLLMService",
    "SimpleCancellationHandle",
    "SimpleStreamCallback",
    "StreamCallback",
    "StreamCancellationHandle",
    # embedding
    "EmbeddingClient",
    "EmbeddingService",
    "RoutingEmbeddingService",
    # rerank
    "NoopRerankClient",
    "RerankClient",
    "RerankResult",
    "RerankService",
    "RoutingRerankService",
    # vlm
    "RoutingVlmService",
    "VlmService",
    # token
    "HeuristicTokenCounterService",
    "TokenCounterService",
    # http utils
    "HttpMediaTypes",
    "HttpResponseHelper",
    "ModelClientErrorType",
    "ModelClientException",
    "ModelUrlResolver",
]
