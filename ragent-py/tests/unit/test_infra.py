"""
Phase 2 unit tests — AI model infrastructure layer.

Tests cover:
  - enums (ModelProvider, Tier, ModelCapability)
  - model_routing (ModelTarget, ModelHealthStore, ModelSelector, ModelRoutingExecutor)
  - chat_client (ChatRequest, SimpleStreamCallback, SimpleCancellationHandle, RoutingLLMService)
  - embedding_client (RoutingEmbeddingService)
  - rerank_client (NoopRerankClient, RerankResult, RoutingRerankService)
  - vlm_client (RoutingVlmService._extract_content, _build_multimodal_body)
  - token_counter (HeuristicTokenCounterService)
  - http_utils (ModelUrlResolver, ModelClientErrorType, ModelClientException, HttpResponseHelper)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.infra.enums import ModelCapability, ModelProvider, Tier


# ===========================================================================
# TestEnums
# ===========================================================================

class TestModelProvider:
    def test_values(self):
        assert ModelProvider.OLLAMA.value == "ollama"
        assert ModelProvider.BAI_LIAN.value == "bailian"
        assert ModelProvider.SILICON_FLOW.value == "siliconflow"
        assert ModelProvider.AI_HUB_MIX.value == "aihubmix"
        assert ModelProvider.NOOP.value == "noop"

    def test_matches(self):
        assert ModelProvider.OLLAMA.matches("ollama")
        assert ModelProvider.OLLAMA.matches("OLLAMA")
        assert not ModelProvider.OLLAMA.matches(None)
        assert not ModelProvider.OLLAMA.matches("bailian")


class TestTier:
    def test_values(self):
        assert Tier.FAST.value == "fast"
        assert Tier.STANDARD.value == "standard"
        assert Tier.DEEP.value == "deep"


class TestModelCapability:
    def test_values(self):
        assert ModelCapability.CHAT.value == "chat"
        assert ModelCapability.EMBEDDING.value == "embedding"
        assert ModelCapability.RERANK.value == "rerank"
        assert ModelCapability.VLM.value == "vlm"

    def test_display_name(self):
        assert ModelCapability.CHAT.display_name == "大模型对话"
        assert ModelCapability.EMBEDDING.display_name == "向量嵌入"
        assert ModelCapability.RERANK.display_name == "重排序"
        assert ModelCapability.VLM.display_name == "视觉语言模型"


# ===========================================================================
# TestModelTarget
# ===========================================================================

class TestModelTarget:
    def test_creation(self):
        from app.infra.model_routing import ModelTarget
        target = ModelTarget(
            id="model-1",
            candidate={"model": "gpt-4", "provider": "openai"},
            provider={"url": "https://api.openai.com", "api_key": "sk-test"},
            timeout_ms=30000,
        )
        assert target.id == "model-1"
        assert target.candidate["model"] == "gpt-4"
        assert target.provider["url"] == "https://api.openai.com"
        assert target.timeout_ms == 30000

    def test_frozen(self):
        from app.infra.model_routing import ModelTarget
        target = ModelTarget(id="m1", candidate={}, provider={})
        with pytest.raises(AttributeError):
            target.id = "m2"


# ===========================================================================
# TestModelHealthStore
# ===========================================================================

class TestModelHealthStore:
    def test_initial_state_allows_call(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        permit = store.allow_call("model-1")
        assert permit is not None
        assert permit.half_open_token == 0  # CLOSED state

    def test_mark_failure_triggers_open(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        # Default threshold is 2
        store.mark_failure("model-1")
        assert not store.is_unavailable("model-1")
        store.mark_failure("model-1")
        assert store.is_unavailable("model-1")

    def test_mark_success_resets(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        store.mark_failure("model-1")
        store.mark_success("model-1")
        assert not store.is_unavailable("model-1")
        permit = store.allow_call("model-1")
        assert permit is not None

    def test_half_open_after_cooldown(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        # Trigger open
        store.mark_failure("model-1")
        store.mark_failure("model-1")
        assert store.is_unavailable("model-1")

        # Simulate cooldown expiry by manipulating open_until
        with store._lock:
            h = store._health["model-1"]
            h.open_until = time.time() * 1000 - 1000  # in the past

        # Now allow_call should transition to HALF_OPEN
        permit = store.allow_call("model-1")
        assert permit is not None
        assert permit.half_open_token > 0

    def test_half_open_failure_reopens(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        store.mark_failure("model-1")
        store.mark_failure("model-1")

        # Move to half-open
        with store._lock:
            h = store._health["model-1"]
            h.open_until = time.time() * 1000 - 1000

        permit = store.allow_call("model-1")
        assert permit is not None

        # Failure in half-open → back to OPEN
        store.mark_failure("model-1")
        assert store.is_unavailable("model-1")

    def test_release_half_open_permit(self):
        from app.infra.model_routing import CallPermit, ModelHealthStore
        store = ModelHealthStore()
        store.mark_failure("model-1")
        store.mark_failure("model-1")

        with store._lock:
            store._health["model-1"].open_until = time.time() * 1000 - 1000

        permit = store.allow_call("model-1")
        assert permit is not None

        store.release_half_open_permit(permit)
        # After release, half_open_in_flight should be False
        with store._lock:
            h = store._health["model-1"]
            assert not h.half_open_in_flight

    def test_none_model_id(self):
        from app.infra.model_routing import ModelHealthStore
        store = ModelHealthStore()
        assert store.allow_call(None) is None
        store.mark_success(None)  # should not raise
        store.mark_failure(None)  # should not raise
        assert not store.is_unavailable(None)


# ===========================================================================
# TestModelRoutingExecutor
# ===========================================================================

class TestModelRoutingExecutor:
    def test_execute_with_fallback_success(self):
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelTarget
        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        target = ModelTarget(id="m1", candidate={"provider": "test"}, provider={})

        result = executor.execute_with_fallback(
            ModelCapability.CHAT,
            [target],
            lambda t: "mock_client",
            lambda client, t: "hello",
        )
        assert result == "hello"

    def test_execute_with_fallback_skips_unavailable(self):
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelTarget
        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        t1 = ModelTarget(id="m1", candidate={"provider": "test"}, provider={})
        t2 = ModelTarget(id="m2", candidate={"provider": "test"}, provider={})

        # Mark m1 as unavailable
        store.mark_failure("m1")
        store.mark_failure("m1")

        result = executor.execute_with_fallback(
            ModelCapability.CHAT,
            [t1, t2],
            lambda t: "mock_client",
            lambda client, t: f"from_{t.id}",
        )
        assert result == "from_m2"

    def test_execute_with_fallback_all_fail(self):
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelTarget
        from app.core.exceptions import RemoteException
        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        target = ModelTarget(id="m1", candidate={"provider": "test"}, provider={})

        with pytest.raises(RemoteException):
            executor.execute_with_fallback(
                ModelCapability.CHAT,
                [target],
                lambda t: "mock_client",
                lambda client, t: (_ for _ in ()).throw(ValueError("boom")),
            )

    def test_execute_with_fallback_skips_none_client(self):
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelTarget
        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        target = ModelTarget(id="m1", candidate={"provider": "test"}, provider={})

        with pytest.raises(Exception):
            executor.execute_with_fallback(
                ModelCapability.CHAT,
                [target],
                lambda t: None,  # client resolver returns None
                lambda client, t: "hello",
            )


# ===========================================================================
# TestChatClient
# ===========================================================================

class TestChatRequest:
    def test_defaults(self):
        from app.infra.chat_client import ChatRequest
        req = ChatRequest()
        assert req.messages == []
        assert req.system_prompt is None
        assert req.temperature is None
        assert req.thinking is False

    def test_custom(self):
        from app.infra.chat_client import ChatRequest
        req = ChatRequest(
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="You are helpful",
            temperature=0.7,
            thinking=True,
        )
        assert len(req.messages) == 1
        assert req.thinking is True


class TestSimpleStreamCallback:
    def test_collects_content(self):
        from app.infra.chat_client import SimpleStreamCallback
        cb = SimpleStreamCallback()
        cb.on_content("Hello ")
        cb.on_content("World")
        assert cb.full_content == "Hello World"
        assert not cb.completed

    def test_collects_thinking(self):
        from app.infra.chat_client import SimpleStreamCallback
        cb = SimpleStreamCallback()
        cb.on_thinking("Let me think... ")
        cb.on_thinking("about this")
        assert cb.full_thinking == "Let me think... about this"

    def test_complete(self):
        from app.infra.chat_client import SimpleStreamCallback
        cb = SimpleStreamCallback()
        cb.on_complete()
        assert cb.completed

    def test_error(self):
        from app.infra.chat_client import SimpleStreamCallback
        cb = SimpleStreamCallback()
        err = ValueError("test error")
        cb.on_error(err)
        assert cb.error is err


class TestSimpleCancellationHandle:
    def test_cancel(self):
        from app.infra.chat_client import SimpleCancellationHandle
        handle = SimpleCancellationHandle()
        assert not handle.is_cancelled
        handle.cancel()
        assert handle.is_cancelled


# ===========================================================================
# TestRoutingLLMService
# ===========================================================================

class TestRoutingLLMService:
    def test_chat_success(self):
        from app.infra.chat_client import ChatRequest, RoutingLLMService
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelSelector, ModelTarget

        mock_client = MagicMock()
        mock_client.provider.return_value = "test"
        mock_client.chat.return_value = "response"

        selector = MagicMock(spec=ModelSelector)
        target = ModelTarget(id="m1", candidate={"provider": "test"}, provider={})
        selector.select_chat_candidates.return_value = [target]

        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        service = RoutingLLMService(selector, store, executor, [mock_client])

        result = service.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
        assert result == "response"

    def test_chat_fallback_on_failure(self):
        from app.infra.chat_client import ChatRequest, RoutingLLMService
        from app.infra.model_routing import ModelHealthStore, ModelRoutingExecutor, ModelSelector, ModelTarget

        mock_client1 = MagicMock()
        mock_client1.provider.return_value = "test1"
        mock_client1.chat.side_effect = ValueError("boom")

        mock_client2 = MagicMock()
        mock_client2.provider.return_value = "test2"
        mock_client2.chat.return_value = "fallback response"

        selector = MagicMock(spec=ModelSelector)
        t1 = ModelTarget(id="m1", candidate={"provider": "test1"}, provider={})
        t2 = ModelTarget(id="m2", candidate={"provider": "test2"}, provider={})
        selector.select_chat_candidates.return_value = [t1, t2]

        store = ModelHealthStore()
        executor = ModelRoutingExecutor(store)
        service = RoutingLLMService(selector, store, executor, [mock_client1, mock_client2])

        result = service.chat(ChatRequest())
        assert result == "fallback response"


# ===========================================================================
# TestRerankClient
# ===========================================================================

class TestNoopRerankClient:
    def test_provider(self):
        from app.infra.rerank_client import NoopRerankClient
        client = NoopRerankClient()
        assert client.provider() == "noop"

    def test_rerank_preserves_order(self):
        from app.infra.rerank_client import NoopRerankClient
        from app.infra.model_routing import ModelTarget
        client = NoopRerankClient()
        target = ModelTarget(id="noop", candidate={}, provider={})
        docs = ["doc1", "doc2", "doc3"]
        results = client.rerank("query", docs, target)
        assert len(results) == 3
        assert results[0].index == 0
        assert results[0].document == "doc1"
        assert results[1].index == 1
        assert results[2].index == 2

    def test_rerank_empty(self):
        from app.infra.rerank_client import NoopRerankClient
        from app.infra.model_routing import ModelTarget
        client = NoopRerankClient()
        target = ModelTarget(id="noop", candidate={}, provider={})
        results = client.rerank("query", [], target)
        assert results == []


class TestRerankResult:
    def test_dataclass(self):
        from app.infra.rerank_client import RerankResult
        r = RerankResult(index=0, score=0.95, document="hello")
        assert r.index == 0
        assert r.score == 0.95
        assert r.document == "hello"


# ===========================================================================
# TestTokenCounter
# ===========================================================================

class TestHeuristicTokenCounterService:
    def test_empty_text(self):
        from app.infra.token_counter import HeuristicTokenCounterService
        svc = HeuristicTokenCounterService()
        assert svc.count_tokens("") == 0
        assert svc.count_tokens(None) == 0
        assert svc.count_tokens("   ") == 0

    def test_ascii_text(self):
        from app.infra.token_counter import HeuristicTokenCounterService
        svc = HeuristicTokenCounterService()
        # "hello" = 5 ascii chars → (5+3)//4 = 2 tokens
        assert svc.count_tokens("hello") == 2

    def test_cjk_text(self):
        from app.infra.token_counter import HeuristicTokenCounterService
        svc = HeuristicTokenCounterService()
        # "你好世界" = 4 CJK chars → 4 tokens
        assert svc.count_tokens("你好世界") == 4

    def test_mixed_text(self):
        from app.infra.token_counter import HeuristicTokenCounterService
        svc = HeuristicTokenCounterService()
        # "hello你好" = 5 ascii + 2 CJK
        # ascii: (5+3)//4 = 2, CJK: 2, total = 4
        assert svc.count_tokens("hello你好") == 4

    def test_single_char_minimum(self):
        from app.infra.token_counter import HeuristicTokenCounterService
        svc = HeuristicTokenCounterService()
        # Single ASCII char → (1+3)//4 = 1 token
        assert svc.count_tokens("a") == 1


# ===========================================================================
# TestHttpUtils
# ===========================================================================

class TestModelClientErrorType:
    def test_from_http_status(self):
        from app.infra.http_utils import ModelClientErrorType
        assert ModelClientErrorType.from_http_status(401) == ModelClientErrorType.UNAUTHORIZED
        assert ModelClientErrorType.from_http_status(403) == ModelClientErrorType.UNAUTHORIZED
        assert ModelClientErrorType.from_http_status(429) == ModelClientErrorType.RATE_LIMITED
        assert ModelClientErrorType.from_http_status(500) == ModelClientErrorType.SERVER_ERROR
        assert ModelClientErrorType.from_http_status(502) == ModelClientErrorType.SERVER_ERROR
        assert ModelClientErrorType.from_http_status(400) == ModelClientErrorType.CLIENT_ERROR
        assert ModelClientErrorType.from_http_status(404) == ModelClientErrorType.CLIENT_ERROR


class TestModelClientException:
    def test_creation(self):
        from app.infra.http_utils import ModelClientErrorType, ModelClientException
        exc = ModelClientException("test error", ModelClientErrorType.NETWORK_ERROR, 500)
        assert exc.args[0] == "test error"
        assert exc.error_type == ModelClientErrorType.NETWORK_ERROR
        assert exc.status_code == 500

    def test_with_cause(self):
        from app.infra.http_utils import ModelClientErrorType, ModelClientException
        cause = ValueError("original")
        exc = ModelClientException("wrapped", ModelClientErrorType.PROVIDER_ERROR, cause=cause)
        assert exc.__cause__ is cause


class TestModelUrlResolver:
    def test_candidate_url_priority(self):
        from app.infra.http_utils import ModelUrlResolver
        provider = {"url": "https://base.com", "endpoints": {"chat": "/v1/chat"}}
        candidate = {"url": "https://custom.com/chat", "model": "gpt-4"}
        result = ModelUrlResolver.resolve_url(provider, candidate, ModelCapability.CHAT)
        assert result == "https://custom.com/chat"

    def test_provider_url_with_endpoint(self):
        from app.infra.http_utils import ModelUrlResolver
        provider = {"url": "https://api.openai.com", "endpoints": {"chat": "/v1/chat/completions"}}
        result = ModelUrlResolver.resolve_url(provider, None, ModelCapability.CHAT)
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_join_url_handles_slashes(self):
        from app.infra.http_utils import ModelUrlResolver
        # Both have slash
        assert ModelUrlResolver._join_url("https://base.com/", "/path") == "https://base.com/path"
        # Neither has slash
        assert ModelUrlResolver._join_url("https://base.com", "path") == "https://base.com/path"
        # Base has slash, path doesn't
        assert ModelUrlResolver._join_url("https://base.com/", "path") == "https://base.com/path"
        # Base no slash, path has
        assert ModelUrlResolver._join_url("https://base.com", "/path") == "https://base.com/path"

    def test_missing_provider_url_raises(self):
        from app.infra.http_utils import ModelClientException, ModelUrlResolver
        with pytest.raises(ModelClientException):
            ModelUrlResolver.resolve_url({}, None, ModelCapability.CHAT)

    def test_missing_endpoint_raises(self):
        from app.infra.http_utils import ModelClientException, ModelUrlResolver
        with pytest.raises(ModelClientException):
            ModelUrlResolver.resolve_url({"url": "https://base.com"}, None, ModelCapability.CHAT)


class TestHttpResponseHelper:
    def test_read_body_none(self):
        from app.infra.http_utils import HttpResponseHelper
        assert HttpResponseHelper.read_body(None) == ""

    def test_read_body_bytes(self):
        from app.infra.http_utils import HttpResponseHelper
        assert HttpResponseHelper.read_body(b"hello") == "hello"

    def test_parse_json_valid(self):
        from app.infra.http_utils import HttpResponseHelper
        result = HttpResponseHelper.parse_json('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_parse_json_none_raises(self):
        from app.infra.http_utils import HttpResponseHelper, ModelClientException
        with pytest.raises(ModelClientException, match="响应为空"):
            HttpResponseHelper.parse_json(None, "test")

    def test_parse_json_invalid_raises(self):
        from app.infra.http_utils import HttpResponseHelper, ModelClientException
        with pytest.raises(ModelClientException, match="JSON 解析失败"):
            HttpResponseHelper.parse_json("not json", "test")

    def test_require_provider_missing(self):
        from app.infra.http_utils import HttpResponseHelper, ModelClientException
        with pytest.raises(ModelClientException, match="提供商配置缺失"):
            HttpResponseHelper.require_provider(None, "test")

    def test_require_api_key_missing(self):
        from app.infra.http_utils import HttpResponseHelper, ModelClientException
        with pytest.raises(ModelClientException, match="API密钥缺失"):
            HttpResponseHelper.require_api_key({"api_key": ""}, "test")

    def test_require_model_missing(self):
        from app.infra.http_utils import HttpResponseHelper, ModelClientException
        from app.infra.model_routing import ModelTarget
        target = ModelTarget(id="m1", candidate={"model": ""}, provider={})
        with pytest.raises(ModelClientException, match="模型名称缺失"):
            HttpResponseHelper.require_model(target, "test")


# ===========================================================================
# TestVlmClient
# ===========================================================================

class TestRoutingVlmServiceExtractContent:
    def test_valid_response(self):
        from app.infra.vlm_client import RoutingVlmService
        resp = {
            "choices": [{"message": {"content": "A cat sitting on a mat"}}]
        }
        assert RoutingVlmService._extract_content(resp) == "A cat sitting on a mat"

    def test_missing_choices(self):
        from app.infra.http_utils import ModelClientException
        from app.infra.vlm_client import RoutingVlmService
        with pytest.raises(ModelClientException, match="缺少 choices"):
            RoutingVlmService._extract_content({})

    def test_empty_choices(self):
        from app.infra.http_utils import ModelClientException
        from app.infra.vlm_client import RoutingVlmService
        with pytest.raises(ModelClientException, match="choices 为空"):
            RoutingVlmService._extract_content({"choices": []})

    def test_missing_message(self):
        from app.infra.http_utils import ModelClientException
        from app.infra.vlm_client import RoutingVlmService
        with pytest.raises(ModelClientException, match="缺少 message"):
            RoutingVlmService._extract_content({"choices": [{"no_message": True}]})

    def test_missing_content(self):
        from app.infra.http_utils import ModelClientException
        from app.infra.vlm_client import RoutingVlmService
        with pytest.raises(ModelClientException, match="缺少 content"):
            RoutingVlmService._extract_content({"choices": [{"message": {"role": "assistant"}}]})

    def test_null_content(self):
        from app.infra.http_utils import ModelClientException
        from app.infra.vlm_client import RoutingVlmService
        with pytest.raises(ModelClientException, match="缺少 content"):
            RoutingVlmService._extract_content({"choices": [{"message": {"content": None}}]})


# ===========================================================================
# TestInfraPackageImport
# ===========================================================================

class TestInfraPackageImport:
    def test_all_symbols_importable(self):
        """Verify all symbols in __all__ are importable from app.infra"""
        import app.infra
        for name in app.infra.__all__:
            assert hasattr(app.infra, name), f"{name} not found in app.infra"

    def test_key_classes(self):
        from app.infra import (
            ChatClient,
            ChatRequest,
            EmbeddingClient,
            HeuristicTokenCounterService,
            ModelCapability,
            ModelClientException,
            ModelHealthStore,
            ModelProvider,
            ModelSelector,
            ModelTarget,
            ModelUrlResolver,
            NoopRerankClient,
            RerankClient,
            RerankResult,
            RoutingLLMService,
            RoutingVlmService,
            Tier,
            TokenCounterService,
            VlmService,
        )
        # Just verify they're all importable
        assert ChatClient is not None
        assert VlmService is not None
