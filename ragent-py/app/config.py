"""
Application configuration — mirrors application.yaml.

Uses Pydantic BaseSettings with nested models to replicate Spring's
ConfigurationProperties hierarchy.  All values can be overridden via
environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Nested config models (mirror Java @ConfigurationProperties classes)
# ---------------------------------------------------------------------------

class ServerSettings(BaseSettings):
    port: int = 9090
    context_path: str = "/api/ragent"


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/ragent"
    pool_size: int = 10
    min_idle: int = 5
    connection_timeout: int = 5000
    idle_timeout: int = 600000
    max_lifetime: int = 1800000


class RedisSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 6379
    password: str = "123456"
    db: int = 0

    @property
    def dsn(self) -> str:
        pwd = f":{self.password}@" if self.password else ""
        return f"redis://{pwd}{self.host}:{self.port}/{self.db}"


class EngineSettings(BaseSettings):
    type: str = "agent"  # "rag" | "agent"


class AgentChatSettings(BaseSettings):
    provider: str = "siliconflow"
    model: str = "deepseek-ai/DeepSeek-V4-Pro"


class AgentMemorySettings(BaseSettings):
    enabled: bool = True
    context_window_chars: int = 1200000
    summary_enabled: bool = True
    evictable_tools: list[str] = Field(default_factory=lambda: ["search_knowledge"])


class AgentSettings(BaseSettings):
    chat: AgentChatSettings = Field(default_factory=AgentChatSettings)
    max_iters: int = 10
    max_retries: int = 2
    sse_timeout_ms: int = 900000
    memory: AgentMemorySettings = Field(default_factory=AgentMemorySettings)


class StorageS3Settings(BaseSettings):
    endpoint: str = "http://localhost:9000"
    access_key: str = "rustfsadmin"
    secret_key: str = "rustfsadmin"
    region: str = "us-east-1"
    path_style: bool = True
    public_url: str = ""


class StorageOssSettings(BaseSettings):
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "cn-hangzhou"
    public_url: str = ""


class RagStorageSettings(BaseSettings):
    type: str = "s3"  # "s3" | "oss"
    kb_bucket: str = "ragent-sources"
    asset_bucket: str = "ragent-assets"
    s3: StorageS3Settings = Field(default_factory=StorageS3Settings)
    oss: StorageOssSettings = Field(default_factory=StorageOssSettings)


class VectorSettings(BaseSettings):
    type: str = "pg"  # "pg" | "milvus"


class KeywordSettings(BaseSettings):
    type: str = "none"  # "none" | "es"
    es_uris: str = "http://127.0.0.1:9200"
    es_index: str = "rag_keyword_store"
    analyzer: str = "ik_max_word"
    search_analyzer: str = "ik_smart"


class GraphSettings(BaseSettings):
    type: str = "none"  # "none" | "lightrag"
    lightrag_base_url: str = "http://127.0.0.1:9621"
    lightrag_query_mode: str = "hybrid"
    embedding_model: str = "qwen-emb-8b"


class RagDefaultSettings(BaseSettings):
    collection_name: str = "rag_default_store"
    dimension: int = 1536
    metric_type: str = "COSINE"
    sse_timeout_ms: int = 300000
    message_chunk_size: int = 1


class RateLimitGlobalSettings(BaseSettings):
    enabled: bool = True
    max_concurrent: int = 10
    max_wait_seconds: int = 15
    lease_seconds: int = 30
    poll_interval_ms: int = 200


class MemorySettings(BaseSettings):
    history_keep_turns: int = 8
    summary_enabled: bool = True
    summary_start_turns: int = 9
    summary_max_chars: int = 400
    title_max_length: int = 30


class SearchScopeSettings(BaseSettings):
    min_intent_score: float = 0.4
    confidence_threshold: float = 0.6
    supplement_ratio: float = 0.25


class SearchChannelVectorSettings(BaseSettings):
    enabled: bool = True


class SearchChannelKeywordSettings(BaseSettings):
    enabled: bool = False


class SearchChannelGraphSettings(BaseSettings):
    enabled: bool = False


class SearchChannelWebSettings(BaseSettings):
    enabled: bool = False
    count: int = 5
    timeout_seconds: int = 10
    api_key: str = ""


class SearchChannelsSettings(BaseSettings):
    timeout_ms: int = 15000
    vector: SearchChannelVectorSettings = Field(default_factory=SearchChannelVectorSettings)
    keyword: SearchChannelKeywordSettings = Field(default_factory=SearchChannelKeywordSettings)
    graph: SearchChannelGraphSettings = Field(default_factory=SearchChannelGraphSettings)
    web_search: SearchChannelWebSettings = Field(default_factory=SearchChannelWebSettings)


class FusionSettings(BaseSettings):
    strategy: str = "rrf"
    rrf_k: int = 20
    rerank_candidate_limit: int = 40
    channel_weights: dict[str, float] = Field(default_factory=lambda: {
        "vector": 1.0, "keyword": 1.0, "graph": 0.8, "web-search": 0.5,
    })


class EvidenceSettings(BaseSettings):
    min_rerank_score: float = 0.2


class SearchSettings(BaseSettings):
    default_top_k: int = 10
    recall_budget: int = 20
    scope: SearchScopeSettings = Field(default_factory=SearchScopeSettings)
    channels: SearchChannelsSettings = Field(default_factory=SearchChannelsSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)


class TraceSettings(BaseSettings):
    enabled: bool = True
    max_error_length: int = 1000


class McpServerEntry(BaseSettings):
    name: str = "default"
    url: str = "http://localhost:9099"


class RagMcpSettings(BaseSettings):
    servers: list[McpServerEntry] = Field(default_factory=lambda: [McpServerEntry()])
    connect_timeout_seconds: float = Field(
        default=10.0,
        alias="connect_timeout_seconds",
        description="单个 MCP Server 连接+工具发现的总超时；连不上只跳过不阻塞启动",
    )


class RagSettings(BaseSettings):
    storage: RagStorageSettings = Field(default_factory=RagStorageSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    keyword: KeywordSettings = Field(default_factory=KeywordSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    default: RagDefaultSettings = Field(default_factory=RagDefaultSettings)
    query_rewrite_enabled: bool = True
    rerank_enabled: bool = True
    citation_enabled: bool = True
    rate_limit_global: RateLimitGlobalSettings = Field(
        default_factory=RateLimitGlobalSettings,
        alias="rate_limit_global",
    )
    memory: MemorySettings = Field(default_factory=MemorySettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)
    mcp: RagMcpSettings = Field(default_factory=RagMcpSettings)


class ProviderEndpoints(BaseSettings):
    chat: str = "/v1/chat/completions"
    embedding: str = "/v1/embeddings"
    rerank: str = ""


class ProviderConfig(BaseSettings):
    url: str = ""
    api_key: str = ""
    endpoints: ProviderEndpoints = Field(default_factory=ProviderEndpoints)


class AIProviderSettings(BaseSettings):
    ollama: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        url="http://localhost:11434",
        endpoints=ProviderEndpoints(chat="/v1/chat/completions", embedding="/v1/embeddings"),
    ))
    bailian: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        url="https://dashscope.aliyuncs.com",
        endpoints=ProviderEndpoints(
            chat="/compatible-mode/v1/chat/completions",
            rerank="/api/v1/services/rerank/text-rerank/text-rerank",
            embedding="/compatible-mode/v1/embeddings",
        ),
    ))
    aihubmix: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        url="https://aihubmix.com",
        endpoints=ProviderEndpoints(chat="/v1/chat/completions", embedding="/v1/embeddings"),
    ))
    siliconflow: ProviderConfig = Field(default_factory=lambda: ProviderConfig(
        url="https://api.siliconflow.cn",
        endpoints=ProviderEndpoints(chat="/v1/chat/completions", embedding="/v1/embeddings"),
    ))


class ModelCandidate(BaseSettings):
    id: str
    provider: str
    model: str
    dimension: int | None = None
    priority: int = 1
    supports_thinking: bool = False


class TierConfig(BaseSettings):
    candidates: list[str] = Field(default_factory=list)
    timeout_ms: int = 30000


class ChatSettings(BaseSettings):
    candidates: list[ModelCandidate] = Field(default_factory=list)
    default_tier: str = "standard"
    deep_thinking_tier: str = "deep"
    tiers: dict[str, TierConfig] = Field(default_factory=dict)


class EmbeddingSettings(BaseSettings):
    default_model: str = "qwen-emb-8b"
    candidates: list[ModelCandidate] = Field(default_factory=list)


class RerankSettings(BaseSettings):
    default_model: str = "qwen3-rerank"
    candidates: list[ModelCandidate] = Field(default_factory=list)


class VlmSettings(BaseSettings):
    default_model: str = "qwen-vl-max"
    candidates: list[ModelCandidate] = Field(default_factory=list)


class SelectionSettings(BaseSettings):
    failure_threshold: int = 2
    open_duration_ms: int = 30000


class StreamSettings(BaseSettings):
    message_chunk_size: int = 1


class AISettings(BaseSettings):
    providers: AIProviderSettings = Field(default_factory=AIProviderSettings)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    vlm: VlmSettings = Field(default_factory=VlmSettings)


class MineruSettings(BaseSettings):
    api_url: str = "https://mineru.net/api/v4"
    api_key: str = ""
    poll_interval_seconds: int = 5
    timeout_seconds: int = 300
    enable_table: bool = True
    enable_formula: bool = True
    ocr: bool = False
    language: str = "ch"
    concurrency_limit: int = 5
    max_wait_seconds: int = 30
    lease_seconds: int = 900


class UploadSettings(BaseSettings):
    max_file_size: str = "50MB"
    max_request_size: str = "100MB"


# ---------------------------------------------------------------------------
# Root settings — single entry point
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Root configuration, mirrors the full application.yaml hierarchy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    server: ServerSettings = Field(default_factory=ServerSettings)
    api_prefix: str = "/api/ragent"

    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Redis
    redis: RedisSettings = Field(default_factory=RedisSettings)

    # Engine
    engine: EngineSettings = Field(default_factory=EngineSettings)
    demo_mode: bool = False

    # Token
    token_timeout: int = 2592000  # seconds (30 days)

    # Agent
    agent: AgentSettings = Field(default_factory=AgentSettings)

    # RAG
    rag: RagSettings = Field(default_factory=RagSettings)

    # AI
    ai: AISettings = Field(default_factory=AISettings)

    # MinerU
    mineru: MineruSettings = Field(default_factory=MineruSettings)

    # Upload
    upload: UploadSettings = Field(default_factory=UploadSettings)


# Singleton
settings = Settings()
