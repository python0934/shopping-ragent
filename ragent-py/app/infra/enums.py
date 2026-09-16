"""
AI model enums — ModelProvider, Tier, ModelCapability.

Mirrors Java infra enums:
  - enums.ModelProvider
  - enums.Tier
  - enums.ModelCapability
"""

from __future__ import annotations

from enum import Enum


class ModelProvider(str, Enum):
    """模型提供商枚举"""

    OLLAMA = "ollama"
    BAI_LIAN = "bailian"
    SILICON_FLOW = "siliconflow"
    AI_HUB_MIX = "aihubmix"
    NOOP = "noop"

    def matches(self, provider: str | None) -> bool:
        return provider is not None and provider.lower() == self.value


class Tier(str, Enum):
    """
    模型档位枚举 — 表达「质量 / 成本 / 时延预算」

    每个枚举值的 key 对应 application.yaml 中 ai.chat.tiers 下的档位键。
    """

    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


class ModelCapability(str, Enum):
    """模型能力枚举"""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    VLM = "vlm"

    @property
    def display_name(self) -> str:
        return {
            ModelCapability.CHAT: "大模型对话",
            ModelCapability.EMBEDDING: "向量嵌入",
            ModelCapability.RERANK: "重排序",
            ModelCapability.VLM: "视觉语言模型",
        }[self]
