"""
VLM (Vision Language Model) client interfaces — VlmService, RoutingVlmService.

Mirrors Java infra vlm classes:
  - vlm.VlmService
  - vlm.RoutingVlmService
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.infra.enums import ModelCapability
from app.infra.http_utils import (
    HttpMediaTypes,
    HttpResponseHelper,
    ModelClientErrorType,
    ModelClientException,
    ModelUrlResolver,
)
from app.infra.model_routing import ModelSelector, ModelTarget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VlmService — interface
# ---------------------------------------------------------------------------

class VlmService(ABC):
    """
    视觉大模型（VLM）访问接口。

    与 LLMService / EmbeddingService / RerankService 同级的第四类模型能力。
    当前唯一用途是知识库入库期的「图生文」：把图片转成可检索的中文描述 + 图中文字 OCR。
    下游问答仍为纯文本模型，VLM 只在写入侧调用，不进入 chat 热路径。

    Mirrors Java vlm.VlmService.
    """

    @abstractmethod
    def describe_image(
        self,
        image_bytes: bytes,
        mime: str,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        图生文：输入图片字节，返回模型生成的文本（中文描述 + 图中文字）。

        Args:
            image_bytes: 图片二进制
            mime: 图片 MIME，如 image/png、image/jpeg
            prompt: 引导提示词
            max_output_tokens: 输出 token 上限，可空（控成本）

        Returns:
            模型返回的描述文本
        """
        ...


# ---------------------------------------------------------------------------
# RoutingVlmService — routing implementation
# ---------------------------------------------------------------------------

class RoutingVlmService(VlmService):
    """
    路由式 VLM 服务实现类。

    复用 chat 链路同款基础设施：provider 配置、URL 解析、错误体系。
    与 chat 的唯一差异是请求体 messages[].content 为多模态数组（text + image_url）。
    入库期单次同步调用，从 ai.vlm 组取首个可用候选即可，无需 fallback。

    Mirrors Java vlm.RoutingVlmService.
    """

    LABEL = "vlm"

    def __init__(self, selector: ModelSelector, http_client: Any | None = None) -> None:
        self.selector = selector
        self.http_client = http_client

    def describe_image(
        self,
        image_bytes: bytes,
        mime: str,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> str:
        target = self._resolve_target()
        provider = HttpResponseHelper.require_provider(target, self.LABEL)
        HttpResponseHelper.require_api_key(provider, self.LABEL)

        url = ModelUrlResolver.resolve_url(provider, target.candidate, ModelCapability.CHAT)
        req_body = self._build_multimodal_body(target, prompt, image_bytes, mime, max_output_tokens)

        # If http_client is provided (e.g. httpx.AsyncClient), use it
        if self.http_client is not None:
            return self._execute_request(url, provider, req_body)

        raise ModelClientException(
            "VLM HTTP 客户端未配置",
            ModelClientErrorType.PROVIDER_ERROR,
        )

    def _resolve_target(self) -> ModelTarget:
        targets = self.selector.select_vlm_candidates()
        if not targets:
            raise ModelClientException(
                "VLM 模型不可用，请检查 ai.vlm 配置",
                ModelClientErrorType.PROVIDER_ERROR,
            )
        return targets[0]

    def _build_multimodal_body(
        self,
        target: ModelTarget,
        prompt: str,
        image: bytes,
        mime: str,
        max_output_tokens: int | None,
    ) -> dict:
        """构造多模态请求体：content 为数组，图片以 base64 data url 内联"""
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"

        text_part = {"type": "text", "text": prompt}
        image_part = {"type": "image_url", "image_url": {"url": data_url}}
        user_msg = {"role": "user", "content": [text_part, image_part]}

        model_name = HttpResponseHelper.require_model(target, self.LABEL)
        body: dict = {"model": model_name, "messages": [user_msg]}
        if max_output_tokens and max_output_tokens > 0:
            body["max_tokens"] = max_output_tokens
        return body

    def _execute_request(self, url: str, provider: dict, req_body: dict) -> str:
        """执行 HTTP 请求并提取响应内容（同步实现）"""
        import httpx

        api_key = HttpResponseHelper.require_api_key(provider, self.LABEL)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": HttpMediaTypes.JSON,
        }

        try:
            resp = httpx.post(url, json=req_body, headers=headers, timeout=60.0)
        except Exception as e:
            raise ModelClientException(
                f"VLM 请求失败: {e}",
                ModelClientErrorType.NETWORK_ERROR,
                cause=e,
            )

        if resp.status_code != 200:
            logger.warning("VLM 请求失败: status=%d, body=%s", resp.status_code, resp.text)
            raise ModelClientException(
                f"VLM 请求失败: HTTP {resp.status_code}",
                ModelClientErrorType.from_http_status(resp.status_code),
                resp.status_code,
            )

        resp_json = HttpResponseHelper.parse_json(resp.text, self.LABEL)
        return self._extract_content(resp_json)

    @staticmethod
    def _extract_content(resp_json: dict) -> str:
        """抽取 OpenAI 兼容响应的 choices[0].message.content"""
        if not resp_json or "choices" not in resp_json:
            raise ModelClientException("VLM 响应缺少 choices", ModelClientErrorType.INVALID_RESPONSE)
        choices = resp_json["choices"]
        if not choices:
            raise ModelClientException("VLM 响应 choices 为空", ModelClientErrorType.INVALID_RESPONSE)
        choice0 = choices[0]
        if not isinstance(choice0, dict) or "message" not in choice0:
            raise ModelClientException("VLM 响应缺少 message", ModelClientErrorType.INVALID_RESPONSE)
        message = choice0["message"]
        if not isinstance(message, dict) or "content" not in message or message["content"] is None:
            raise ModelClientException("VLM 响应缺少 content", ModelClientErrorType.INVALID_RESPONSE)
        return message["content"]

