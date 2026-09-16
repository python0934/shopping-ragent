"""
Agent schemas — request/response models for Agent chat, conversations, meta.

Mirrors Java:
  - agent.controller.vo.AgentConversationVO
  - agent.controller.vo.AgentMessageVO
  - agent.controller.vo.AgentMetaVO
  - agent.dto.AgentBlock / AgentMetaPayload / AgentMessageDelta /
    AgentToolProgress / AgentCompletionPayload / AgentHintPayload
  - agent.controller.AgentConversationController.TitleRequest / BatchDeleteRequest

Field names are camelCase on purpose: the frontend contract must not change.
Payload records annotated @JsonInclude(NON_NULL) in Java serialise with
``model_dump(exclude_none=True)`` here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# AgentBlock — 运行轨迹块（reasoning / answer / tool）
# ---------------------------------------------------------------------------

class AgentBlock(BaseModel):
    """
    运行轨迹块，回放还原时间线。

    Java 侧标注 @JsonInclude(NON_NULL)，空字段不参与序列化。
    """

    model_config = ConfigDict(populate_by_name=True)

    kind: str | None = Field(default=None, description="reasoning / answer / tool")
    at: str | None = Field(default=None, description="产生时刻 yyyy-MM-dd'T'HH:mm:ss")
    text: str | None = Field(default=None, description="reasoning / answer 的正文")
    name: str | None = Field(default=None, description="tool 名")
    displayName: str | None = Field(default=None, description="tool 展示名")
    status: str | None = Field(default=None, description="tool 终态 done / failed / interrupted")
    result: str | None = Field(default=None, description="tool 结果文本，超长截断")
    toolCallId: str | None = Field(default=None, description="供应商侧 tool_call id")

    def to_wire(self) -> dict[str, Any]:
        """序列化为 SSE / JSONB 落库形态，剔除 null 字段。"""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Agent Conversation VO
# ---------------------------------------------------------------------------

class AgentConversationVO(BaseModel):
    """Agent 会话列表项"""

    conversationId: str = Field(..., description="会话ID")
    title: str | None = Field(default=None, description="会话标题")
    lastTime: datetime | None = Field(default=None, description="最后活动时间")
    turns: int | None = Field(default=None, description="已进行轮数（用户提问计数）")


# ---------------------------------------------------------------------------
# Agent Message VO
# ---------------------------------------------------------------------------

class AgentMessageVO(BaseModel):
    """Agent 消息回放项"""

    id: str = Field(..., description="消息ID")
    role: str | None = Field(default=None, description="user / assistant")
    content: str | None = Field(default=None, description="消息正文")
    thinkingContent: str | None = Field(default=None, description="思考内容")
    blocks: list[AgentBlock] | None = Field(
        default=None,
        description="运行轨迹块；旧数据为 null 时由前端按 content/thinking 合成",
    )
    messageStatus: str | None = Field(default=None, description="NORMAL / INTERRUPTED")
    createTime: datetime | None = Field(default=None, description="创建时间")


# ---------------------------------------------------------------------------
# Agent Meta VO — /agent/v1/meta 探活身份
# ---------------------------------------------------------------------------

class AgentMetaVO(BaseModel):
    """Agent 引擎探活与身份，绝不带密钥"""

    framework: str = Field(..., description="执行框架标识")
    model: str | None = Field(default=None, description="当前对话模型名")
    maxIters: int | None = Field(default=None, description="ReAct 循环上限")
    capabilities: list[str] = Field(default_factory=list, description="能力清单，随实况增删")
    toolProvider: str | None = Field(default=None, description="native / native + mcp")
    mcpConfigured: bool = Field(default=False, description="意图树是否已挂载可用 MCP 工具")


# ---------------------------------------------------------------------------
# SSE payloads — mirrors agent.dto records
# ---------------------------------------------------------------------------

class AgentMetaPayload(BaseModel):
    """meta 事件：会话与任务元信息"""

    conversationId: str
    taskId: str


class AgentMessageDelta(BaseModel):
    """message 事件：增量正文 / 思考"""

    type: str = Field(..., description="response / think")
    delta: str


class AgentToolProgress(BaseModel):
    """tool 事件：工具进度"""

    name: str | None = None
    displayName: str | None = None
    status: str | None = Field(default=None, description="start / end")
    result: str | None = None
    ok: bool | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AgentCompletionPayload(BaseModel):
    """finish / cancel 事件：回复终态"""

    messageId: str | None = None
    title: str | None = None
    messageStatus: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AgentHintPayload(BaseModel):
    """hint 事件：运行提示，不落库"""

    code: str
    text: str


# ---------------------------------------------------------------------------
# Request bodies — mirrors AgentConversationController records
# ---------------------------------------------------------------------------

class AgentTitleRequest(BaseModel):
    """重命名会话请求体"""

    title: str | None = None


class AgentBatchDeleteRequest(BaseModel):
    """批量删除会话请求体"""

    ids: list[str] | None = None
