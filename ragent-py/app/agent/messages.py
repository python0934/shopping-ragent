"""
Agent message model — mirrors AgentScope ``Msg`` / ``ContentBlock`` hierarchy.

AgentScope 的 Java SDK 无 Python 对等物，这里按同一套块语义自建：
一条 ``Msg`` 由若干有序 ``ContentBlock`` 组成，块分四类
（text / thinking / tool_use / tool_result），与 OpenAI 兼容端点的
``content`` / ``reasoning_content`` / ``tool_calls`` / ``role=tool``
四种载荷一一对应。

所有块都可 JSON 往返，状态存储（t_agent_state.payload）直接序列化整棵 Msg。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from app.agent.enums import MsgRole, ToolResultState
from app.core.snowflake import get_snowflake_id_str

T = TypeVar("T", bound="ContentBlock")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """正文块"""

    text: str = ""
    type: str = field(default="text", init=False)


@dataclass
class ThinkingBlock:
    """深度思考块"""

    thinking: str = ""
    type: str = field(default="thinking", init=False)


@dataclass
class ToolUseBlock:
    """工具调用块 — 模型请求调用某个工具"""

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="tool_use", init=False)


@dataclass
class ToolResultBlock:
    """工具结果块 — 工具执行回执"""

    id: str = ""
    name: str | None = None
    output: list[ContentBlock] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    state: ToolResultState = ToolResultState.SUCCESS
    type: str = field(default="tool_result", init=False)

    def output_text(self) -> str:
        """拼接输出里的全部文本块"""
        return "\n".join(
            b.text for b in self.output if isinstance(b, TextBlock) and b.text
        )


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock

_BLOCK_TYPES: dict[str, type] = {
    "text": TextBlock,
    "thinking": ThinkingBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
}


def block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """块 -> JSON dict（递归处理 tool_result.output）"""
    if isinstance(block, ToolResultBlock):
        data: dict[str, Any] = {
            "type": "tool_result",
            "id": block.id,
            "name": block.name,
            "output": [block_to_dict(b) for b in block.output],
            "metadata": block.metadata,
            "state": block.state.value if isinstance(block.state, ToolResultState) else block.state,
        }
        return data
    if isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    return {"type": "text", "text": getattr(block, "text", "")}


def block_from_dict(data: dict[str, Any]) -> ContentBlock | None:
    """JSON dict -> 块，未知类型返回 None"""
    if not isinstance(data, dict):
        return None
    kind = data.get("type", "text")
    if kind == "tool_result":
        state_raw = data.get("state") or ToolResultState.SUCCESS.value
        try:
            state = ToolResultState(state_raw)
        except ValueError:
            state = ToolResultState.SUCCESS
        return ToolResultBlock(
            id=data.get("id") or "",
            name=data.get("name"),
            output=[b for b in (block_from_dict(o) for o in (data.get("output") or [])) if b is not None],
            metadata=data.get("metadata"),
            state=state,
        )
    if kind == "tool_use":
        return ToolUseBlock(
            id=data.get("id") or "",
            name=data.get("name") or "",
            input=data.get("input") or {},
        )
    if kind == "thinking":
        return ThinkingBlock(thinking=data.get("thinking") or "")
    if kind == "text":
        return TextBlock(text=data.get("text") or "")
    return None


# ---------------------------------------------------------------------------
# Msg
# ---------------------------------------------------------------------------

@dataclass
class Msg:
    """
    一条会话消息。

    ``with_content`` 返回新对象而非就地改，裁剪器据此建立 origin -> replaced
    的替换映射（对应 Java 的 IdentityHashMap）。

    ``name`` 承载压缩摘要的哨兵标记（__compaction_summary__），
    ``timestamp`` 供摘要笔录渲染时刻，两者都对应 AgentScope Msg 的同名字段。
    """

    role: MsgRole
    content: list[ContentBlock] = field(default_factory=list)
    id: str = field(default_factory=get_snowflake_id_str)
    name: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] | None = None

    # -- 构造便捷方法 -------------------------------------------------------

    @staticmethod
    def system(text: str) -> Msg:
        return Msg(role=MsgRole.SYSTEM, content=[TextBlock(text=text)])

    @staticmethod
    def user(text: str) -> Msg:
        return Msg(role=MsgRole.USER, content=[TextBlock(text=text)])

    @staticmethod
    def assistant(text: str = "", thinking: str = "") -> Msg:
        blocks: list[ContentBlock] = []
        if thinking:
            blocks.append(ThinkingBlock(thinking=thinking))
        if text:
            blocks.append(TextBlock(text=text))
        return Msg(role=MsgRole.ASSISTANT, content=blocks)

    @staticmethod
    def tool_result(result: ToolResultBlock) -> Msg:
        return Msg(role=MsgRole.TOOL, content=[result])

    # -- 访问器 -------------------------------------------------------------

    def get_content_blocks(self, block_type: type[T]) -> list[T]:
        """按类型取块，content 为空返回空列表"""
        if not self.content:
            return []
        return [b for b in self.content if isinstance(b, block_type)]

    def get_text_content(self) -> str:
        """拼接全部正文块"""
        return "".join(b.text for b in self.get_content_blocks(TextBlock))

    def get_thinking_content(self) -> str:
        """拼接全部思考块"""
        return "".join(b.thinking for b in self.get_content_blocks(ThinkingBlock))

    def with_content(self, content: list[ContentBlock]) -> Msg:
        """复制一份并换掉 content，id / role / name / timestamp / metadata 原样带走"""
        return Msg(
            role=self.role,
            content=list(content),
            id=self.id,
            name=self.name,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value if isinstance(self.role, MsgRole) else self.role,
            "name": self.name,
            "timestamp": self.timestamp,
            "content": [block_to_dict(b) for b in self.content],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Msg:
        role_raw = data.get("role") or MsgRole.USER.value
        try:
            role = MsgRole(role_raw)
        except ValueError:
            role = MsgRole.USER
        return Msg(
            role=role,
            content=[b for b in (block_from_dict(c) for c in (data.get("content") or [])) if b is not None],
            id=data.get("id") or get_snowflake_id_str(),
            name=data.get("name"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Context chars — mirrors memory.AgentContextChars
# ---------------------------------------------------------------------------

class AgentContextChars:
    """
    上下文体量口径：字符数作为 token 的粗代理，非文本块按零计。
    """

    @staticmethod
    def total(context: list[Msg] | None) -> int:
        if not context:
            return 0
        return sum(AgentContextChars.of_msg(msg) for msg in context)

    @staticmethod
    def of_msg(msg: Msg | None) -> int:
        if msg is None or not msg.content:
            return 0
        return sum(AgentContextChars.of_block(block) for block in msg.content)

    @staticmethod
    def of_block(block: ContentBlock) -> int:
        if isinstance(block, TextBlock):
            return len(block.text or "")
        if isinstance(block, ThinkingBlock):
            return len(block.thinking or "")
        if isinstance(block, ToolUseBlock):
            return len(block.name or "") + len(str(block.input) if block.input is not None else "")
        if isinstance(block, ToolResultBlock):
            return AgentContextChars.of_output(block)
        return 0

    @staticmethod
    def of_output(block: ToolResultBlock) -> int:
        if not block.output:
            return 0
        return sum(AgentContextChars.of_block(nested) for nested in block.output)
