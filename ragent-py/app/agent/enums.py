"""
Agent enums — mirrors agent.enums.AgentMessageStatus / AgentSSEEventType.
"""

from __future__ import annotations

from enum import Enum


class AgentMessageStatus(str, Enum):
    """消息终态"""

    NORMAL = "NORMAL"
    """正常完成"""

    INTERRUPTED = "INTERRUPTED"
    """用户中断，内容为已生成的部分"""


class AgentSSEEventType(str, Enum):
    """
    Agent SSE 事件类型。

    ``value`` 即前端 ``EventSource`` 监听的事件名，改名等于改契约。
    """

    META = "meta"
    """会话与任务元信息"""

    MESSAGE = "message"
    """增量消息（response / think）"""

    TOOL = "tool"
    """工具进度 {name, displayName, status: start|end, result, ok}"""

    HINT = "hint"
    """运行提示（如达到迭代上限的熔断预告），不落库"""

    FINISH = "finish"
    """回复完成"""

    DONE = "done"
    """流结束"""

    CANCEL = "cancel"
    """用户取消"""


class AgentEventType(str, Enum):
    """
    ReAct 内部事件类型 — 对应 AgentScope 的 AgentEvent.Type。

    只在进程内流转，不出网；SSE 侧的映射由 AgentStreamEventBridge 负责。
    """

    TEXT_BLOCK_DELTA = "TEXT_BLOCK_DELTA"
    THINKING_BLOCK_DELTA = "THINKING_BLOCK_DELTA"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_RESULT_TEXT_DELTA = "TOOL_RESULT_TEXT_DELTA"
    TOOL_RESULT_END = "TOOL_RESULT_END"
    HINT_BLOCK = "HINT_BLOCK"
    EXCEED_MAX_ITERS = "EXCEED_MAX_ITERS"
    AGENT_RESULT = "AGENT_RESULT"


class ToolResultState(str, Enum):
    """工具执行结果状态"""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class MsgRole(str, Enum):
    """消息角色 — 对应 AgentScope MsgRole"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
