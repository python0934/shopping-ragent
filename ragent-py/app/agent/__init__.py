"""
Agent execution engine — native ReAct loop, tool registry, memory, state store.

Mirrors Java package ``com.nageoffer.ai.ragent.agent``:

  ==========================  ==========================================
  Java                        Python
  ==========================  ==========================================
  enums.AgentMessageStatus    enums.AgentMessageStatus
  enums.AgentSSEEventType     enums.AgentSSEEventType
  memory.AgentContextChars    messages.context_chars
  memory.AgentContextTrimmer  memory.AgentContextTrimmer
  memory.AgentMemoryProperties memory.AgentMemoryBudget
  state.PgAgentStateStore     state_store.PgAgentStateStore
  service.handler.AgentRunGate run_gate.AgentRunGate
  tool.AgentToolCatalog       tools.AgentToolCatalog
  tool.KnowledgeSearchTool    tools.KnowledgeSearchTool
  tool.McpToolBridge          tools.McpToolBridge
  config.ReActAgentProvider   react_agent.ReActAgentProvider
  (AgentScope ReActAgent)     react_agent.ReActAgent
  service.handler.AgentStreamEventBridge event_bridge.AgentStreamEventBridge
  ==========================  ==========================================

AgentScope 的 Java SDK 在 Python 侧不存在对等物，ReAct 循环按同一份事件协议
（TEXT_BLOCK_DELTA / TOOL_CALL_START / TOOL_RESULT_END / AGENT_RESULT ...）
自实现，事件名与 SSE 帧结构对前端保持不变。
"""

from app.agent.enums import AgentMessageStatus, AgentSSEEventType

__all__ = [
    "AgentMessageStatus",
    "AgentSSEEventType",
]
