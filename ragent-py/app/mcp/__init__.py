"""
MCP 客户端 — mirrors rag.core.mcp.*.

  ==================================  ==============================
  Java (rag.core.mcp)                本包
  ==================================  ==============================
  McpToolRegistry                    registry.McpToolRegistry
  DefaultMcpToolRegistry             registry.McpToolRegistry
  McpToolExecutor                    registry.McpClientToolExecutor
  McpClientToolExecutor              registry.McpClientToolExecutor
  McpClientProperties                config.settings.rag.mcp
  McpClientAutoConfiguration         registry.McpClientManager
  ==================================  ==============================

Java 用官方 ``io.modelcontextprotocol`` 同步 SDK 在 ``@PostConstruct`` 里连完所有
Server；这里换成 ``mcp`` 异步 SDK，连接挪到 FastAPI lifespan 里做，语义一致：
连不上的 Server 只记日志跳过，不阻塞启动。
"""

from app.mcp.registry import (
    McpClientManager,
    McpClientToolExecutor,
    McpToolRegistry,
    mcp_client_manager,
    mcp_tool_registry,
)

__all__ = [
    "McpClientManager",
    "McpClientToolExecutor",
    "McpToolRegistry",
    "mcp_client_manager",
    "mcp_tool_registry",
]
