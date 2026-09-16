"""
意图节点注册表 — mirrors rag.core.intent.IntentNodeRegistry / DefaultIntentClassifier
的树加载部分。

Agent 工具目录要在请求路径上同步读「当前启用的 MCP 叶子节点」，而 Python 侧的
DB 访问是异步的；因此这里维护一份内存快照，由 ``refresh()`` 在启动和意图树变更后
异步重建，读侧保持同步，与 Java 走 Redis 缓存的效果一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag import IntentNodeDO

logger = logging.getLogger(__name__)

INTENT_KIND_MCP = 2
"""kind 取值：0 RAG / 1 SYSTEM / 2 MCP"""


@dataclass(frozen=True)
class IntentNode:
    """运行期意图节点 — mirrors rag.core.intent.IntentNode（只留 Agent 侧要用的字段）"""

    id: str
    """intent_code，树内主键"""
    parent_id: str | None = None
    name: str = ""
    description: str | None = None
    level: int = 0
    kind: int = 0
    mcp_tool_id: str | None = None
    param_prompt_template: str | None = None
    children: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_mcp(self) -> bool:
        return self.kind == INTENT_KIND_MCP

    def to_dict(self) -> dict[str, Any]:
        """AgentToolCatalog 按 mcpToolId / name / description 三个键读取"""
        return {
            "id": self.id,
            "parentId": self.parent_id,
            "name": self.name,
            "description": self.description,
            "level": self.level,
            "kind": self.kind,
            "mcpToolId": self.mcp_tool_id,
            "paramPromptTemplate": self.param_prompt_template,
        }


@dataclass
class DbIntentNodeRegistry:
    """DB 支撑的意图节点注册表，读侧同步、写侧异步刷新"""

    _nodes: dict[str, IntentNode] = field(default_factory=dict, init=False, repr=False)

    async def refresh(self, db: AsyncSession) -> int:
        """
        重建快照，返回节点总数。

        过滤条件与 Java ``loadIntentTreeFromDB`` 一致：deleted = 0 且 enabled = 1。
        """
        rows = (await db.execute(
            select(IntentNodeDO).where(
                IntentNodeDO.deleted == 0,
                IntentNodeDO.enabled == 1,
            )
        )).scalars().all()

        nodes: dict[str, IntentNode] = {}
        for row in rows:
            code = (row.intent_code or "").strip()
            if not code:
                continue
            nodes[code] = IntentNode(
                id=code,
                parent_id=(row.parent_code or "").strip() or None,
                name=row.name or "",
                description=row.description,
                level=int(row.level or 0),
                kind=int(row.kind or 0),
                mcp_tool_id=row.mcp_tool_id,
                param_prompt_template=row.param_prompt_template,
            )

        # 父子挂接：父节点不在快照里（被停用/删除）就当根，与 Java 组树行为一致
        children: dict[str, list[str]] = {}
        for node in nodes.values():
            parent_id = node.parent_id
            if parent_id and parent_id in nodes and parent_id != node.id:
                children.setdefault(parent_id, []).append(node.id)

        self._nodes = {
            code: replace(node, children=tuple(sorted(children.get(code, ()))))
            for code, node in nodes.items()
        }
        logger.debug("意图树快照重建完成, 总节点数: %d", len(self._nodes))
        return len(self._nodes)

    def get_node_by_id(self, node_id: str | None) -> IntentNode | None:
        if not node_id or not node_id.strip():
            return None
        return self._nodes.get(node_id.strip())

    def list_mcp_tool_nodes(self) -> list[dict[str, Any]]:
        """当前已启用、可参与路由的 MCP 叶子节点，按节点 ID 升序"""
        return [
            node.to_dict()
            for node in sorted(self._nodes.values(), key=lambda n: n.id)
            if node.is_leaf and node.is_mcp and (node.mcp_tool_id or "").strip()
        ]

    def list_all_nodes(self) -> list[IntentNode]:
        return list(self._nodes.values())

    def clear(self) -> None:
        self._nodes.clear()


intent_node_registry = DbIntentNodeRegistry()
"""进程级单例：Agent 工具目录与 meta 探活共用同一份快照"""
