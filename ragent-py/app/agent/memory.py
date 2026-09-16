"""
Agent 会话记忆 — 预算派生 + 上下文裁剪。

Mirrors Java:
  - memory.AgentMemoryProperties  -> AgentMemoryBudget
  - memory.AgentContextTrimmer    -> AgentContextTrimmer

两道门按固定比例从 contextWindowChars 派生：过半触发裁剪（只把老工具结果换成
等长占位），过八成触发压缩（见 compaction 模块）。裁剪不碰 IO 也不改列表长度，
只替换 tool_result 而不动 tool_use，因此永远不会产生孤儿结果块。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from app.agent.enums import MsgRole
from app.agent.messages import (
    AgentContextChars,
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from app.config import settings

logger = logging.getLogger(__name__)


# ===========================================================================
# AgentMemoryBudget — mirrors AgentMemoryProperties
# ===========================================================================

@dataclass(frozen=True)
class AgentMemoryBudget:
    """
    记忆预算：配置项 + 按比例派生的各道门。

    比例常量在 Java 侧是 private static final，这里同样不开放配置，
    换模型只需要动 ``context_window_chars`` 一个数。
    """

    TRIM_TRIGGER_RATIO: float = 0.5
    """裁剪门：过半即动手，只把老工具结果换成占位"""

    COMPACT_TRIGGER_RATIO: float = 0.8
    """压缩门：裁剪顶不住才动手，余两成给压缩期间的模型调用"""

    KEEP_RECENT_RATIO: float = 0.2
    """保留段：压缩切点之后至少留这么多原文"""

    KEEP_RECENT_CYCLES: int = 2
    """往前保几个已完成的工具循环，本轮和未闭合的额外保护不占配额"""

    CLEAR_AT_LEAST_RATIO: float = 0.2
    """可回收量低于当前上下文的这个比例就不动"""

    SUMMARY_MAX_RATIO: float = 0.1
    SUMMARY_MAX_FLOOR_CHARS: int = 1500
    SUMMARY_MAX_CEIL_CHARS: int = 6000
    """摘要正文上限：0.1×预算 夹进 [1500, 6000]"""

    enabled: bool = True
    context_window_chars: int = 1_200_000
    summary_enabled: bool = True
    evictable_tools: tuple[str, ...] = ("search_knowledge",)

    @property
    def trim_trigger_chars(self) -> int:
        return int(self.context_window_chars * self.TRIM_TRIGGER_RATIO)

    @property
    def compact_trigger_chars(self) -> int:
        return int(self.context_window_chars * self.COMPACT_TRIGGER_RATIO)

    @property
    def keep_recent_chars(self) -> int:
        return int(self.context_window_chars * self.KEEP_RECENT_RATIO)

    @property
    def keep_recent_cycles(self) -> int:
        return self.KEEP_RECENT_CYCLES

    @property
    def clear_at_least_ratio(self) -> float:
        return self.CLEAR_AT_LEAST_RATIO

    @property
    def summary_max_chars(self) -> int:
        """夹在上下限之间返回"""
        derived = int(self.context_window_chars * self.SUMMARY_MAX_RATIO)
        return min(max(derived, self.SUMMARY_MAX_FLOOR_CHARS), self.SUMMARY_MAX_CEIL_CHARS)

    @classmethod
    def from_settings(cls) -> AgentMemoryBudget:
        """从 settings.agent.memory 构造"""
        mem = settings.agent.memory
        return cls(
            enabled=mem.enabled,
            context_window_chars=mem.context_window_chars,
            summary_enabled=mem.summary_enabled,
            evictable_tools=tuple(mem.evictable_tools or ()),
        )


# ===========================================================================
# AgentContextTrimmer — mirrors memory.AgentContextTrimmer
# ===========================================================================

@dataclass(frozen=True)
class TrimResult:
    """
    裁剪结果。

    ``replacements`` 是 (origin, replaced) 有序对，供调用方把同一批 Msg 换进
    本轮上行列表；用元组而非 dict 是为了同时持有两端引用，避免 id 被回收复用。
    """

    reclaimed_chars: int = 0
    replacements: tuple[tuple[Msg, Msg], ...] = ()

    @property
    def changed(self) -> bool:
        return self.reclaimed_chars > 0 and bool(self.replacements)


TRIM_UNCHANGED = TrimResult(0, ())
"""未发生裁剪的共享常量，对应 Java 的 TrimResult.UNCHANGED"""


@dataclass
class _Cycle:
    """一个工具循环：一条带 tool_use 的 assistant 消息 + 其后配对的 tool 消息"""

    start_index: int
    tool_indexes: list[int] = field(default_factory=list)
    pending_ids: set[str] = field(default_factory=set)


@dataclass
class _Candidate:
    """一个可回收的工具结果块"""

    msg_index: int
    block: ToolResultBlock
    origin_chars: int
    reclaimable: int
    input: str | None


class AgentContextTrimmer:
    """
    会话上下文裁剪：把过老的工具结果换成等长占位说明。
    """

    EVICTED_PREFIX = "[历史工具结果已省略，原长 "
    EVICTED_CHARS = " 字符"
    EVICTED_INPUT = "，原入参 "
    EVICTED_SUFFIX = "]"

    EVICTED_INPUT_MAX_CHARS = 120
    """长入参截断，避免占位比原文还大"""

    def __init__(self, budget: AgentMemoryBudget | None = None) -> None:
        self.budget = budget or AgentMemoryBudget.from_settings()

    # -- 入口 ---------------------------------------------------------------

    def trim_in_place(self, context: list[Msg] | None) -> TrimResult:
        """就地裁剪，返回替换映射供调用方同步上行列表"""
        if not context:
            return TRIM_UNCHANGED

        total_chars = AgentContextChars.total(context)
        if total_chars <= self.budget.trim_trigger_chars:
            return TRIM_UNCHANGED

        cycles = self._split_cycles(context)
        protected = self._protected_cycles(context, cycles, self.budget.keep_recent_cycles)
        candidates = self._collect_candidates(context, cycles, protected, self.budget.evictable_tools)

        reclaimable = sum(c.reclaimable for c in candidates)
        # 可回收量不够下限就整次放弃
        clear_at_least = math.ceil(total_chars * self.budget.clear_at_least_ratio)
        if reclaimable < clear_at_least:
            logger.debug(
                "上下文裁剪跳过, 总字符: %s, 可回收: %s, 下限: %s",
                total_chars, reclaimable, clear_at_least,
            )
            return TRIM_UNCHANGED

        replacements = self._apply(context, candidates)
        logger.info(
            "上下文裁剪完成, 总字符: %s -> %s, 命中消息: %s, 工具结果: %s",
            total_chars, total_chars - reclaimable, len(replacements), len(candidates),
        )
        return TrimResult(reclaimable, tuple(replacements))

    # -- 循环切分 -----------------------------------------------------------

    def _split_cycles(self, context: list[Msg]) -> list[_Cycle]:
        """
        按工具循环切分：一条带 tool_use 的 assistant 消息开启一个循环，
        遇到用户消息或纯文本回答即闭合。
        """
        cycles: list[_Cycle] = []
        current: _Cycle | None = None

        for i, msg in enumerate(context):
            if msg.role == MsgRole.TOOL:
                if current is not None:
                    current.tool_indexes.append(i)
                    for block in msg.get_content_blocks(ToolResultBlock):
                        current.pending_ids.discard(block.id)
                continue

            tool_uses = msg.get_content_blocks(ToolUseBlock)
            if msg.role == MsgRole.ASSISTANT and tool_uses:
                current = _Cycle(
                    start_index=i,
                    tool_indexes=[],
                    pending_ids={b.id for b in tool_uses},
                )
                cycles.append(current)
                continue

            current = None

        return cycles

    def _protected_cycles(
        self,
        context: list[Msg],
        cycles: list[_Cycle],
        keep_recent_cycles: int,
    ) -> set[int]:
        """本轮和未闭合的循环额外保护不占配额，keepRecentCycles 只在本轮之前计数"""
        turn_start = self._last_user_index(context)
        result: set[int] = set()
        kept = 0

        for i in range(len(cycles) - 1, -1, -1):
            cycle = cycles[i]
            if cycle.start_index > turn_start or cycle.pending_ids:
                result.add(i)
                continue
            if kept < keep_recent_cycles:
                result.add(i)
                kept += 1

        return result

    @staticmethod
    def _last_user_index(context: list[Msg]) -> int:
        """取不到用户消息返回 -1，全部循环落保护区"""
        for i in range(len(context) - 1, -1, -1):
            if context[i].role == MsgRole.USER:
                return i
        return -1

    # -- 候选收集 -----------------------------------------------------------

    def _collect_candidates(
        self,
        context: list[Msg],
        cycles: list[_Cycle],
        protected: set[int],
        evictable_tools: tuple[str, ...] | list[str],
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []

        for c, cycle in enumerate(cycles):
            if c in protected:
                continue
            inputs = self._tool_inputs(context[cycle.start_index])
            for msg_index in cycle.tool_indexes:
                for block in context[msg_index].get_content_blocks(ToolResultBlock):
                    # 工具名为空即框架级错误结果，跳过
                    if block.name is None or block.name not in evictable_tools:
                        continue
                    if self._is_evicted(block):
                        continue
                    tool_input = inputs.get(block.id)
                    origin_chars = AgentContextChars.of_output(block)
                    reclaimable = origin_chars - len(self._preview(origin_chars, tool_input))
                    if reclaimable > 0:
                        candidates.append(_Candidate(
                            msg_index=msg_index,
                            block=block,
                            origin_chars=origin_chars,
                            reclaimable=reclaimable,
                            input=tool_input,
                        ))

        return candidates

    def _tool_inputs(self, msg: Msg) -> dict[str, str | None]:
        """按 tool_use id 配对取入参，整个打平不按工具名解析"""
        return {
            block.id: self._clip_input(str(block.input) if block.input is not None else "")
            for block in msg.get_content_blocks(ToolUseBlock)
        }

    def _clip_input(self, value: str | None) -> str | None:
        if not value or not value.strip() or value == "null":
            return None
        if len(value) <= self.EVICTED_INPUT_MAX_CHARS:
            return value
        return value[: self.EVICTED_INPUT_MAX_CHARS] + "…"

    # -- 应用 ---------------------------------------------------------------

    def _apply(self, context: list[Msg], candidates: list[_Candidate]) -> list[tuple[Msg, Msg]]:
        """原位替换：先全部重建再统一写回，中途异常不改 context"""
        hit: dict[int, _Candidate] = {}
        touched: set[int] = set()
        for candidate in candidates:
            hit[id(candidate.block)] = candidate
            touched.add(candidate.msg_index)

        staged: dict[int, Msg] = {}
        replacements: list[tuple[Msg, Msg]] = []

        for msg_index in sorted(touched):
            origin = context[msg_index]
            rebuilt = []
            for block in origin.content:
                candidate = hit.get(id(block)) if isinstance(block, ToolResultBlock) else None
                rebuilt.append(self._evict(candidate) if candidate is not None else block)
            replaced = origin.with_content(rebuilt)
            staged[msg_index] = replaced
            replacements.append((origin, replaced))

        for msg_index, replaced in staged.items():
            context[msg_index] = replaced

        return replacements

    def _evict(self, candidate: _Candidate) -> ToolResultBlock:
        """重建带全 id/name/metadata/state，漏 state 会把挂起/失败洗成默认值"""
        origin = candidate.block
        return ToolResultBlock(
            id=origin.id,
            name=origin.name,
            output=[TextBlock(text=self._preview(candidate.origin_chars, candidate.input))],
            metadata=dict(origin.metadata) if origin.metadata else None,
            state=origin.state,
        )

    def _preview(self, origin_chars: int, tool_input: str | None) -> str:
        """占位带原入参，让模型知道当时问的是什么"""
        text = f"{self.EVICTED_PREFIX}{origin_chars}{self.EVICTED_CHARS}"
        if tool_input is not None:
            text += f"{self.EVICTED_INPUT}{tool_input}"
        return text + self.EVICTED_SUFFIX

    def _is_evicted(self, block: ToolResultBlock) -> bool:
        """靠占位前缀识别已清理块，不依赖 metadata"""
        output = block.output
        return (
            output is not None
            and len(output) == 1
            and isinstance(output[0], TextBlock)
            and (output[0].text or "").startswith(self.EVICTED_PREFIX)
        )
