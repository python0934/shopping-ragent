"""
Agent 流式事件桥 — mirrors agent.service.handler.AgentStreamEventBridge / AgentRunHandle.

把 ReAct 循环的内部事件翻译成 SSE 帧，推入 ``SseSender`` 的队列，同时累积运行轨迹块。
事件名即前端 ``EventSource`` 的监听名，改名等于改契约，一律照搬 Java 侧字面量。

落库分两种收尾：
  - 正常完成 -> ``finish``，messageStatus = NORMAL
  - 取消/断开 -> ``cancel``，messageStatus = INTERRUPTED；只有已产出内容才落库
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable

from app.agent.enums import AgentEventType, AgentMessageStatus, AgentSSEEventType, ToolResultState
from app.agent.react_agent import AgentEvent
from app.core.sse import SseSender
from app.schemas.agent import (
    AgentBlock,
    AgentCompletionPayload,
    AgentHintPayload,
    AgentMessageDelta,
    AgentMetaPayload,
    AgentToolProgress,
)

logger = logging.getLogger(__name__)

DELTA_TYPE_RESPONSE = "response"
DELTA_TYPE_THINK = "think"

TOOL_STATUS_START = "start"
TOOL_STATUS_END = "end"

TOOL_STATE_RUNNING = "running"
TOOL_STATE_DONE = "done"
TOOL_STATE_FAILED = "failed"
TOOL_STATE_INTERRUPTED = "interrupted"

HINT_AGENT = "AGENT_HINT"
HINT_MAX_ITERATIONS = "MAX_ITERATIONS"
HINT_MAX_ITERATIONS_TEXT = "已达到最大推理轮次，正在根据已有信息生成回答"

TOOL_RESULT_MAX_CHARS = 64_000
"""单条工具结果回传给前端的上限，超出截断——轨迹是给浏览器渲染的，不该无界增长"""

TOOL_TRUNCATED_SUFFIX = "\n…（结果过长已截断）"

FALLBACK_CALL_KEY = "__anonymous__"
"""供应商漏发 toolCallId 时的占位键，保证进度仍能配对"""

BLOCK_TIME_PATTERN = "%Y-%m-%dT%H:%M:%S"

PersistCallback = Callable[..., Awaitable[str]]
"""(content, thinking, blocks, status) -> messageId"""


def _block_time() -> str:
    return datetime.now().strftime(BLOCK_TIME_PATTERN)


# ---------------------------------------------------------------------------
# AgentRunHandle — mirrors agent.service.handler.AgentRunHandle
# ---------------------------------------------------------------------------

@dataclass
class AgentRunHandle:
    """
    一次运行的把手：停止入口。

    ``on_stop`` 由编排层挂上（释放闸门 + 驱逐状态缓存），桥只负责调它。
    """

    task_id: str
    conversation_id: str
    user_id: str
    cancel: Callable[[], None] = field(default=lambda: None)
    on_stop: Callable[[], Awaitable[None]] | None = None
    _stopping: bool = field(default=False, repr=False)

    async def stop(self) -> None:
        """幂等：重复停止只生效一次"""
        if self._stopping:
            return
        self._stopping = True
        self.cancel()
        if self.on_stop is not None:
            await self.on_stop()


# ---------------------------------------------------------------------------
# AgentStreamEventBridge
# ---------------------------------------------------------------------------

@dataclass
class _ToolSlot:
    """一次工具调用的进行中状态"""

    call_key: str
    name: str
    display_name: str
    block_index: int
    buffered: list[str] = field(default_factory=list)
    buffered_chars: int = 0
    truncated: bool = False
    done: bool = False
    failed: bool = False


class AgentStreamEventBridge:
    """
    订阅 ReAct 事件流 -> 推 SSE + 累积 blocks -> 收尾落库。

    用法::

        bridge = AgentStreamEventBridge(sender, ..., persist_callback)
        await bridge.send_meta()
        await bridge.run(agent.stream_events(question, ctx))
    """

    def __init__(
        self,
        sender: SseSender,
        conversation_id: str,
        task_id: str,
        persist_callback: PersistCallback,
        catalog_display_name: Callable[[str], str] | None = None,
    ) -> None:
        self.sender = sender
        self.conversation_id = conversation_id
        self.task_id = task_id
        self._persist = persist_callback
        self._display_name = catalog_display_name or (lambda n: n)

        self.blocks: list[AgentBlock] = []
        self._answer_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._tool_slots: dict[str, _ToolSlot] = {}
        self._tool_order: list[str] = []
        self._settled = False
        self._final_answer: str = ""
        self.persisted_message_id: str = ""

    def set_display_name_resolver(self, resolver: Callable[[str], str] | None) -> None:
        """
        后接展示名解析。

        meta 帧得先于工具目录解析下发（前端等不到首帧会当连接卡死），
        而展示名要等目录定格，故桥先建、解析口后插。
        """
        self._display_name = resolver or (lambda n: n)

    # -- 元事件 -------------------------------------------------------------

    async def send_meta(self) -> None:
        """首帧：前端拿到 conversationId 就能建会话条目，taskId 供停止用"""
        await self.sender.send_event(
            AgentSSEEventType.META.value,
            AgentMetaPayload(conversationId=self.conversation_id, taskId=self.task_id).model_dump(),
        )

    # -- 主循环 -------------------------------------------------------------

    async def run(self, events: AsyncGenerator[AgentEvent, None]) -> None:
        """消费事件流，全程写 sender；正常走完发 finish，被打断发 cancel"""
        try:
            async for event in events:
                await self._handle(event)
            await self._on_complete()
        except asyncio.CancelledError:
            await self._on_cancelled()
            raise
        except Exception as e:
            logger.exception("Agent 流式事件桥异常: %s", e)
            await self._on_error(e)
        finally:
            await events.aclose()

    async def _handle(self, event: AgentEvent) -> None:
        etype = event.type

        if etype == AgentEventType.TEXT_BLOCK_DELTA:
            if not event.delta:
                return
            self._answer_parts.append(event.delta)
            self._extend_block("answer", event.delta)
            await self._push(
                AgentSSEEventType.MESSAGE,
                AgentMessageDelta(type=DELTA_TYPE_RESPONSE, delta=event.delta).model_dump(),
            )

        elif etype == AgentEventType.THINKING_BLOCK_DELTA:
            if not event.delta:
                return
            self._thinking_parts.append(event.delta)
            self._extend_block("reasoning", event.delta)
            await self._push(
                AgentSSEEventType.MESSAGE,
                AgentMessageDelta(type=DELTA_TYPE_THINK, delta=event.delta).model_dump(),
            )

        elif etype == AgentEventType.TOOL_CALL_START:
            await self._on_tool_start(event)

        elif etype == AgentEventType.TOOL_RESULT_TEXT_DELTA:
            self._buffer_tool_result(event)

        elif etype == AgentEventType.TOOL_RESULT_END:
            await self._on_tool_end(event)

        elif etype == AgentEventType.HINT_BLOCK:
            if event.hint:
                await self._push_hint(HINT_AGENT, event.hint)

        elif etype == AgentEventType.EXCEED_MAX_ITERS:
            await self._push_hint(HINT_MAX_ITERATIONS, HINT_MAX_ITERATIONS_TEXT)

        elif etype == AgentEventType.AGENT_RESULT:
            if event.result is not None:
                self._final_answer = event.result.get_text_content()

    # -- 推送 ---------------------------------------------------------------

    async def _push(self, event_type: AgentSSEEventType, data: Any) -> None:
        if self.sender.is_closed:
            return
        await self.sender.send_event(event_type.value, data)

    async def _push_hint(self, code: str, text: str) -> None:
        await self._push(
            AgentSSEEventType.HINT,
            AgentHintPayload(code=code, text=text).model_dump(),
        )

    # -- 文本 / 思考块 ------------------------------------------------------

    def _extend_block(self, kind: str, delta: str) -> None:
        """同类增量续在末块上；被工具打断后另起新块，轨迹才看得出穿插顺序"""
        if self.blocks and self.blocks[-1].kind == kind:
            self.blocks[-1].text = (self.blocks[-1].text or "") + delta
        else:
            self.blocks.append(AgentBlock(kind=kind, at=_block_time(), text=delta))

    # -- 工具 ---------------------------------------------------------------

    async def _on_tool_start(self, event: AgentEvent) -> None:
        name = event.tool_call_name or ""
        call_key = event.tool_call_id or FALLBACK_CALL_KEY
        display_name = self._display_name(name) or name

        self.blocks.append(AgentBlock(
            kind="tool",
            at=_block_time(),
            name=name,
            displayName=display_name,
            status=TOOL_STATE_RUNNING,
            toolCallId=event.tool_call_id or "",
        ))

        # 同名工具并发调用会撞键，撞了就退化成按调用序追加
        if call_key in self._tool_slots:
            call_key = f"{call_key}#{len(self._tool_order)}"

        self._tool_slots[call_key] = _ToolSlot(
            call_key=call_key,
            name=name,
            display_name=display_name,
            block_index=len(self.blocks) - 1,
        )
        self._tool_order.append(call_key)

        await self._push(
            AgentSSEEventType.TOOL,
            AgentToolProgress(
                name=name, displayName=display_name, status=TOOL_STATUS_START,
            ).to_wire(),
        )

    def _buffer_tool_result(self, event: AgentEvent) -> None:
        """结果增量攒着，收尾一次下发——前端只关心最终结果，逐帧推会刷爆轨迹"""
        slot = self._find_slot(event.tool_call_id)
        if slot is None or not event.delta:
            return
        if slot.buffered_chars >= TOOL_RESULT_MAX_CHARS:
            slot.truncated = True
            return
        slot.buffered.append(event.delta)
        slot.buffered_chars += len(event.delta)

    async def _on_tool_end(self, event: AgentEvent) -> None:
        slot = self._find_slot(event.tool_call_id)
        failed = event.state == ToolResultState.ERROR

        if event.result_text is not None:
            text = event.result_text
        elif slot is not None:
            text = "".join(slot.buffered)
        else:
            text = ""

        if len(text) > TOOL_RESULT_MAX_CHARS:
            text = text[:TOOL_RESULT_MAX_CHARS]
            if slot is not None:
                slot.truncated = True

        name = (slot.name if slot else event.tool_call_name) or ""
        display_name = (slot.display_name if slot else self._display_name(name)) or name

        if slot is not None:
            slot.done = True
            slot.failed = failed
            self._seal_tool_block(slot, text, failed)

        await self._push(
            AgentSSEEventType.TOOL,
            AgentToolProgress(
                name=name,
                displayName=display_name,
                status=TOOL_STATUS_END,
                result=text,
                ok=not failed,
            ).to_wire(),
        )

    def _find_slot(self, tool_call_id: str | None) -> _ToolSlot | None:
        slot = self._tool_slots.get(tool_call_id or FALLBACK_CALL_KEY)
        if slot is None and tool_call_id:
            # 供应商可能在结束帧才补出 id，按名称回落最近一个未完成的槽
            for call_key in reversed(self._tool_order):
                candidate = self._tool_slots.get(call_key)
                if candidate is not None and not candidate.done:
                    return candidate
        return slot

    def _seal_tool_block(self, slot: _ToolSlot, text: str, failed: bool) -> None:
        if slot.block_index >= len(self.blocks):
            return
        block = self.blocks[slot.block_index]
        block.status = TOOL_STATE_FAILED if failed else TOOL_STATE_DONE
        block.result = text + TOOL_TRUNCATED_SUFFIX if slot.truncated else text

    # -- 轨迹封口 -----------------------------------------------------------

    @property
    def content(self) -> str:
        """以流式增量为准，空则回落框架终答"""
        streamed = "".join(self._answer_parts)
        return streamed if streamed else self._final_answer

    @property
    def thinking_content(self) -> str:
        return "".join(self._thinking_parts)

    def settled_blocks(self) -> list[dict[str, Any]] | None:
        """
        封口轨迹块：未跑完的工具标 interrupted，剔掉空块。

        全空返回 None —— 空轨迹落库只会让前端多渲染一个空气泡。
        """
        done_indexes = {slot.block_index for slot in self._tool_slots.values() if slot.done}

        sealed: list[AgentBlock] = []
        for index, block in enumerate(self.blocks):
            if block.kind == "tool":
                if not block.name:
                    continue
                if index not in done_indexes:
                    block.status = TOOL_STATE_INTERRUPTED
            elif not (block.text or "").strip():
                continue
            sealed.append(block)

        return [b.to_wire() for b in sealed] if sealed else None

    # -- 收尾 ---------------------------------------------------------------

    async def _on_complete(self) -> None:
        """正常收尾：落库 + finish + done"""
        if self._settled:
            return
        self._settled = True

        self.persisted_message_id = await self._do_persist(AgentMessageStatus.NORMAL)
        await self._push(
            AgentSSEEventType.FINISH,
            AgentCompletionPayload(
                messageId=self.persisted_message_id or None,
                messageStatus=AgentMessageStatus.NORMAL.value,
            ).to_wire(),
        )
        await self._push(AgentSSEEventType.DONE, {})
        await self.sender.complete()

    async def _on_cancelled(self) -> None:
        """取消收尾：有内容或有工具轨迹才落库，状态 INTERRUPTED"""
        if self._settled:
            return
        self._settled = True

        if not self.content.strip() and not self._tool_order:
            # 一个字都没吐就断了，落库只会留一条空消息
            await self._push(AgentSSEEventType.CANCEL, {})
            await self.sender.complete()
            return

        self.persisted_message_id = await self._do_persist(AgentMessageStatus.INTERRUPTED)
        await self._push(
            AgentSSEEventType.CANCEL,
            AgentCompletionPayload(
                messageId=self.persisted_message_id or None,
                messageStatus=AgentMessageStatus.INTERRUPTED.value,
            ).to_wire(),
        )
        await self.sender.complete()

    async def _on_error(self, error: Exception) -> None:
        if self._settled:
            return
        self._settled = True
        if self.content.strip():
            # 已经吐了半截答案，落库标 INTERRUPTED 比整条丢掉好
            await self._do_persist(AgentMessageStatus.INTERRUPTED)
        await self.sender.fail(error)

    async def _do_persist(self, status: AgentMessageStatus) -> str:
        """落库失败不能连带整条流失败：答案已经吐给用户了，丢的只是一条历史"""
        try:
            return await self._persist(
                self.content,
                self.thinking_content,
                self.settled_blocks(),
                status,
            )
        except Exception as e:
            logger.error("Agent 消息落库失败: %s", e)
            return ""
