"""
ReAct 执行引擎 — 替代 Java 侧的 AgentScope ``ReActAgent``。

AgentScope 的 Java SDK 在 Python 无对等物，这里按同一份事件协议自实现：
循环内每轮「推理 -> 工具调用 -> 结果回填」，事件名与 Java 侧
``AgentEvent.Type`` 一一对应，SSE 帧结构对前端保持不变。

  ==============================  ==============================
  AgentScope                      本模块
  ==============================  ==============================
  ReActAgent.streamEvents         ReActAgent.stream_events
  ReActAgent.interrupt            ReActAgent.interrupt
  RuntimeContext                  tools.RuntimeContext
  AgentEvent                      AgentEvent
  ReActAgentProvider              ReActAgentProvider
  ==============================  ==============================
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Protocol

import httpx

from app.agent.compaction import AgentContextCompactionMiddleware
from app.agent.enums import AgentEventType, MsgRole, ToolResultState
from app.agent.messages import (
    Msg,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from app.agent.state_store import CONTEXT_STATE_KEY, PgAgentStateStore
from app.agent.tools import AgentTool, ResolvedCatalog, RuntimeContext, ToolCallParam
from app.config import settings
from app.core.exceptions import RemoteException

logger = logging.getLogger(__name__)

STRUCTURED_OUTPUT_TOOL_NAME = "generate_response"
"""框架内部工具名，不暴露给业务（对应 ReActAgent.STRUCTURED_OUTPUT_TOOL_NAME）"""


# ---------------------------------------------------------------------------
# 事件 — mirrors AgentScope AgentEvent 子类
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    """
    ReAct 循环产生的事件。

    一个扁平结构承载全部事件类型，字段按 type 取用；Java 侧是一族子类，
    这里合并是为了让 SSE 桥用 match 分发而不必 isinstance 链。
    """

    type: AgentEventType
    delta: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    state: ToolResultState = ToolResultState.SUCCESS
    result_text: str | None = None
    hint: str = ""
    result: Msg | None = None


def _now_stamp() -> str:
    """跳天回放需全量时刻，不带时区偏移沿用前端约定"""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# 模型客户端
# ---------------------------------------------------------------------------

@dataclass
class ModelToolCall:
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCompletion:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)


DeltaHandler = Callable[[str, str], None]
"""(delta_type, delta) -> None；delta_type 取 "response" / "think" """


class AgentModelClient(Protocol):
    """Agent 侧模型调用口：一次流式补全，返回聚合后的完成量"""

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: DeltaHandler | None = None,
    ) -> ModelCompletion:
        ...


def msg_to_openai(msg: Msg) -> list[dict[str, Any]]:
    """
    Msg -> OpenAI 兼容消息。

    一条 Msg 可能同时带 tool_use 与 tool_result，OpenAI 协议里分属
    ``assistant.tool_calls`` 与 ``role=tool`` 两种消息，故返回列表。
    """
    role = msg.role
    text = msg.get_text_content()
    thinking = msg.get_thinking_content()
    tool_uses = msg.get_content_blocks(ToolUseBlock)
    tool_results = msg.get_content_blocks(ToolResultBlock)

    out: list[dict[str, Any]] = []

    if role == MsgRole.TOOL:
        for result in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": result.id,
                "content": result.output_text(),
            })
        return out

    if role == MsgRole.ASSISTANT:
        payload: dict[str, Any] = {"role": "assistant"}
        if text:
            payload["content"] = text
        elif not tool_uses:
            payload["content"] = ""
        else:
            payload["content"] = None
        if thinking:
            payload["reasoning_content"] = thinking
        if tool_uses:
            payload["tool_calls"] = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.input or {}, ensure_ascii=False),
                    },
                }
                for b in tool_uses
            ]
        out.append(payload)
        for result in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": result.id,
                "content": result.output_text(),
            })
        return out

    out.append({"role": role.value, "content": text})
    return out


class OpenAiCompatModelClient:
    """
    OpenAI 兼容端点的流式客户端（单模型，无 fallback）。

    Agent 侧配置 ``agent.chat.provider`` + ``agent.chat.model`` 引用
    ``ai.providers`` 解析 url / api-key / endpoints.chat，与 Java AgentProperties 一致。
    """

    def __init__(self, provider_key: str | None = None, model: str | None = None) -> None:
        chat_cfg = settings.agent.chat
        self.provider_key = provider_key or chat_cfg.provider
        self.model = model or chat_cfg.model

    def _resolve(self) -> tuple[str, str]:
        provider = getattr(settings.ai.providers, self.provider_key, None)
        if provider is None:
            raise RemoteException(f"Agent 模型供应商未配置: {self.provider_key}")
        base_url = (provider.url or "").rstrip("/")
        endpoint = (provider.endpoints.chat or "").lstrip("/")
        if not base_url or not endpoint:
            raise RemoteException(f"Agent 模型供应商地址不完整: {self.provider_key}")
        if not provider.api_key:
            raise RemoteException(f"Agent 模型供应商密钥缺失: {self.provider_key}")
        return f"{base_url}/{endpoint}", provider.api_key

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: DeltaHandler | None = None,
    ) -> ModelCompletion:
        url, api_key = self._resolve()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        completion = ModelCompletion()
        # 供应商把并行的多个 tool_call 按 index 分片下发，按 index 聚合
        calls: dict[int, ModelToolCall] = {}
        raw_args: dict[int, list[str]] = {}

        timeout = httpx.Timeout(settings.agent.sse_timeout_ms / 1000.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            ) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode("utf-8", errors="replace")
                    raise RemoteException(f"Agent 模型调用失败: HTTP {response.status_code} {detail[:500]}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        completion.thinking += reasoning
                        if on_delta:
                            on_delta("think", reasoning)

                    content = delta.get("content") or ""
                    if content:
                        completion.text += content
                        if on_delta:
                            on_delta("response", content)

                    for call in delta.get("tool_calls") or []:
                        index = call.get("index") or 0
                        slot = calls.setdefault(index, ModelToolCall())
                        if call.get("id"):
                            slot.id = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot.name = function["name"]
                        if function.get("arguments"):
                            raw_args.setdefault(index, []).append(function["arguments"])

        for index, slot in sorted(calls.items()):
            joined = "".join(raw_args.get(index, []))
            try:
                parsed = json.loads(joined) if joined else {}
                slot.arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                logger.warning("工具调用入参非法 JSON, name: %s, raw: %s", slot.name, joined[:200])
                slot.arguments = {}
            completion.tool_calls.append(slot)

        return completion


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    ReAct 循环：推理 -> 工具调用 -> 结果回填，直到模型不再要求调用工具。

    会话上下文按 (userId, sessionId) 从状态存储加载与回写，多轮记忆跨请求续上。
    """

    def __init__(
        self,
        model_client: AgentModelClient,
        tools: list[AgentTool],
        system_prompt: str,
        state_store: PgAgentStateStore,
        max_iters: int = 10,
        max_retries: int = 2,
        memory: AgentContextCompactionMiddleware | None = None,
        catalog: ResolvedCatalog | None = None,
    ) -> None:
        self.model_client = model_client
        self.tools_by_name = {t.name: t for t in tools}
        self.system_prompt = system_prompt or ""
        self.state_store = state_store
        self.max_iters = max_iters
        self.max_retries = max_retries
        self.memory = memory
        self.catalog = catalog
        # 打断旗标按 (userId, sessionId) 记，interrupt 与循环在不同 task 里跑
        self._interrupted: set[tuple[str, str]] = set()

    # -- 打断 ---------------------------------------------------------------

    def interrupt(self, user_id: str, session_id: str) -> None:
        self._interrupted.add((user_id, session_id))

    def is_interrupted(self, user_id: str, session_id: str) -> bool:
        return (user_id, session_id) in self._interrupted

    def clear_interrupt(self, user_id: str, session_id: str) -> None:
        self._interrupted.discard((user_id, session_id))

    # -- 主循环 -------------------------------------------------------------

    async def stream_events(
        self,
        question: str,
        runtime_context: RuntimeContext,
    ) -> AsyncGenerator[AgentEvent, None]:
        """
        按事件流跑完一轮对话；调用方负责把事件桥到 SSE。

        循环本体跑在独立 task 里，事件经队列转手：模型客户端的增量回调是同步的，
        不拆开就只能等整轮跑完再一次性吐，流式就没了。
        """
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        task = asyncio.create_task(self._run_turn(question, runtime_context, queue))

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            # 哨兵只代表事件发完，循环本体的异常要在这里重新抛出
            await task
        finally:
            if not task.done():
                task.cancel()

    async def _run_turn(
        self,
        question: str,
        runtime_context: RuntimeContext,
        queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        """ReAct 循环本体，全部产出经 queue 外送，结束投 None 哨兵"""
        user_id = runtime_context.user_id
        session_id = runtime_context.session_id
        self.clear_interrupt(user_id, session_id)

        def emit(delta_type: str, delta: str) -> None:
            event_type = (
                AgentEventType.THINKING_BLOCK_DELTA
                if delta_type == "think"
                else AgentEventType.TEXT_BLOCK_DELTA
            )
            queue.put_nowait(AgentEvent(event_type, delta=delta))

        try:
            context = await self._load_context(user_id, session_id)
            context.append(Msg(
                role=MsgRole.USER,
                content=[TextBlock(text=question)],
                timestamp=_now_stamp(),
            ))

            tool_schemas = [t.to_function_schema() for t in self.tools_by_name.values()]
            exhausted = True

            for _ in range(self.max_iters):
                if self.is_interrupted(user_id, session_id):
                    exhausted = False
                    break

                if self.memory is not None:
                    await self.memory.before_reasoning(context, user_id, session_id)

                completion = await self._reason_with_retry(context, tool_schemas, emit)

                if completion.tool_calls:
                    context.append(self._to_tool_use_msg(completion))
                    for call in completion.tool_calls:
                        queue.put_nowait(AgentEvent(
                            AgentEventType.TOOL_CALL_START,
                            tool_call_id=call.id,
                            tool_call_name=call.name,
                        ))
                        if self.is_interrupted(user_id, session_id):
                            break
                        result_block = await self._execute_tool(call, runtime_context)
                        # 原生工具一次返回全量结果，故不拆 TOOL_RESULT_TEXT_DELTA，
                        # 结果文本随 TOOL_RESULT_END 一并下发
                        queue.put_nowait(AgentEvent(
                            AgentEventType.TOOL_RESULT_END,
                            tool_call_id=call.id,
                            tool_call_name=call.name,
                            state=result_block.state,
                            result_text=result_block.output_text(),
                        ))
                        context.append(Msg(role=MsgRole.TOOL, content=[result_block]))
                    continue

                # 无工具调用即终答
                result_msg = self._to_answer_msg(completion)
                context.append(result_msg)
                queue.put_nowait(AgentEvent(AgentEventType.AGENT_RESULT, result=result_msg))
                exhausted = False
                break

            if exhausted and not self.is_interrupted(user_id, session_id):
                # 跑满迭代上限：仍生成一次总结，只提示不判失败
                queue.put_nowait(AgentEvent(AgentEventType.EXCEED_MAX_ITERS))
                summary = await self._final_summary(context)
                if summary is not None:
                    context.append(summary)
                    queue.put_nowait(AgentEvent(AgentEventType.AGENT_RESULT, result=summary))

            await self._save_context(user_id, session_id, context)
        finally:
            self.clear_interrupt(user_id, session_id)
            queue.put_nowait(None)

    # -- 单轮推理 -----------------------------------------------------------

    async def _reason_with_retry(
        self,
        context: list[Msg],
        tool_schemas: list[dict[str, Any]],
        emit: DeltaHandler | None = None,
    ) -> ModelCompletion:
        """单次模型调用失败重试 max_retries 次"""
        messages = self._build_upstream(context)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.model_client.stream_completion(messages, tool_schemas, emit)
            except Exception as e:
                last_error = e
                logger.warning("Agent 推理失败, 第 %s 次重试: %s", attempt + 1, e)
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        raise RemoteException("Agent 推理失败，请稍后再试", cause=last_error)

    def _build_upstream(self, context: list[Msg]) -> list[dict[str, Any]]:
        """人设 + 上下文摊成 OpenAI 消息列表"""
        messages: list[dict[str, Any]] = []
        if self.system_prompt.strip():
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in context:
            messages.extend(msg_to_openai(msg))
        return messages

    @staticmethod
    def _to_tool_use_msg(completion: ModelCompletion) -> Msg:
        return Msg(
            role=MsgRole.ASSISTANT,
            content=[
                ToolUseBlock(id=c.id, name=c.name, input=dict(c.arguments or {}))
                for c in completion.tool_calls
            ],
            timestamp=_now_stamp(),
        )

    @staticmethod
    def _to_answer_msg(completion: ModelCompletion) -> Msg:
        blocks: list[Any] = []
        if completion.thinking:
            blocks.append(ThinkingBlock(thinking=completion.thinking))
        if completion.text:
            blocks.append(TextBlock(text=completion.text))
        return Msg(role=MsgRole.ASSISTANT, content=blocks, timestamp=_now_stamp())

    async def _execute_tool(self, call: ModelToolCall, runtime_context: RuntimeContext) -> ToolResultBlock:
        tool = self.tools_by_name.get(call.name)
        if tool is None:
            return AgentTool.build_result(call.id, call.name, f"未知工具: {call.name}", True)
        try:
            return await tool.call(ToolCallParam(
                tool_call_id=call.id,
                input=dict(call.arguments or {}),
                runtime_context=runtime_context,
            ))
        except Exception as e:
            logger.error("工具执行异常, name: %s, error: %s", call.name, e)
            return AgentTool.build_result(call.id, call.name, f"工具执行异常: {e}", True)

    async def _final_summary(self, context: list[Msg]) -> Msg | None:
        """达到迭代上限后不带工具再问一次，逼模型给出当前结论"""
        try:
            completion = await self.model_client.stream_completion(
                self._build_upstream(context), [], None,
            )
        except Exception as e:
            logger.error("Agent 迭代上限总结失败: %s", e)
            return None
        if not completion.text:
            return None
        return Msg(
            role=MsgRole.ASSISTANT,
            content=[TextBlock(text=completion.text)],
            timestamp=_now_stamp(),
        )

    # -- 状态存取 -----------------------------------------------------------

    async def _load_context(self, user_id: str, session_id: str) -> list[Msg]:
        raw = await self.state_store.get_list(user_id, session_id, CONTEXT_STATE_KEY)
        return [Msg.from_dict(item) for item in raw if isinstance(item, dict)]

    async def _save_context(self, user_id: str, session_id: str, context: list[Msg]) -> None:
        await self.state_store.save_list(
            user_id, session_id, CONTEXT_STATE_KEY, [m.to_dict() for m in context],
        )
