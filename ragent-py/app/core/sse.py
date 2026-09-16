"""
SSE (Server-Sent Events) infrastructure — mirrors Java SseEmitterSender / StreamTaskManager.

Provides:
  - SseSender: thread-safe async SSE event sender
  - SSE event formatting helpers
  - StreamTaskManager: lifecycle management for streaming tasks
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE event formatting
# ---------------------------------------------------------------------------

def format_sse_event(event_name: str, data: Any) -> str:
    """
    Format a single SSE event string.

    Output format (mirrors Spring SseEmitter):
        event: <event_name>
        data: <json_or_raw>
        <blank line>
    """
    if isinstance(data, str):
        data_str = data
    else:
        data_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {data_str}\n\n"


def format_sse_data_only(data: Any) -> str:
    """Format a data-only SSE event (no event name)."""
    if isinstance(data, str):
        data_str = data
    else:
        data_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data_str}\n\n"


# ---------------------------------------------------------------------------
# SSE event names — mirrors Java SSEEventType enum
# ---------------------------------------------------------------------------

class SSEEventType:
    META = "meta"
    MESSAGE = "message"
    FINISH = "finish"
    DONE = "done"
    CANCEL = "cancel"
    REJECT = "reject"
    # Agent-only events
    TOOL = "tool"
    HINT = "hint"


# ---------------------------------------------------------------------------
# SseSender — mirrors Java SseEmitterSender
# ---------------------------------------------------------------------------

class SseSender:
    """
    Async SSE sender with safe close semantics.

    Mirrors Java SseEmitterSender:
    - AtomicBoolean closed → asyncio flag
    - sendEvent(name, data) → send event to queue
    - complete() → CAS close
    - fail(throwable) → CAS close with error
    """

    def __init__(self, queue: asyncio.Queue[str | None]):
        self._queue = queue
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def send_event(self, event_name: str, data: Any) -> None:
        """Send a named SSE event. No-op if closed."""
        if self._closed:
            return
        try:
            await self._queue.put(format_sse_event(event_name, data))
        except Exception as e:
            await self.fail(e)

    async def send_meta(self, conversation_id: str, task_id: str) -> None:
        """Convenience: send meta event."""
        await self.send_event(SSEEventType.META, {
            "conversationId": conversation_id,
            "taskId": task_id,
        })

    async def send_message(self, msg_type: str, delta: str) -> None:
        """Convenience: send message chunk."""
        await self.send_event(SSEEventType.MESSAGE, {
            "type": msg_type,
            "delta": delta,
        })

    async def send_finish(
        self,
        message_id: str,
        title: str | None = None,
        sources: list[dict] | None = None,
        message_status: str = "NORMAL",
    ) -> None:
        """Convenience: send finish event."""
        payload: dict[str, Any] = {
            "messageId": message_id,
            "messageStatus": message_status,
        }
        if title is not None:
            payload["title"] = title
        if sources is not None:
            payload["sources"] = sources
        await self.send_event(SSEEventType.FINISH, payload)

    async def send_done(self) -> None:
        """Send [DONE] sentinel and complete."""
        await self.send_event(SSEEventType.DONE, "[DONE]")

    async def send_cancel(self, message_id: str) -> None:
        """Send cancel event."""
        await self.send_event(SSEEventType.CANCEL, {"messageId": message_id})

    async def send_reject(self, message: str) -> None:
        """Send reject event."""
        await self.send_event(SSEEventType.REJECT, {"message": message})

    async def complete(self) -> None:
        """Normal completion — CAS close, put None sentinel."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self._queue.put(None)  # sentinel

    async def fail(self, error: Exception | None = None) -> None:
        """Error completion — CAS close."""
        if error:
            logger.warning("SSE send failed: %s", error)
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        await self._queue.put(None)


# ---------------------------------------------------------------------------
# SSE event stream generator — used as StreamingResponse body
# ---------------------------------------------------------------------------

async def sse_event_stream(queue: asyncio.Queue[str | None]) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE strings from a queue.
    Stops when it receives a None sentinel.
    """
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event


# ---------------------------------------------------------------------------
# StreamTaskManager — mirrors Java StreamTaskManager
# ---------------------------------------------------------------------------

class StreamTaskManager:
    """
    Manages active streaming tasks (SSE connections).
    Supports cancellation by task_id.
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._senders: dict[str, SseSender] = {}
        self._lock = asyncio.Lock()

    async def register(self, task_id: str, task: asyncio.Task, sender: SseSender) -> None:
        async with self._lock:
            self._tasks[task_id] = task
            self._senders[task_id] = sender

    async def unregister(self, task_id: str) -> None:
        async with self._lock:
            self._tasks.pop(task_id, None)
            self._senders.pop(task_id, None)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running stream task by ID. Returns True if found."""
        async with self._lock:
            task = self._tasks.get(task_id)
            sender = self._senders.get(task_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def get_sender(self, task_id: str) -> SseSender | None:
        return self._senders.get(task_id)


# Global singleton
stream_task_manager = StreamTaskManager()
