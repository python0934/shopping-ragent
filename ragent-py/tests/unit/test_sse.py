"""Unit tests for app.core.sse — SSE event formatting and sender."""

import asyncio
import json

import pytest

from app.core.sse import (
    SSEEventType,
    SseSender,
    format_sse_data_only,
    format_sse_event,
    sse_event_stream,
)


class TestFormatSseEvent:
    def test_named_event_with_dict(self):
        result = format_sse_event("meta", {"conversationId": "123", "taskId": "456"})
        assert result.startswith("event: meta\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        # Parse the data portion
        data_line = result.split("\n")[1]
        data_str = data_line[len("data: "):]
        parsed = json.loads(data_str)
        assert parsed["conversationId"] == "123"
        assert parsed["taskId"] == "456"

    def test_named_event_with_string(self):
        result = format_sse_event("done", "[DONE]")
        assert "event: done\n" in result
        assert "data: [DONE]\n" in result

    def test_named_event_with_chinese(self):
        result = format_sse_event("message", {"type": "response", "delta": "你好世界"})
        assert "你好世界" in result
        # Ensure no unicode escaping
        assert "\\u" not in result

    def test_data_only_format(self):
        result = format_sse_data_only({"key": "value"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert "event:" not in result


class TestSSEEventType:
    def test_event_names(self):
        assert SSEEventType.META == "meta"
        assert SSEEventType.MESSAGE == "message"
        assert SSEEventType.FINISH == "finish"
        assert SSEEventType.DONE == "done"
        assert SSEEventType.CANCEL == "cancel"
        assert SSEEventType.REJECT == "reject"
        assert SSEEventType.TOOL == "tool"
        assert SSEEventType.HINT == "hint"


class TestSseSender:
    @pytest.fixture
    def sender_and_queue(self):
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        sender = SseSender(queue)
        return sender, queue

    @pytest.mark.asyncio
    async def test_send_event(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_event("meta", {"id": "1"})
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: meta" in event
        assert '"id":"1"' in event

    @pytest.mark.asyncio
    async def test_send_meta(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_meta("conv1", "task1")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: meta" in event
        assert '"conversationId":"conv1"' in event
        assert '"taskId":"task1"' in event

    @pytest.mark.asyncio
    async def test_send_message(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_message("response", "你好")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: message" in event
        assert '"type":"response"' in event
        assert '"delta":"你好"' in event

    @pytest.mark.asyncio
    async def test_send_finish(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_finish("msg1", title="标题", message_status="NORMAL")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: finish" in event
        assert '"messageId":"msg1"' in event
        assert '"messageStatus":"NORMAL"' in event

    @pytest.mark.asyncio
    async def test_send_done(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_done()
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: done" in event
        assert "[DONE]" in event

    @pytest.mark.asyncio
    async def test_complete_closes_sender(self, sender_and_queue):
        sender, queue = sender_and_queue
        assert sender.is_closed is False
        await sender.complete()
        assert sender.is_closed is True
        sentinel = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert sentinel is None

    @pytest.mark.asyncio
    async def test_double_complete_is_safe(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.complete()
        await sender.complete()  # Should not raise
        assert sender.is_closed is True

    @pytest.mark.asyncio
    async def test_send_after_close_is_noop(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.complete()
        await sender.send_event("meta", {"id": "1"})  # Should not raise
        # Only the None sentinel should be in the queue
        item = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert item is None
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_send_cancel(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_cancel("msg1")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: cancel" in event
        assert '"messageId":"msg1"' in event

    @pytest.mark.asyncio
    async def test_send_reject(self, sender_and_queue):
        sender, queue = sender_and_queue
        await sender.send_reject("演示模式")
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert "event: reject" in event
        assert "演示模式" in event


class TestSseEventStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events_until_none(self):
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        await queue.put("event: meta\ndata: {}\n\n")
        await queue.put("event: done\ndata: [DONE]\n\n")
        await queue.put(None)

        events = []
        async for chunk in sse_event_stream(queue):
            events.append(chunk)

        assert len(events) == 2
        assert "meta" in events[0]
        assert "done" in events[1]
