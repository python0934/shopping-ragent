"""
Phase 7 端到端测试 — Agent SSE 流式对话全链路.

真跑：认证中间件 -> 并发闸门 -> 会话/提问落库 -> Agent 装配 -> ReAct 循环
-> 事件桥 -> SSE 帧 -> 答复落库 -> 闸门释放。
只替换：Redis（认证 + 闸门）、大模型（流式补全）、知识检索门面。

断言全部落在前端实际消费的线上形态上：事件名、data 字段名（camelCase）、
落库的轨迹块结构，以及「异常也必须是 HTTP 200」这条硬契约。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable

from app.agent.enums import AgentMessageStatus
from app.agent.react_agent import ModelCompletion
from app.agent.run_gate import RUNNING_KEY_PREFIX
from app.config import settings
from app.core.error_code import BaseErrorCode

from .conftest import (
    AUTH_HEADERS,
    TEST_USER_ID,
    FakeMcpExecutor,
    answer_step,
    event_names,
    frames_of,
    run_async,
    seed_intent_node,
    stream_frames,
    tool_step,
)

ANSWER = "退货政策是签收后七天内可无理由退货。"
THINKING = "用户在问售后政策，先查知识库。"


# ---------------------------------------------------------------------------
# 断言辅助
# ---------------------------------------------------------------------------

def _joined(frames: list[Any], delta_type: str) -> str:
    """把同一类增量帧按到达顺序拼回完整文本"""
    return "".join(
        f.data.get("delta", "")
        for f in frames_of(frames, "message")
        if f.data.get("type") == delta_type
    )


def _load_messages(client: Any, conversation_id: str) -> list[dict[str, Any]]:
    """走真接口取会话轨迹，顺带验一遍消息查询契约"""
    body = client.get(
        f"/agent/v1/conversations/{conversation_id}/messages", headers=AUTH_HEADERS,
    ).json()
    assert body["code"] == "0"
    return body["data"]


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0, what: str = "条件") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"等待{what}超时")


def _gate_key() -> str:
    return RUNNING_KEY_PREFIX + TEST_USER_ID


def _gate_slot(client: Any) -> str | None:
    return client.fake_redis.data.get(_gate_key())


# ---------------------------------------------------------------------------
# 正常链路
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_full_stream_chain(self, client, install_agent, seeded_agent):
        """一轮直答：meta -> message -> finish -> done，答复与轨迹一并落库"""
        install_agent(answer_step(ANSWER, thinking=THINKING))

        frames = stream_frames(client, "退货政策是什么")

        # 事件名与顺序即前端 EventSource 的契约
        assert event_names(frames)[0] == "meta"
        assert event_names(frames)[-2:] == ["finish", "done"]
        assert not frames_of(frames, "cancel")

        meta = frames_of(frames, "meta")[0].data
        conversation_id = meta["conversationId"]
        assert conversation_id
        assert meta["taskId"]

        # 增量拼回来必须等于终答，思考链走 think 通道
        assert _joined(frames, "response") == ANSWER
        assert _joined(frames, "think") == THINKING
        # 分片下发才算流式：一整坨吐出来前端就没有打字机效果了
        assert len(frames_of(frames, "message")) > 1

        finish = frames_of(frames, "finish")[0].data
        assert finish["messageStatus"] == AgentMessageStatus.NORMAL.value
        assert finish["messageId"]
        assert frames_of(frames, "done")[0].data == {}

        rows = _load_messages(client, conversation_id)
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[0]["content"] == "退货政策是什么"

        assistant = rows[1]
        assert assistant["content"] == ANSWER
        assert assistant["thinkingContent"] == THINKING
        assert assistant["messageStatus"] == AgentMessageStatus.NORMAL.value
        assert assistant["id"] == finish["messageId"]
        assert [(b["kind"], b["text"]) for b in assistant["blocks"]] == [
            ("reasoning", THINKING), ("answer", ANSWER),
        ]

    def test_gate_is_released_after_stream(self, client, install_agent, seeded_agent):
        """运行位没释放的话，该用户会被自己的上一轮对话永久挡在门外"""
        install_agent(answer_step(ANSWER))

        frames = stream_frames(client, "退货政策是什么")

        assert frames_of(frames, "done")
        assert _gate_slot(client) is None

    def test_conversation_shows_up_in_list(self, client, install_agent, seeded_agent):
        """流跑完后侧边栏要能立刻看到这条会话"""
        install_agent(answer_step(ANSWER))

        conversation_id = frames_of(stream_frames(client, "退货政策是什么"), "meta")[0].data["conversationId"]

        body = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()
        assert body["code"] == "0"
        listed = [c for c in body["data"] if c["conversationId"] == conversation_id]
        assert listed, f"会话未出现在列表中: {body['data']}"
        assert listed[0]["title"]
        assert listed[0]["turns"] == 1

    def test_multi_turn_keeps_context(self, client, install_agent, seeded_agent):
        """第二轮沿用同一个 conversationId，模型要能看到上一轮的答复"""
        stub = install_agent(answer_step("第一轮答复"), answer_step("第二轮答复"))

        first = frames_of(stream_frames(client, "第一个问题"), "meta")[0].data
        second = frames_of(
            stream_frames(client, "第二个问题", conversation_id=first["conversationId"]), "meta",
        )[0].data

        assert second["conversationId"] == first["conversationId"]
        assert second["taskId"] != first["taskId"]

        # 上下文由状态存储跨请求续上：第二轮的上游消息里必须带第一轮的内容
        upstream = stub.model.calls[1]["messages"]
        assert any("第一轮答复" in str(m.get("content", "")) for m in upstream)

        rows = _load_messages(client, first["conversationId"])
        assert [r["role"] for r in rows] == ["user", "assistant", "user", "assistant"]

    def test_system_prompt_is_loaded_from_agent_profile(self, client, install_agent, seeded_agent):
        """人设来自 t_agent_prompt 的 AGENT_MAIN 槽位，没配就等于没上身"""
        stub = install_agent(answer_step(ANSWER))

        stream_frames(client, "退货政策是什么")

        system = [m for m in stub.model.calls[0]["messages"] if m["role"] == "system"]
        assert system and "端到端测试用的智能体" in system[0]["content"]


# ---------------------------------------------------------------------------
# 工具链路
# ---------------------------------------------------------------------------

class TestToolChain:

    def test_knowledge_tool_frames(self, client, install_agent, seeded_agent):
        """原生工具：start/end 两帧，展示名走目录映射，轨迹块封成 done"""
        stub = install_agent(
            tool_step(("call_1", "search_knowledge", {"query": "退货政策"})),
            answer_step(ANSWER),
            search_reply="知识库说：七天无理由。",
        )

        frames = stream_frames(client, "退货政策是什么")

        tools = frames_of(frames, "tool")
        assert [t.data["status"] for t in tools] == ["start", "end"]
        assert tools[0].data["name"] == "search_knowledge"
        assert tools[0].data["displayName"] == "知识库检索"
        assert tools[1].data["ok"] is True
        assert tools[1].data["result"] == "知识库说：七天无理由。"
        assert stub.search_calls == ["退货政策"]

        conversation_id = frames_of(frames, "meta")[0].data["conversationId"]
        blocks = _load_messages(client, conversation_id)[1]["blocks"]
        assert [b["kind"] for b in blocks] == ["tool", "answer"]
        assert blocks[0]["name"] == "search_knowledge"
        assert blocks[0]["displayName"] == "知识库检索"
        assert blocks[0]["status"] == "done"
        assert blocks[0]["toolCallId"] == "call_1"
        assert blocks[0]["result"] == "知识库说：七天无理由。"

    def test_unknown_tool_is_reported_as_failed(self, client, install_agent, seeded_agent):
        """模型幻觉出不存在的工具时，结果帧要标失败而不是整条流崩掉"""
        install_agent(
            tool_step(("call_x", "no_such_tool", {})),
            answer_step(ANSWER),
        )

        frames = stream_frames(client, "退货政策是什么")

        end = frames_of(frames, "tool")[-1].data
        assert end["ok"] is False
        assert "未知工具" in end["result"]
        assert frames_of(frames, "finish"), "工具失败不该阻断收尾"

    def test_mcp_tool_bound_from_intent_tree(self, client, install_agent, seeded_agent):
        """MCP 工具：意图树配了 + 注册表有执行器，才会挂进本轮工具集"""
        executor = FakeMcpExecutor("weather_query", description="查天气", reply="北京晴 26 度")
        stub = install_agent(
            tool_step(("call_m", "weather_query", {"city": "北京"})),
            answer_step("北京今天晴，26 度。"),
        )
        stub.mcp_registry.register(executor)

        async def _seed() -> None:
            async with client.db_factory() as db:
                await seed_intent_node(
                    db, "INTENT_WEATHER", kind=2, mcp_tool_id="weather_query", name="天气查询",
                    description="查询指定城市的实时天气",
                )

        run_async(_seed())

        frames = stream_frames(client, "北京天气怎么样")

        tools = frames_of(frames, "tool")
        assert tools[0].data["name"] == "weather_query"
        assert tools[0].data["displayName"] == "天气查询"
        assert tools[-1].data["ok"] is True
        assert executor.calls == [{"city": "北京"}]

        # 工具定义确实下发给了模型，否则它无从知道该调什么
        schema_names = {t["function"]["name"] for t in stub.model.calls[0]["tools"]}
        assert {"search_knowledge", "weather_query"} <= schema_names

    def test_meta_reports_mcp_configured(self, client, install_agent, seeded_agent):
        """控制台靠 meta 决定是否展工具面板：意图树 + 注册表都到位才算配好"""
        stub = install_agent(answer_step(ANSWER))
        stub.mcp_registry.register(FakeMcpExecutor("weather_query"))

        before = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["data"]
        assert before["mcpConfigured"] is False
        assert "mcp-tools" not in before["capabilities"]

        async def _seed() -> None:
            async with client.db_factory() as db:
                await seed_intent_node(db, "INTENT_WEATHER", kind=2, mcp_tool_id="weather_query")

        run_async(_seed())

        after = client.get("/agent/v1/meta", headers=AUTH_HEADERS).json()["data"]
        assert after["mcpConfigured"] is True
        assert "mcp-tools" in after["capabilities"]
        assert after["toolProvider"] == "native + mcp"


# ---------------------------------------------------------------------------
# 异常与边界
# ---------------------------------------------------------------------------

class TestFailurePaths:

    def test_max_iterations_emits_hint_and_falls_back(self, client, install_agent, seeded_agent, monkeypatch):
        """跑满迭代上限：先提示再逼一次总结，答案走终答回落而不是流式增量"""
        monkeypatch.setattr(settings.agent, "max_iters", 2)
        install_agent(
            tool_step(("c1", "search_knowledge", {"query": "退货政策"})),
            tool_step(("c2", "search_knowledge", {"query": "换货政策"})),
            answer_step("综合已有信息：七天内可退货。"),
        )

        frames = stream_frames(client, "退货政策是什么")

        hints = frames_of(frames, "hint")
        assert len(hints) == 1
        assert hints[0].data["code"] == "MAX_ITERATIONS"
        assert hints[0].data["text"] == "已达到最大推理轮次，正在根据已有信息生成回答"

        assert frames_of(frames, "finish"), "迭代上限不是失败，仍要正常收尾"
        # 总结那次调用不带增量回调，内容只能靠 AGENT_RESULT 回落
        assert not _joined(frames, "response")

        conversation_id = frames_of(frames, "meta")[0].data["conversationId"]
        assert _load_messages(client, conversation_id)[1]["content"] == "综合已有信息：七天内可退货。"

    def test_model_failure_ends_stream_without_finish(self, client, install_agent, seeded_agent, monkeypatch):
        """模型炸了：流直接收尾，不发 finish，也不留一条空的助手消息"""
        monkeypatch.setattr(settings.agent, "max_retries", 0)

        async def boom(on_delta: Any) -> ModelCompletion:
            raise RuntimeError("上游模型不可用")

        install_agent(boom)

        frames = stream_frames(client, "退货政策是什么")

        assert event_names(frames) == ["meta"]
        assert not frames_of(frames, "finish")
        assert not frames_of(frames, "done")

        conversation_id = frames_of(frames, "meta")[0].data["conversationId"]
        rows = _load_messages(client, conversation_id)
        # 提问已经落库，答复一个字都没吐就不该留空壳
        assert [r["role"] for r in rows] == ["user"]
        assert _gate_slot(client) is None

    def test_busy_gate_is_rejected_with_200(self, client, install_agent, seeded_agent):
        """运行位被占：拒绝必须走统一信封，前端只看 code 不看 HTTP 状态"""
        install_agent(answer_step(ANSWER))
        client.fake_redis.data[_gate_key()] = "other-task|other-conv"

        resp = client.get("/agent/v1/chat", params={"question": "你好"}, headers=AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] != "0"
        assert body["message"] == "当前会话处理中，请稍后再发起新的对话"
        assert body["data"] is None

    def test_blank_question_is_rejected(self, client, install_agent, seeded_agent):
        install_agent(answer_step(ANSWER))

        body = client.get("/agent/v1/chat", params={"question": "   "}, headers=AUTH_HEADERS).json()

        assert body["code"] != "0"
        assert body["message"] == "问题不能为空"

    def test_unauthenticated_chat_is_rejected(self, client, install_agent, seeded_agent):
        resp = client.get("/agent/v1/chat", params={"question": "你好"})

        assert resp.status_code == 200
        assert resp.json()["code"] == BaseErrorCode.NOT_LOGIN.code


# ---------------------------------------------------------------------------
# 中途停止
# ---------------------------------------------------------------------------

class TestStop:

    def test_stop_interrupts_running_stream(self, client, install_agent, seeded_agent):
        """
        前端在流未结束时点「停止」：另起线程并发发 stop，主线程仍在读流。

        TestClient 会把整个 ASGI 调用跑完才返回响应体，所以中途状态只能
        靠并发请求去触发，这也是浏览器里的真实时序。
        """
        half_answer = "已经生成的半截答案"

        async def slow(on_delta: Any) -> ModelCompletion:
            on_delta("response", half_answer)
            await asyncio.sleep(30)
            return ModelCompletion(text="不会到达这里")

        stub = install_agent(slow)
        outcome: dict[str, Any] = {}

        def _consume() -> None:
            try:
                outcome["frames"] = stream_frames(client, "讲个很长的故事")
            except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回主线程
                outcome["error"] = exc

        worker = threading.Thread(target=_consume, daemon=True)
        worker.start()
        try:
            # 模型被调到就说明后台任务已注册，此时停才停得掉
            _wait_until(lambda: stub.model.rounds >= 1, what="首轮模型调用")
            task_id = (_gate_slot(client) or "").split("|")[0]
            assert task_id

            stop_body = client.post(
                "/agent/v1/stop", params={"taskId": task_id}, headers=AUTH_HEADERS,
            ).json()
            assert stop_body["code"] == "0"

            worker.join(timeout=15)
            assert not worker.is_alive(), "停止后流没有结束"
        finally:
            if worker.is_alive():
                worker.join(timeout=5)

        assert "error" not in outcome, outcome.get("error")
        frames = outcome["frames"]

        assert frames_of(frames, "cancel"), f"缺少 cancel 帧: {event_names(frames)}"
        assert not frames_of(frames, "finish")
        assert not frames_of(frames, "done")

        cancel = frames_of(frames, "cancel")[0].data
        assert cancel["messageStatus"] == AgentMessageStatus.INTERRUPTED.value
        assert cancel["messageId"]

        # 半截答案要留住，用户已经看见的东西不能凭空消失
        assert _joined(frames, "response") == half_answer
        conversation_id = frames_of(frames, "meta")[0].data["conversationId"]
        assistant = _load_messages(client, conversation_id)[1]
        assert assistant["content"] == half_answer
        assert assistant["messageStatus"] == AgentMessageStatus.INTERRUPTED.value
        assert _gate_slot(client) is None

    def test_stop_unknown_task_is_noop(self, client, install_agent, seeded_agent):
        body = client.post(
            "/agent/v1/stop", params={"taskId": "no-such-task"}, headers=AUTH_HEADERS,
        ).json()

        assert body["code"] == "0"
        assert body["data"] is None

    def test_delete_conversation_stops_running_stream(self, client, install_agent, seeded_agent):
        """删会话前先停流，否则流会把已删会话的消息又写回来"""
        conversation_id = "conv-e2e-delete"

        async def slow(on_delta: Any) -> ModelCompletion:
            on_delta("response", "半截")
            await asyncio.sleep(30)
            return ModelCompletion(text="不会到达")

        stub = install_agent(slow)
        outcome: dict[str, Any] = {}

        def _consume() -> None:
            try:
                outcome["frames"] = stream_frames(client, "讲个故事", conversation_id=conversation_id)
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        worker = threading.Thread(target=_consume, daemon=True)
        worker.start()
        try:
            _wait_until(lambda: stub.model.rounds >= 1, what="首轮模型调用")
            body = client.delete(
                f"/agent/v1/conversations/{conversation_id}", headers=AUTH_HEADERS,
            ).json()
            assert body["code"] == "0"

            worker.join(timeout=15)
            assert not worker.is_alive(), "删会话没能停掉正在跑的流"
        finally:
            if worker.is_alive():
                worker.join(timeout=5)

        assert "error" not in outcome, outcome.get("error")
        assert frames_of(outcome["frames"], "cancel")
        assert _gate_slot(client) is None

        listed = client.get("/agent/v1/conversations", headers=AUTH_HEADERS).json()["data"]
        assert conversation_id not in [c["conversationId"] for c in listed]


# ---------------------------------------------------------------------------
# 会话回放
# ---------------------------------------------------------------------------

class TestReplay:

    def test_history_endpoint_replays_tool_trace(self, client, install_agent, seeded_agent):
        """刷新页面后重新拉历史，工具轨迹要能原样还原"""
        question = "退货政策是什么"
        install_agent(
            tool_step(("call_1", "search_knowledge", {"query": question})),
            answer_step(ANSWER),
        )

        conversation_id = frames_of(stream_frames(client, question), "meta")[0].data["conversationId"]
        rows = _load_messages(client, conversation_id)

        assert [r["role"] for r in rows] == ["user", "assistant"]
        tool_block = rows[1]["blocks"][0]
        assert tool_block["at"], "轨迹块缺时间戳，前端时间线会塌"
        assert tool_block["status"] == "done"
        # 空字段不参与序列化，对齐 Java 侧 @JsonInclude(NON_NULL)
        assert "text" not in tool_block
