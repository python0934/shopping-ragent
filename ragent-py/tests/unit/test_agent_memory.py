"""
Phase 5 单元测试 — Agent 消息模型与记忆层.

覆盖 messages / memory / compaction / state_store 四个模块。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.compaction import (
    AgentContextCompactionMiddleware,
    AgentContextCompactor,
    AgentConversationSummarizer,
)
from app.agent.enums import MsgRole, ToolResultState
from app.agent.memory import (
    TRIM_UNCHANGED,
    AgentContextTrimmer,
    AgentMemoryBudget,
)
from app.agent.messages import (
    AgentContextChars,
    Msg,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    block_from_dict,
    block_to_dict,
)
from app.agent.state_store import ANONYMOUS_USER, CONTEXT_STATE_KEY, PgAgentStateStore, _safe_user


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _result_block(text: str, call_id: str = "call-1", name: str = "search_knowledge") -> ToolResultBlock:
    return ToolResultBlock(id=call_id, name=name, output=[TextBlock(text=text)])


def _tool_cycle(query: str, result: str, call_id: str = "call-1") -> list[Msg]:
    """一轮完整的工具调用：assistant 发起 + tool 回填"""
    return [
        Msg(role=MsgRole.ASSISTANT, content=[
            ToolUseBlock(id=call_id, name="search_knowledge", input={"query": query}),
        ]),
        Msg(role=MsgRole.TOOL, content=[_result_block(result, call_id)]),
    ]


def _budget(**overrides: Any) -> AgentMemoryBudget:
    base: dict[str, Any] = {
        "enabled": True,
        "context_window_chars": 10_000,
        "summary_enabled": True,
        "evictable_tools": ("search_knowledge",),
    }
    base.update(overrides)
    return AgentMemoryBudget(**base)


# ---------------------------------------------------------------------------
# messages — Msg / ContentBlock
# ---------------------------------------------------------------------------

class TestMessages:

    def test_text_block_default_type(self):
        assert TextBlock(text="hi").type == "text"

    def test_msg_static_constructors(self):
        assert Msg.system("p").role == MsgRole.SYSTEM
        assert Msg.user("q").get_text_content() == "q"

        assistant = Msg.assistant(text="a", thinking="t")
        assert assistant.get_text_content() == "a"
        assert assistant.get_thinking_content() == "t"

    def test_assistant_without_thinking_has_no_thinking_block(self):
        assert Msg.assistant(text="only").get_content_blocks(ThinkingBlock) == []

    def test_tool_result_msg_role(self):
        msg = Msg.tool_result(_result_block("r"))
        assert msg.role == MsgRole.TOOL
        assert msg.get_content_blocks(ToolResultBlock)[0].output_text() == "r"

    def test_get_content_blocks_filters_by_type(self):
        msg = Msg(role=MsgRole.ASSISTANT, content=[
            TextBlock(text="a"),
            ToolUseBlock(id="1", name="n", input={}),
            TextBlock(text="b"),
        ])
        assert len(msg.get_content_blocks(TextBlock)) == 2
        assert len(msg.get_content_blocks(ToolUseBlock)) == 1

    def test_get_text_content_joins_all_text_blocks(self):
        msg = Msg(role=MsgRole.ASSISTANT, content=[TextBlock(text="a"), TextBlock(text="b")])
        assert msg.get_text_content() == "ab"

    def test_with_content_returns_new_object_keeping_identity(self):
        origin = Msg.user("q")
        origin.name = "sentinel"
        origin.timestamp = "2026-01-01T00:00:00"

        replaced = origin.with_content([TextBlock(text="new")])
        assert replaced is not origin
        assert replaced.id == origin.id
        assert replaced.role == origin.role
        assert replaced.name == "sentinel"
        assert replaced.timestamp == "2026-01-01T00:00:00"
        assert replaced.get_text_content() == "new"
        # 原对象不受影响
        assert origin.get_text_content() == "q"

    def test_tool_result_output_text_concatenates(self):
        block = ToolResultBlock(
            id="1", name="n",
            output=[TextBlock(text="a"), TextBlock(text="b"), ThinkingBlock(thinking="skip")],
        )
        # 多段输出换行拼接，思考块不算工具正文
        assert block.output_text() == "a\nb"

    def test_block_dict_roundtrip(self):
        for block in (
            TextBlock(text="t"),
            ThinkingBlock(thinking="th"),
            ToolUseBlock(id="i", name="n", input={"k": "v"}),
            _result_block("out", "i", "n"),
        ):
            restored = block_from_dict(block_to_dict(block))
            assert restored is not None
            assert type(restored) is type(block)

    def test_block_from_dict_unknown_type_returns_none(self):
        assert block_from_dict({"type": "not_a_block"}) is None

    def test_tool_result_block_dict_preserves_state_and_metadata(self):
        block = ToolResultBlock(
            id="i", name="n", output=[TextBlock(text="x")],
            metadata={"k": 1}, state=ToolResultState.ERROR,
        )
        restored = block_from_dict(block_to_dict(block))
        assert isinstance(restored, ToolResultBlock)
        assert restored.state == ToolResultState.ERROR
        assert restored.metadata == {"k": 1}

    def test_msg_dict_roundtrip(self):
        msg = Msg(role=MsgRole.ASSISTANT, content=[
            ThinkingBlock(thinking="th"),
            ToolUseBlock(id="i", name="n", input={"q": "1"}),
        ])
        msg.name = "sentinel"
        msg.timestamp = "2026-01-01T00:00:00"

        restored = Msg.from_dict(msg.to_dict())
        assert restored.id == msg.id
        assert restored.role == MsgRole.ASSISTANT
        assert restored.name == "sentinel"
        assert restored.timestamp == "2026-01-01T00:00:00"
        assert restored.get_thinking_content() == "th"
        assert restored.get_content_blocks(ToolUseBlock)[0].input == {"q": "1"}

    def test_msg_from_dict_tolerates_string_role(self):
        assert Msg.from_dict({"role": "user", "content": [{"type": "text", "text": "hi"}]}).role == MsgRole.USER

    def test_msg_to_json_is_parseable(self):
        import json

        assert json.loads(Msg.user("q").to_json())["role"] == "user"


class TestAgentContextChars:

    def test_total_sums_all_messages(self):
        context = [Msg.user("abc"), Msg.assistant(text="de")]
        assert AgentContextChars.total(context) == 5

    def test_total_of_empty_and_none(self):
        assert AgentContextChars.total([]) == 0
        assert AgentContextChars.total(None) == 0

    def test_of_msg_counts_thinking_and_tool_io(self):
        msg = Msg(role=MsgRole.ASSISTANT, content=[
            TextBlock(text="ab"),
            ThinkingBlock(thinking="cd"),
            ToolUseBlock(id="i", name="n", input={"k": "ef"}),
        ])
        # 文本 2 + 思考 2 + 入参序列化后含 "ef"
        assert AgentContextChars.of_msg(msg) >= 6

    def test_of_msg_none_is_zero(self):
        assert AgentContextChars.of_msg(None) == 0

    def test_of_output_counts_tool_result_text(self):
        assert AgentContextChars.of_output(_result_block("12345")) == 5


# ---------------------------------------------------------------------------
# memory — AgentMemoryBudget
# ---------------------------------------------------------------------------

class TestAgentMemoryBudget:

    def test_thresholds_derive_from_context_window(self):
        budget = _budget(context_window_chars=10_000)
        assert budget.trim_trigger_chars == 5_000
        assert budget.compact_trigger_chars == 8_000
        assert budget.keep_recent_chars == 2_000

    def test_keep_recent_cycles_at_least_one(self):
        assert _budget().keep_recent_cycles >= 1

    def test_summary_max_chars_clamped_to_floor(self):
        # 120000 * 0.1 = 12000 > 6000，取上限
        assert _budget(context_window_chars=1_200_000).summary_max_chars == 6_000

    def test_summary_max_chars_clamped_to_ceiling(self):
        # 1000 * 0.1 = 100 < 1500，取下限
        assert _budget(context_window_chars=1_000).summary_max_chars == 1_500

    def test_summary_max_chars_within_bounds(self):
        assert _budget(context_window_chars=30_000).summary_max_chars == 3_000

    def test_from_settings_reads_config(self):
        budget = AgentMemoryBudget.from_settings()
        assert budget.context_window_chars > 0
        assert budget.enabled in (True, False)


# ---------------------------------------------------------------------------
# memory — AgentContextTrimmer
# ---------------------------------------------------------------------------

class TestAgentContextTrimmer:

    def test_trim_below_threshold_is_noop(self):
        context = [Msg.user("q"), *_tool_cycle("q", "short")]
        trimmer = AgentContextTrimmer(_budget(context_window_chars=100_000))

        result = trimmer.trim_in_place(context)
        assert result is TRIM_UNCHANGED
        assert not result.changed
        assert context[2].get_content_blocks(ToolResultBlock)[0].output_text() == "short"

    def test_trim_none_and_empty_context(self):
        trimmer = AgentContextTrimmer(_budget())
        assert trimmer.trim_in_place(None) is TRIM_UNCHANGED
        assert trimmer.trim_in_place([]) is TRIM_UNCHANGED

    def test_trim_evicts_old_tool_results_over_threshold(self):
        # 窗口 1000 -> 触发线 500；三段各 400 字符的工具结果，最近两段受保护
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context += _tool_cycle("q2", "c" * 400, "call-2")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        result = trimmer.trim_in_place(context)

        assert result.changed
        assert result.reclaimed_chars > 0

        evicted = context[2].get_content_blocks(ToolResultBlock)[0].output_text()
        assert evicted.startswith(AgentContextTrimmer.EVICTED_PREFIX)
        assert "400" in evicted
        assert evicted.endswith(AgentContextTrimmer.EVICTED_SUFFIX)

    def test_trim_keeps_recent_cycles_untouched(self):
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        trimmer.trim_in_place(context)

        # 最后一段工具结果必须原样保留，否则本轮推理就没了证据
        last_result = context[-2].get_content_blocks(ToolResultBlock)[0].output_text()
        assert last_result == "b" * 400

    def test_trim_preserves_block_identity_fields(self):
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        trimmer.trim_in_place(context)

        evicted = context[2].get_content_blocks(ToolResultBlock)[0]
        assert evicted.id == "call-0"
        assert evicted.name == "search_knowledge"
        assert evicted.state == ToolResultState.SUCCESS

    def test_trim_skips_tools_outside_whitelist(self):
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        # 非白名单工具的大结果不许动
        context.append(Msg(role=MsgRole.ASSISTANT, content=[
            ToolUseBlock(id="call-x", name="write_file", input={"p": "x"}),
        ]))
        context.append(Msg(role=MsgRole.TOOL, content=[_result_block("z" * 400, "call-x", "write_file")]))
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        trimmer.trim_in_place(context)

        assert context[4].get_content_blocks(ToolResultBlock)[0].output_text() == "z" * 400

    def test_trim_is_idempotent(self):
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context += _tool_cycle("q2", "c" * 400, "call-2")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        first = trimmer.trim_in_place(context)
        second = trimmer.trim_in_place(context)

        assert first.changed
        # 已清理过的块不再算候选，第二轮不该又「回收」一次
        assert second.reclaimed_chars == 0

    def test_trim_replacements_pair_origin_with_rebuilt(self):
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context += _tool_cycle("q2", "c" * 400, "call-2")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        result = trimmer.trim_in_place(context)

        assert len(result.replacements) >= 1
        for origin, rebuilt in result.replacements:
            assert origin is not rebuilt
            assert origin.id == rebuilt.id

    def test_trim_evicted_preview_clips_long_input(self):
        # 四段循环：近两段受保护，前两段才是候选；结果取 800 字符确保过「至少清掉 20%」的底线
        context: list[Msg] = [Msg.user("q0")]
        context.append(Msg(role=MsgRole.ASSISTANT, content=[
            ToolUseBlock(id="call-0", name="search_knowledge", input={"query": "x" * 500}),
        ]))
        context.append(Msg(role=MsgRole.TOOL, content=[_result_block("a" * 800, "call-0")]))
        context += _tool_cycle("q1", "b" * 800, "call-1")
        context += _tool_cycle("q2", "c" * 800, "call-2")
        context += _tool_cycle("q3", "d" * 800, "call-3")
        context.append(Msg.user("latest"))

        trimmer = AgentContextTrimmer(_budget(context_window_chars=1_000))
        result = trimmer.trim_in_place(context)
        assert result.changed

        evicted = context[2].get_content_blocks(ToolResultBlock)[0].output_text()
        assert evicted.startswith(AgentContextTrimmer.EVICTED_PREFIX)
        assert AgentContextTrimmer.EVICTED_INPUT in evicted
        # 入参预览被截到 EVICTED_INPUT_MAX_CHARS，不会把 500 字符原样塞回占位串
        assert len(evicted) < 800
        assert "x" * (AgentContextTrimmer.EVICTED_INPUT_MAX_CHARS + 1) not in evicted


# ---------------------------------------------------------------------------
# compaction — 安全切点与孤儿检测
# ---------------------------------------------------------------------------

class _FakeLlm:
    """摘要模型桩：按脚本返回，记录调用次数"""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0

    async def complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        return "## 结论\n摘要正文"


class _FakePromptResolver:
    def __init__(self, template: str = "请摘要：{material}") -> None:
        self.template = template

    async def resolve(self, slot: Any) -> str:
        return self.template

    async def render(self, slot: Any, slots: dict[str, str] | None = None) -> str:
        text = self.template
        for key, value in (slots or {}).items():
            text = text.replace("{" + key + "}", value)
        return text


class TestAgentContextCompactor:

    def test_cutoff_only_lands_on_user_turn_start(self):
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()),
            _budget(context_window_chars=1_000),
        )
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 300, "call-0")
        context.append(Msg.user("q1"))
        context += _tool_cycle("q1", "b" * 300, "call-1")
        context.append(Msg.user("q2"))
        context += _tool_cycle("q2", "c" * 300, "call-2")

        cutoff = compactor._find_safe_cutoff(context, keep_recent_chars=400)
        # 切点必须落在 user 消息上，否则会把半轮工具调用劈开
        assert cutoff > 0
        assert context[cutoff].role == MsgRole.USER

    def test_cutoff_returns_minus_one_when_no_user_boundary_exists(self):
        """尾部保留量之外再无用户轮起点，宁可放弃压缩也不能劈开工具配对"""
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()),
            _budget(),
        )
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 300, "call-0")

        assert compactor._find_safe_cutoff(context, keep_recent_chars=10_000) == -1

    def test_cutoff_returns_minus_one_when_nothing_can_be_dropped(self):
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()),
            _budget(),
        )
        assert compactor._find_safe_cutoff([Msg.user("only")], keep_recent_chars=10_000) <= 0

    def test_orphan_tool_result_detected(self):
        orphan = [Msg(role=MsgRole.TOOL, content=[_result_block("r", "call-x")])]
        assert AgentContextCompactor._has_orphan_tool_result(orphan)

    def test_paired_tool_result_is_not_orphan(self):
        paired = [
            Msg(role=MsgRole.ASSISTANT, content=[
                ToolUseBlock(id="call-x", name="n", input={}),
            ]),
            Msg(role=MsgRole.TOOL, content=[_result_block("r", "call-x")]),
        ]
        assert not AgentContextCompactor._has_orphan_tool_result(paired)

    def test_summary_msg_is_marked_and_unwrappable(self):
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()),
            _budget(),
        )
        summary = compactor._build_summary_msg("摘要正文")

        assert AgentContextCompactor._is_summary(summary)
        assert summary.role == MsgRole.USER
        assert AgentContextCompactor._unwrap(summary) == "摘要正文"
        # 头部说明必须写清这是背景而非新指令，否则模型会当成用户又问了一遍
        assert AgentContextCompactor.SUMMARY_HEADER in summary.get_text_content()

    def test_non_summary_msg_is_not_marked(self):
        assert not AgentContextCompactor._is_summary(Msg.user("q"))

    @pytest.mark.asyncio
    async def test_compact_skips_when_material_too_small(self):
        """摘要本身也占额度，换出的原文不够多就不值得压"""
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()),
            _budget(context_window_chars=100_000),
        )
        context = [Msg.user("q"), Msg.assistant(text="a")]
        assert not await compactor.compact_in_place(context, "u1", "s1")

    @pytest.mark.asyncio
    async def test_compact_replaces_head_with_summary(self):
        llm = _FakeLlm(["## 结论\n甲\n\n## 依据\n乙\n\n## 待办\n丙"])
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(llm, _FakePromptResolver()),
            _budget(context_window_chars=1_000),
        )
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 700, "call-0")
        context.append(Msg.user("q1"))
        context += _tool_cycle("q1", "b" * 300, "call-1")
        context.append(Msg.user("latest"))

        before = AgentContextChars.total(context)
        assert await compactor.compact_in_place(context, "u1", "s1")

        assert AgentContextCompactor._is_summary(context[0])
        assert context[-1].get_text_content() == "latest"
        assert AgentContextChars.total(context) < before
        # 保留段的工具配对不能被压缩破坏
        assert not AgentContextCompactor._has_orphan_tool_result(context)

    @pytest.mark.asyncio
    async def test_compact_folds_previous_summary_into_new_one(self):
        """已有摘要必须当素材喂回去，否则每代都从头摘，早期信息逐代蒸发"""
        llm = _FakeLlm(["## 结论\n甲\n\n## 依据\n乙\n\n## 待办\n丙"])
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(llm, _FakePromptResolver()),
            _budget(context_window_chars=1_000),
        )
        context: list[Msg] = [compactor._build_summary_msg("## 结论\n旧摘要")]
        context.append(Msg.user("q0"))
        context += _tool_cycle("q0", "a" * 700, "call-0")
        context.append(Msg.user("q1"))
        context += _tool_cycle("q1", "b" * 300, "call-1")
        context.append(Msg.user("latest"))

        assert await compactor.compact_in_place(context, "u1", "s1")
        # 旧摘要被吸收，只剩一份摘要消息
        assert sum(1 for m in context if AgentContextCompactor._is_summary(m)) == 1


# ---------------------------------------------------------------------------
# compaction — 摘要器
# ---------------------------------------------------------------------------

class TestAgentConversationSummarizer:

    def _summarizer(self, replies: list[str] | None = None) -> tuple[AgentConversationSummarizer, _FakeLlm]:
        llm = _FakeLlm(replies)
        return AgentConversationSummarizer(llm, _FakePromptResolver(), _budget()), llm

    def test_render_transcript_includes_roles_and_text(self):
        summarizer, _ = self._summarizer()
        material = [Msg.user("问题一"), Msg.assistant(text="答案一")]

        transcript = summarizer.render_transcript(material)
        assert "问题一" in transcript
        assert "答案一" in transcript

    def test_render_transcript_excludes_thinking(self):
        """思考过程是中间态，进摘要会被当成结论复述"""
        summarizer, _ = self._summarizer()
        material = [Msg.assistant(text="可见", thinking="内心独白")]

        transcript = summarizer.render_transcript(material)
        assert "可见" in transcript
        assert "内心独白" not in transcript

    def test_render_transcript_includes_tool_calls(self):
        summarizer, _ = self._summarizer()
        material = _tool_cycle("查一下", "查到了")

        transcript = summarizer.render_transcript(material)
        assert "search_knowledge" in transcript
        assert "查到了" in transcript

    def test_validate_rejects_blank(self):
        summarizer, _ = self._summarizer()
        assert summarizer.validate("   ", 6_000) is None

    def test_validate_rejects_too_few_sections(self):
        summarizer, _ = self._summarizer()
        assert summarizer.validate("只有一句话没有分节", 6_000) is None

    def test_validate_accepts_well_formed_summary(self):
        summarizer, _ = self._summarizer()
        summary = "## 结论\n甲\n\n## 依据\n乙\n\n## 待办\n丙"
        assert summarizer.validate(summary, 6_000) == summary

    def test_validate_clips_at_section_boundary_when_over_budget(self):
        summarizer, _ = self._summarizer()
        summary = "## A\n" + "a" * 100 + "\n\n## B\n" + "b" * 100 + "\n\n## C\n" + "c" * 100

        clipped = summarizer.validate(summary, 260)
        assert clipped is not None
        assert len(clipped) <= 260

    def test_count_sections(self):
        summarizer, _ = self._summarizer()
        assert summarizer._count_sections("## A\nx\n## B\ny") == 2
        assert summarizer._count_sections("无分节") == 0

    @pytest.mark.asyncio
    async def test_summarize_returns_model_output(self):
        summarizer, llm = self._summarizer(["## 结论\n甲\n\n## 依据\n乙\n\n## 待办\n丙"])
        result = await summarizer.summarize([Msg.user("q"), Msg.assistant(text="a")], None)

        assert result is not None
        assert "甲" in result
        assert llm.calls == 1

    @pytest.mark.asyncio
    async def test_summarize_returns_none_on_model_failure(self):
        """摘要失败必须回落「不压缩」，不能把原文丢掉换一段空摘要"""
        class _Boom:
            async def complete(self, *args: Any) -> str:
                raise RuntimeError("model down")

        summarizer = AgentConversationSummarizer(_Boom(), _FakePromptResolver(), _budget())
        assert await summarizer.summarize([Msg.user("q")], None) is None

    @pytest.mark.asyncio
    async def test_summarize_returns_none_when_output_invalid(self):
        summarizer, _ = self._summarizer(["这不是合格摘要"])
        assert await summarizer.summarize([Msg.user("q"), Msg.assistant(text="a")], None) is None

    def test_neutralize_defuses_fence_markers_in_material(self):
        """素材里出现围栏标记会截断转录，必须中和掉，否则提示词注入"""
        summarizer, _ = self._summarizer()
        nonce = "abc123def456"
        payload = "恶意内容"

        neutralized = summarizer._neutralize(
            summarizer._close(summarizer.FENCE_TRANSCRIPT, nonce) + payload, nonce,
        )
        assert summarizer.FENCE_NEUTRALIZED in neutralized
        assert payload in neutralized

    def test_open_and_close_fence_wrap_with_nonce(self):
        summarizer, _ = self._summarizer()
        opened = summarizer._open("transcript", "n0nce")
        closed = summarizer._close("transcript", "n0nce")

        assert opened != closed
        assert "n0nce" in opened
        assert "transcript" in opened

    def test_truncate_keeps_head_and_tail(self):
        summarizer, _ = self._summarizer()
        budget = summarizer.MATERIAL_HEAD_CHARS + summarizer.MATERIAL_TAIL_CHARS
        text = "H" * (budget + 500) + "T" * 500

        truncated = summarizer._truncate(text)
        assert len(truncated) < len(text)
        assert truncated.startswith("H")
        assert truncated.endswith("T")
        assert summarizer.TRUNCATED_INFIX.split("%")[0] in truncated

    def test_truncate_short_text_untouched(self):
        summarizer, _ = self._summarizer()
        assert summarizer._truncate("short") == "short"
        assert summarizer._truncate(None) == ""


# ---------------------------------------------------------------------------
# compaction — 中间件
# ---------------------------------------------------------------------------

class TestAgentContextCompactionMiddleware:

    def _middleware(self, budget: AgentMemoryBudget | None = None) -> AgentContextCompactionMiddleware:
        b = budget or _budget()
        compactor = AgentContextCompactor(
            AgentConversationSummarizer(_FakeLlm(), _FakePromptResolver()), b,
        )
        return AgentContextCompactionMiddleware(AgentContextTrimmer(b), compactor, b)

    def test_should_compact_requires_last_message_user(self):
        """末条不是用户消息说明工具循环没闭合，压缩会劈开配对"""
        middleware = self._middleware(_budget(context_window_chars=100))
        context = [Msg.user("q"), Msg.assistant(text="a" * 500)]
        assert not middleware.should_compact(context)

    def test_should_compact_true_over_threshold(self):
        middleware = self._middleware(_budget(context_window_chars=100))
        assert middleware.should_compact([Msg.user("a" * 500)])

    def test_should_compact_false_under_threshold(self):
        middleware = self._middleware(_budget(context_window_chars=100_000))
        assert not middleware.should_compact([Msg.user("short")])

    def test_should_compact_false_on_empty_context(self):
        assert not self._middleware().should_compact([])

    def test_trim_delegates_to_trimmer(self):
        middleware = self._middleware(_budget(context_window_chars=100_000))
        assert middleware.trim([Msg.user("q")]) is TRIM_UNCHANGED

    def test_trim_swallows_trimmer_errors(self):
        """裁剪炸了不能连累整轮推理，退回原上下文继续跑"""
        middleware = self._middleware()

        class _Boom:
            def trim_in_place(self, context: Any) -> Any:
                raise RuntimeError("trim failed")

        middleware.trimmer = _Boom()  # type: ignore[assignment]
        assert middleware.trim([Msg.user("q")]) is TRIM_UNCHANGED

    @pytest.mark.asyncio
    async def test_before_reasoning_trims_when_below_compact_threshold(self):
        middleware = self._middleware(_budget(context_window_chars=1_000))
        context: list[Msg] = [Msg.user("q0")]
        context += _tool_cycle("q0", "a" * 400, "call-0")
        context += _tool_cycle("q1", "b" * 400, "call-1")
        context += _tool_cycle("q2", "c" * 400, "call-2")
        context.append(Msg.user("latest"))

        await middleware.before_reasoning(context, "u1", "s1")
        evicted = context[2].get_content_blocks(ToolResultBlock)[0].output_text()
        assert evicted.startswith(AgentContextTrimmer.EVICTED_PREFIX)

    @pytest.mark.asyncio
    async def test_before_reasoning_noop_on_small_context(self):
        middleware = self._middleware(_budget(context_window_chars=100_000))
        context = [Msg.user("q"), Msg.assistant(text="a")]

        assert not await middleware.before_reasoning(context, "u1", "s1")
        assert len(context) == 2


# ---------------------------------------------------------------------------
# state_store
# ---------------------------------------------------------------------------

class _FakeResult:
    """对齐 SQLAlchemy Result 的常用取数口"""

    def __init__(self, rows: list[Any] | None = None, scalar: Any = None) -> None:
        self._rows = rows or []
        self._scalar = scalar

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list[Any]:
        return self._rows

    def scalar_one_or_none(self) -> Any:
        return self._scalar

    def first(self) -> Any:
        if self._rows:
            return self._rows[0]
        return (self._scalar,) if self._scalar is not None else None


class _FakeBind:
    """让 state_store 走生产路径（PG 原子 UPSERT）而非 SQLite 兜底"""

    class _Dialect:
        name = "postgresql"

    dialect = _Dialect()


class _FakeSession:
    """记录 execute 调用并按脚本回值"""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = list(results or [])
        self.statements: list[Any] = []
        self.added: list[Any] = []

    async def execute(self, statement: Any, params: Any = None) -> Any:
        self.statements.append(statement)
        return self.results.pop(0) if self.results else _FakeResult()

    def get_bind(self) -> Any:
        return _FakeBind()

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        pass


class TestStateStore:

    def test_anonymous_user_sentinel(self):
        assert ANONYMOUS_USER == "__anon__"
        assert CONTEXT_STATE_KEY == "context"

    def test_safe_user_falls_back_to_anonymous(self):
        assert _safe_user(None) == ANONYMOUS_USER
        assert _safe_user("") == ANONYMOUS_USER
        assert _safe_user("  ") == ANONYMOUS_USER
        assert _safe_user("u1") == "u1"

    def test_to_payload_uses_to_dict_when_available(self):
        assert PgAgentStateStore._to_payload(Msg.user("q"))["role"] == "user"

    def test_to_payload_recurses_into_lists(self):
        payload = PgAgentStateStore._to_payload([Msg.user("a"), Msg.assistant(text="b")])
        assert isinstance(payload, list)
        assert payload[0]["role"] == "user"
        assert payload[1]["role"] == "assistant"

    def test_to_payload_passes_through_plain_values(self):
        assert PgAgentStateStore._to_payload({"k": 1}) == {"k": 1}
        assert PgAgentStateStore._to_payload("raw") == "raw"

    @pytest.mark.asyncio
    async def test_get_list_returns_rows(self):
        session = _FakeSession([_FakeResult(scalar=[{"role": "user"}])])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert await store.get_list("u1", "s1", CONTEXT_STATE_KEY) == [{"role": "user"}]

    @pytest.mark.asyncio
    async def test_get_list_empty_when_payload_not_a_list(self):
        session = _FakeSession([_FakeResult(scalar={"k": 1})])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert await store.get_list("u1", "s1", CONTEXT_STATE_KEY) == []

    @pytest.mark.asyncio
    async def test_get_list_empty_when_no_row(self):
        session = _FakeSession([_FakeResult(scalar=None)])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert await store.get_list("u1", "s1", CONTEXT_STATE_KEY) == []

    @pytest.mark.asyncio
    async def test_list_session_ids_dedupes_keeping_order(self):
        session = _FakeSession([_FakeResult(rows=[("s1",), ("s2",), ("s1",), ("s3",), ("s2",)])])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert await store.list_session_ids("u1") == ["s1", "s2", "s3"]

    @pytest.mark.asyncio
    async def test_save_and_delete_issue_statements(self):
        session = _FakeSession()
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        await store.save("u1", "s1", CONTEXT_STATE_KEY, {"a": 1})
        await store.delete("u1", "s1")
        await store.delete_key("u1", "s1", CONTEXT_STATE_KEY)

        assert len(session.statements) == 3

    @pytest.mark.asyncio
    async def test_exists_reports_presence(self):
        session = _FakeSession([_FakeResult(rows=[("context",)])])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert await store.exists("u1", "s1")

    @pytest.mark.asyncio
    async def test_exists_false_when_absent(self):
        session = _FakeSession([_FakeResult(rows=[])])
        store = PgAgentStateStore(session)  # type: ignore[arg-type]

        assert not await store.exists("u1", "s1")
