"""
智能体提示词服务 — 槽位元数据、模板工具、Redis 缓存与解析器。

Mirrors Java ``rag.core.prompt``:
  - PromptTemplateUtils     -> PromptTemplateUtils
  - AgentPromptSlot         -> AgentPromptSlot
  - AgentPromptCacheManager -> AgentPromptCacheManager
  - AgentPromptResolver     -> AgentPromptResolver
  - rag.config.OrchestrationMode -> OrchestrationMode

放在 services 而非 agent 包下：WorkFlow（RAG）与 Agent 两种编排都要读同一批槽位，
Java 侧也归属 rag 模块。
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentProfileDO, AgentPromptDO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OrchestrationMode — mirrors rag.config.OrchestrationMode
# ---------------------------------------------------------------------------

class OrchestrationMode(str, Enum):
    """编排架构：WorkFlow 固定管线 / Agent ReAct 自主决策"""

    WORKFLOW = "workflow"
    AGENT = "agent"


# ---------------------------------------------------------------------------
# PromptTemplateUtils — mirrors rag.core.prompt.PromptTemplateUtils
# ---------------------------------------------------------------------------

class PromptTemplateUtils:
    """提示词模板工具：清格式、填占位、切分节"""

    _MULTI_BLANK_LINES = re.compile(r"(\n){3,}")
    _SECTION_HEADER = re.compile(r"^---\s*section:\s*(\S+)\s*---$", re.MULTILINE)

    @staticmethod
    def cleanup_prompt(prompt: str | None) -> str:
        """三个以上连续换行压成两个，再去首尾空白"""
        if prompt is None:
            return ""
        return PromptTemplateUtils._MULTI_BLANK_LINES.sub("\n\n", prompt).strip()

    @staticmethod
    def fill_slots(template: str | None, slots: dict[str, str] | None) -> str:
        """``{key}`` 形式的占位符逐个替换，值为 None 当空串"""
        if template is None:
            return ""
        if not slots:
            return template
        result = template
        for key, value in slots.items():
            result = result.replace("{" + key + "}", value or "")
        return result

    @staticmethod
    def parse_sections(content: str | None) -> dict[str, str]:
        """把含 ``--- section: name ---`` 分隔符的模板解析为 name -> content"""
        sections: dict[str, str] = {}
        if not content or not content.strip():
            return sections

        matches = list(PromptTemplateUtils._SECTION_HEADER.finditer(content))
        last_start = -1
        last_name: str | None = None

        for match in matches:
            if last_name is not None:
                sections[last_name] = PromptTemplateUtils._trim_section(
                    content[last_start: match.start()]
                )
            last_name = match.group(1)
            last_start = match.end()

        if last_name is not None:
            sections[last_name] = PromptTemplateUtils._trim_section(content[last_start:])

        return sections

    @staticmethod
    def _trim_section(section: str) -> str:
        """去掉 section 内容的首尾空行，但保留内部结构"""
        if section.startswith("\n"):
            section = section[1:]
        return section.rstrip()


# ---------------------------------------------------------------------------
# AgentPromptSlot — mirrors rag.core.prompt.AgentPromptSlot
# ---------------------------------------------------------------------------

class PromptSlotGroup(str, Enum):
    """控制台分栏，按生效范围而非历史归属划分"""

    WORKFLOW = "WorkFlow专属"
    AGENT = "Agent专属"
    COMMON = "通用"


class AgentPromptSlot(str, Enum):
    """
    智能体提示词槽位，是槽位元数据的唯一权威源。

    槽位按功能命名而非按架构命名：生效范围会随 v1/v2 演进变化，不编码进标识符。
    枚举值即 t_agent_prompt.slot_key，与 Java ``name()`` 一致（全大写）。
    """

    SYSTEM_CHAT = "SYSTEM_CHAT"
    MCP_ANSWER = "MCP_ANSWER"
    MIXED_ANSWER = "MIXED_ANSWER"
    AGENT_MAIN = "AGENT_MAIN"
    KNOWLEDGE_TOOL_DESCRIPTION = "KNOWLEDGE_TOOL_DESCRIPTION"
    AGENT_CONTEXT_COMPACTION = "AGENT_CONTEXT_COMPACTION"
    KB_ANSWER = "KB_ANSWER"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"
    RECOMMENDED_QUESTIONS = "RECOMMENDED_QUESTIONS"

    # -- 元数据 -------------------------------------------------------------

    @property
    def display_name(self) -> str:
        return _SLOT_META[self][0]

    @property
    def group(self) -> PromptSlotGroup:
        return _SLOT_META[self][1]

    @property
    def effective_modes(self) -> frozenset[OrchestrationMode]:
        return _SLOT_META[self][2]

    @property
    def inactive_reason(self) -> str | None:
        """未生效时展示给管理员的原因，两种架构都生效的槽位为 None"""
        return _SLOT_META[self][3]

    @property
    def required_placeholders(self) -> frozenset[str]:
        """必须出现的占位符，缺失会让下游规则静默失效，故在保存时拒绝"""
        return _SLOT_META[self][4]

    @property
    def editor_hint(self) -> str | None:
        """编辑器上方的写法提醒，仅当用法不同于普通提示词时才写"""
        return _SLOT_META[self][5]

    # -- 行为 ---------------------------------------------------------------

    def is_effective_in(self, mode: OrchestrationMode) -> bool:
        return mode in self.effective_modes

    @classmethod
    def effective_in(cls, mode: OrchestrationMode) -> list[AgentPromptSlot]:
        """当前架构下真正会被读取的槽位，控制台拿它当覆盖率的分母"""
        return [slot for slot in cls if slot.is_effective_in(mode)]

    @classmethod
    def find(cls, key: str | None) -> AgentPromptSlot | None:
        if not key:
            return None
        for slot in cls:
            if slot.value.lower() == key.strip().lower():
                return slot
        return None


_WORKFLOW_ONLY = frozenset({OrchestrationMode.WORKFLOW})
_AGENT_ONLY = frozenset({OrchestrationMode.AGENT})
_BOTH_MODES = frozenset({OrchestrationMode.WORKFLOW, OrchestrationMode.AGENT})

# (displayName, group, effectiveModes, inactiveReason, requiredPlaceholders, editorHint)
_SLOT_META: dict[AgentPromptSlot, tuple[str, PromptSlotGroup, frozenset[OrchestrationMode],
                                        str | None, frozenset[str], str | None]] = {
    AgentPromptSlot.SYSTEM_CHAT: (
        "闲聊应答", PromptSlotGroup.WORKFLOW, _WORKFLOW_ONLY,
        "Agent 模式下由主 Agent 直接应答", frozenset(), None,
    ),
    AgentPromptSlot.MCP_ANSWER: (
        "MCP数据应答", PromptSlotGroup.WORKFLOW, _WORKFLOW_ONLY,
        "Agent 模式下改用原生工具调用，无独立的数据合成环节", frozenset(), None,
    ),
    AgentPromptSlot.MIXED_ANSWER: (
        "混合来源应答", PromptSlotGroup.WORKFLOW, _WORKFLOW_ONLY,
        "Agent 模式下由主 Agent 综合多个工具的结果", frozenset(), None,
    ),
    AgentPromptSlot.AGENT_MAIN: (
        "Agent人设", PromptSlotGroup.AGENT, _AGENT_ONLY,
        "WorkFlow 模式不经过 ReAct 架构", frozenset(), None,
    ),
    # 唯一一个不进对话消息的槽位：它随工具定义下发，模型在调用前就要读懂
    AgentPromptSlot.KNOWLEDGE_TOOL_DESCRIPTION: (
        "知识库工具声明", PromptSlotGroup.AGENT, _AGENT_ONLY,
        "WorkFlow 模式不注册原生知识库工具", frozenset(),
        "模型靠它判断要不要查知识库，此时还看不到检索结果；写清这个库覆盖哪类问题，不必在这里规定回答风格",
    ),
    AgentPromptSlot.AGENT_CONTEXT_COMPACTION: (
        "Agent上下文压缩", PromptSlotGroup.AGENT, _AGENT_ONLY,
        "WorkFlow 模式不做上下文压缩，长会话走「历史对话摘要」",
        frozenset({"{summary_max_chars}"}),
        "产物会以历史消息的身份回填进后续每一轮，而被它替代的原文届时已经删除；"
        "要求写清调用过哪些工具、得到什么结论、还剩什么没做，不必在这里规定回答风格",
    ),
    # 两种架构共用：WorkFlow 下由主链路合成，Agent 下由 RAG Tool 内部合成
    AgentPromptSlot.KB_ANSWER: (
        "知识库应答", PromptSlotGroup.COMMON, _BOTH_MODES,
        None, frozenset(), None,
    ),
    # 与「Agent 上下文压缩」不可合并：这份是话题索引不含结论，那份必须留结论
    AgentPromptSlot.CONVERSATION_SUMMARY: (
        "历史对话摘要", PromptSlotGroup.WORKFLOW, _WORKFLOW_ONLY,
        "Agent 模式的长会话改由「Agent 上下文压缩」承接",
        frozenset({"{summary_max_chars}"}), None,
    ),
    AgentPromptSlot.RECOMMENDED_QUESTIONS: (
        "追问推荐", PromptSlotGroup.COMMON, _BOTH_MODES,
        None, frozenset({"{chunks}", "{count}", "{question}", "{answer}"}), None,
    ),
}


# ---------------------------------------------------------------------------
# AgentPromptCacheManager — mirrors rag.core.prompt.AgentPromptCacheManager
# ---------------------------------------------------------------------------

class AgentPromptCacheManager:
    """
    智能体提示词缓存管理器。

    缓存的是激活智能体叠加自定义提示词之后的结果，命中即可直接取用。
    """

    CACHE_KEY = "ragent:agent:resolved-prompts:v2"
    """缓存结构随提示词槽位集合变化时递增版本，避免旧缓存缺少新增槽位"""

    CACHE_EXPIRE_SECONDS = 3600

    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def get_from_cache(self) -> dict[str, str] | None:
        """槽位到提示词的映射，缓存不存在或解析失败返回 None"""
        try:
            raw = await self.redis.get(self.CACHE_KEY)
            if raw is None:
                return None
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.error("从 Redis 读取智能体提示词缓存失败: %s", e)
            return None

    async def save_to_cache(self, prompts: dict[str, str]) -> None:
        try:
            await self.redis.set(
                self.CACHE_KEY,
                json.dumps(prompts, ensure_ascii=False),
                ex=self.CACHE_EXPIRE_SECONDS,
            )
        except Exception as e:
            logger.error("保存智能体提示词到 Redis 缓存失败: %s", e)

    async def clear_cache(self) -> None:
        """任何智能体或槽位写操作后必须调用，否则改动直到过期才生效"""
        try:
            await self.redis.delete(self.CACHE_KEY)
            logger.info("智能体提示词缓存已清除")
        except Exception as e:
            logger.error("清除智能体提示词缓存失败: %s", e)


# ---------------------------------------------------------------------------
# AgentPromptResolver — mirrors rag.core.prompt.AgentPromptResolver
# ---------------------------------------------------------------------------

class AgentPromptResolver:
    """
    智能体提示词解析器。

    优先取激活智能体的槽位，空白则回落内置智能体；面向终端用户的提示词一律从此处读取。
    """

    def __init__(self, db: AsyncSession, cache_manager: AgentPromptCacheManager | None = None) -> None:
        self.db = db
        self.cache_manager = cache_manager

    async def resolve(self, slot: AgentPromptSlot | None) -> str:
        """槽位提示词，内置智能体也没配时返回空串"""
        if slot is None:
            return ""
        resolved = await self.resolve_all()
        return resolved.get(slot.value) or ""

    async def render(self, slot: AgentPromptSlot, slots: dict[str, str] | None = None) -> str:
        """填充占位符并清理格式，语义与 PromptTemplateLoader#render 一致"""
        template = await self.resolve(slot)
        return PromptTemplateUtils.cleanup_prompt(PromptTemplateUtils.fill_slots(template, slots))

    async def resolve_all(self) -> dict[str, str]:
        """全部槽位的最终生效内容，缺失的槽位不出现在 map 中"""
        if self.cache_manager is not None:
            cached = await self.cache_manager.get_from_cache()
            if cached is not None:
                return cached

        resolved = await self._load_from_db()

        if self.cache_manager is not None:
            await self.cache_manager.save_to_cache(resolved)
        return resolved

    async def load_own_prompts(self, agent_id: str | None) -> dict[str, str]:
        """读取某个智能体自身配置的槽位，不做回落，供控制台编辑态展示"""
        own: dict[str, str] = {}
        if not agent_id or not agent_id.strip():
            return own

        result = await self.db.execute(
            select(AgentPromptDO).where(
                AgentPromptDO.agent_id == agent_id,
                AgentPromptDO.deleted == 0,
            )
        )
        for prompt in result.scalars().all():
            own[prompt.slot_key] = prompt.content or ""
        return own

    async def _load_from_db(self) -> dict[str, str]:
        builtin = await self._first_by_flag("builtin")
        if builtin is None:
            logger.warning("未找到内置智能体，空槽位将无提示词可回落")

        # 先铺内置作为基线，再让激活智能体的非空槽位覆盖；两者同为一条时重复覆盖无副作用
        resolved: dict[str, str] = {}
        await self._put_non_blank(resolved, builtin)
        await self._put_non_blank(resolved, await self._first_by_flag("active"))
        return resolved

    async def _first_by_flag(self, flag: str) -> AgentProfileDO | None:
        result = await self.db.execute(
            select(AgentProfileDO)
            .where(getattr(AgentProfileDO, flag) == 1, AgentProfileDO.deleted == 0)
            .order_by(AgentProfileDO.create_time.asc(), AgentProfileDO.id.asc())
        )
        return result.scalars().first()

    async def _put_non_blank(self, target: dict[str, str], profile: AgentProfileDO | None) -> None:
        """后写入者覆盖前者，空白内容不参与覆盖，以此实现回落"""
        if profile is None:
            return
        own = await self.load_own_prompts(profile.id)
        for slot_key, content in own.items():
            if content and content.strip():
                target[slot_key] = content
