"""
知识检索门面 — mirrors rag.service.KnowledgeSearchFacade.

Agent 模式下 rag 对外的唯一检索窄口：检索 -> KB_ANSWER 合成，返回可直接引用的答案文本。

Java 侧完整链路是「改写 -> 意图解析 -> 歧义引导 -> 多通道检索 -> 合成」。
本实现保全核心功能：向量检索（pgvector）+ 可选重排 + LLM 合成，
意图树作用域过滤与歧义引导留待 RAG 检索引擎整体移植时接入。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.prompt_service import AgentPromptResolver

logger = logging.getLogger(__name__)

EMPTY_RESULT = "未在知识库中检索到与该问题相关的内容。"

DEFAULT_TOP_K = 8
"""向量召回条数，重排后只留前若干条进合成"""

RERANK_KEEP = 5

MIN_SCORE = 0.2
"""余弦相似度下限，低于此值视作噪声不进上下文"""

CONTEXT_MAX_CHARS = 12_000
"""喂给合成的证据上限，超出按分块顺序截断"""


class KnowledgeSearchFacade:
    """知识检索门面：``search_knowledge`` 工具的唯一后端"""

    def __init__(
        self,
        db: AsyncSession,
        llm: Any | None = None,
        embedding_service: Any | None = None,
        rerank_service: Any | None = None,
        prompt_resolver: AgentPromptResolver | None = None,
    ) -> None:
        """
        :param llm: 需支持 ``async complete(system, user, max_tokens)``，或 None 时直接返回证据
        :param embedding_service: infra.embedding_client.EmbeddingService（同步接口）
        :param rerank_service: infra.rerank_client.RerankService（同步接口），可为 None
        """
        self.db = db
        self.llm = llm
        self.embedding_service = embedding_service
        self.rerank_service = rerank_service
        self.prompt_resolver = prompt_resolver

    async def search(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        """
        检索并合成答案。

        :param history: 主 Agent 会话的近期 user/assistant 轮次，仅用于指代消解
        """
        question = await self._rewrite(query, history or [])

        chunks = await self._vector_search(question, DEFAULT_TOP_K)
        if not chunks:
            return EMPTY_RESULT

        if self.rerank_service is not None:
            chunks = await self._rerank(question, chunks)

        context = self._build_context(chunks)
        if not context.strip():
            return EMPTY_RESULT

        return await self._synthesize(question, context)

    # ------------------------------------------------------------------
    # 改写
    # ------------------------------------------------------------------

    async def _rewrite(self, query: str, history: list[dict[str, str]]) -> str:
        """
        指代消解：有历史才值得改写，没历史时原样返回。

        改写失败一律回落原问句 —— 检索不到总比问错了好。
        """
        if not history or self.llm is None:
            return query
        try:
            transcript = "\n".join(
                f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
                for m in history[-4:]
            )
            rewritten = await self.llm.complete(
                "你是查询改写器。把用户的最新问题改写成不依赖上下文也能独立理解的完整问句，"
                "只输出改写后的问句本身，不要解释。",
                f"近期对话：\n{transcript}\n\n最新问题：{query}\n\n改写后：",
                200,
            )
            result = (rewritten or "").strip()
            return result or query
        except Exception as e:
            logger.warning("查询改写失败，回落原问句: %s", e)
            return query

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    async def _vector_search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        """pgvector 余弦检索；嵌入服务不可用时回落关键词 LIKE，保证工具始终有产出"""
        vector = await self._embed(question)
        if vector is None:
            return await self._keyword_search(question, top_k)

        collections = await self._active_collections()
        if not collections:
            return []

        vec_literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        sql = text("""
            SELECT v.content AS content,
                   v.metadata AS metadata,
                   1 - (v.embedding <=> CAST(:vec AS vector)) AS score
            FROM t_knowledge_vector v
            WHERE v.collection_name = ANY(:collections)
            ORDER BY v.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
        """)
        rows = (await self.db.execute(sql, {
            "vec": vec_literal,
            "collections": collections,
            "top_k": top_k,
        })).mappings().all()

        return [
            self._to_chunk(row)
            for row in rows
            if float(row["score"] or 0) >= MIN_SCORE
        ]

    async def _keyword_search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        """嵌入服务缺失时的兜底：按分块内容做包含匹配"""
        sql = text("""
            SELECT c.content AS content, c.id AS chunk_id, c.kb_id AS kb_id
            FROM t_knowledge_chunk c
            WHERE c.deleted = 0 AND c.enabled = 1 AND c.content ILIKE :pattern
            ORDER BY c.char_count DESC NULLS LAST
            LIMIT :top_k
        """)
        rows = (await self.db.execute(sql, {
            "pattern": f"%{question[:64]}%",
            "top_k": top_k,
        })).mappings().all()

        return [
            {
                "content": str(row["content"] or ""),
                "doc_name": "",
                "kb_id": str(row["kb_id"] or ""),
                "score": 0.5,
            }
            for row in rows
        ]

    async def _embed(self, question: str) -> list[float] | None:
        if self.embedding_service is None:
            return None
        try:
            # EmbeddingService 是同步接口，丢到线程池别堵住事件循环
            vectors = await asyncio.to_thread(self.embedding_service.embed, [question])
            return vectors[0] if vectors else None
        except Exception as e:
            logger.warning("查询嵌入失败，回落关键词检索: %s", e)
            return None

    async def _active_collections(self) -> list[str]:
        """启用中知识库的 collection 名，检索作用域据此收窄"""
        rows = (await self.db.execute(text("""
            SELECT collection_name FROM t_knowledge_base
            WHERE deleted = 0 AND status = 1 AND collection_name IS NOT NULL
        """))).all()
        return [str(r[0]) for r in rows if r[0]]

    @staticmethod
    def _to_chunk(row: Any) -> dict[str, Any]:
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            metadata = {}
        return {
            "content": str(row["content"] or ""),
            "doc_name": str(metadata.get("docName") or metadata.get("doc_name") or ""),
            "kb_id": str(metadata.get("kbId") or metadata.get("kb_id") or ""),
            "score": float(row["score"] or 0),
        }

    async def _rerank(self, question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """重排失败一律回落原序：召回已经够用了，别把整次检索赔进去"""
        try:
            documents = [c["content"] for c in chunks]
            results = await asyncio.to_thread(self.rerank_service.rerank, question, documents)
            reranked = [chunks[r.index] for r in results if 0 <= r.index < len(chunks)]
            return reranked[:RERANK_KEEP] or chunks[:RERANK_KEEP]
        except Exception as e:
            logger.warning("重排失败，回落向量序: %s", e)
            return chunks[:RERANK_KEEP]

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        """证据拼装：带上来源名，合成阶段才好指明出处"""
        parts: list[str] = []
        total = 0
        for index, chunk in enumerate(chunks, start=1):
            content = chunk["content"].strip()
            if not content:
                continue
            source = f"（来源：{chunk['doc_name']}）" if chunk["doc_name"] else ""
            block = f"[{index}]{source}\n{content}"
            if total + len(block) > CONTEXT_MAX_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    async def _synthesize(self, question: str, context: str) -> str:
        """
        合成答案。

        没有 LLM 或提示词解析失败时直接回证据原文 —— 工具结果给主 Agent 消化，
        原文也是可用信息，比报错强。
        """
        if self.llm is None:
            return context

        system_prompt = ""
        if self.prompt_resolver is not None:
            try:
                from app.services.prompt_service import AgentPromptSlot

                system_prompt = await self.prompt_resolver.render(
                    AgentPromptSlot.KB_ANSWER,
                    {"question": question, "kbContext": context},
                )
            except Exception as e:
                logger.warning("KB_ANSWER 提示词解析失败，回落内置模板: %s", e)

        if system_prompt.strip():
            # render 已把槽位填进正文，直接当用户消息发
            try:
                return (await self.llm.complete("", system_prompt, 4096)).strip() or context
            except Exception as e:
                logger.error("知识库答案合成失败，回落证据原文: %s", e)
                return context

        fallback_system = (
            "你是知识库问答助手。只依据给定证据回答问题，证据不足时明确说明；"
            "不要编造，不要提及证据编号。"
        )
        try:
            answer = await self.llm.complete(
                fallback_system,
                f"问题：{question}\n\n证据：\n{context}",
                4096,
            )
            return (answer or "").strip() or context
        except Exception as e:
            logger.error("知识库答案合成失败，回落证据原文: %s", e)
            return context
