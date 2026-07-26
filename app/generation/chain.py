"""RAG Chain - 编排层：缓存 + 检索管道委托 + 生成"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Generator

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config import get_settings, build_chat_llm
from app.ingestion.indexer import HierarchicalIndexer
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from app.retrieval.agent import AgenticRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.parent_child import ParentChildRetriever
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.router import QueryRouter
from app.retrieval.caching import get_semantic_cache, EmbeddingCache
from app.retrieval.conversation import ConversationRewriter
from app.generation.faithfulness import FaithfulnessChecker, regen_until_faithful
from app.generation.citation import CitationBuilder, Citation
from app.generation.answer_boost import AnswerBooster
from app.generation.streaming import speculative_faithful_stream
from app.generation.prompts import (
    RAG_SIMPLE_PROMPT, RAG_CHAT_PROMPT, DIRECT_ANSWER_PROMPT, FALLBACK_RESPONSE,
    STRICT_RAG_PROMPT, STRICT_RAG_CHAT_PROMPT,
)
from app.observability.tracing import get_tracer
from app.observability.metrics import get_metrics

logger = logging.getLogger(__name__)

__all__ = ["RAGChain", "RAGResponse", "RetrievalResult"]


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    sources: List[Document]
    retrieval_result: RetrievalResult
    total_time_ms: float = 0
    cache_hit: bool = False
    # F3 生成忠实度自检观测字段
    faithful: bool | None = None      # True=忠实 / False=含幻觉 / None=未检查或未知
    faithfulness_score: float = 0.0   # 被支撑论断占比 [0,1]
    regenerated: bool = False         # 是否因不忠实触发了严格重生成
    # RAG 3.0 增强观测字段
    citations: List[Citation] = field(default_factory=list)  # F7 引用溯源
    short_answer: str = ""            # F10 抽取的核心短答案
    self_consistency_used: bool = False  # F10 是否触发自一致性
    rewritten_query: str = ""         # F12 历史感知重写后的查询


@dataclass
class _GenOutcome:
    """生成步骤的契约：invoke / invoke_stream 的生成路径都归一为这四个值。"""
    answer: str
    faithful: bool | None = None      # True=忠实 / False=含幻觉 / None=未检查
    fb_score: float = 0.0
    regenerated: bool = False


class RAGChain:
    """
    RAG 编排层：
    [Cache Check] -> RetrievalPipeline.run() -> Generate -> [Cache Put]

    检索全链路（门控/改写/召回/融合/重排/评估/补救）由 RetrievalPipeline 承担。
    """

    def __init__(
        self,
        indexer: HierarchicalIndexer | None = None,
        use_query_transform: bool = True,
        use_rerank: bool = True,
        use_graph: bool = True,
        query_strategy: str = "multi_query",
    ):
        self._settings = get_settings()
        settings = self._settings

        self.indexer = indexer or HierarchicalIndexer()
        self.dense_retriever = DenseRetriever(self.indexer)
        self.sparse_retriever = SparseRetriever(self.indexer)
        self.reranker = Reranker() if use_rerank else None
        self.query_transformer = QueryTransformer() if use_query_transform else None
        self.graph_retriever = (
            GraphRetriever() if (use_graph and settings.graph_enabled) else None
        )

        self.parent_child_retriever: ParentChildRetriever | None = None
        if settings.use_parent_child:
            try:
                self.parent_child_retriever = ParentChildRetriever(
                    embeddings=self.indexer.embeddings
                )
            except Exception as e:
                logger.warning(f"Parent-Child 初始化失败: {e}")

        self.crag_evaluator: CRAGEvaluator | None = None
        if settings.use_crag:
            self.crag_evaluator = CRAGEvaluator()

        self.query_router: QueryRouter | None = (
            QueryRouter() if settings.use_query_router else None
        )

        # F3 生成忠实度自检（幻觉检测）
        self.faithfulness_checker: FaithfulnessChecker | None = (
            FaithfulnessChecker() if settings.use_faithfulness_check else None
        )

        self.semantic_cache = get_semantic_cache() if settings.cache_enabled else None

        # ===== RAG 3.0 增强组件（每项独立开关，异常均优雅降级） =====
        # F9 L1 Embedding 缓存：包装查询编码，省掉重复 embedding
        self.embedding_cache = (
            EmbeddingCache(self.indexer.embeddings) if settings.use_embedding_cache else None
        )
        # F7 引用溯源（零在线 LLM）
        self.citation_builder: CitationBuilder | None = (
            CitationBuilder(embeddings=self.indexer.embeddings)
            if settings.use_citations else None
        )
        # F10 答案质量增强（聚焦/抽取/自一致性）
        self.answer_booster = AnswerBooster()
        # F12 多轮对话记忆 / 历史感知查询重写
        self.conversation_rewriter: ConversationRewriter | None = (
            ConversationRewriter() if settings.use_history_rewrite else None
        )

        self.use_query_transform = use_query_transform
        self.use_rerank = use_rerank
        self.use_graph = use_graph and settings.graph_enabled
        self.query_strategy = query_strategy

        self.llm = build_chat_llm(
            streaming=True,
            # 延迟治理（2026-07-26）：60s×2 → 30s×1，消灭超时/重试叠加的离群尾；
            # max_tokens 封顶防无界长答案拖慢生成
            timeout=30,
            retries=1,
            max_tokens=settings.answer_max_tokens,
        )

        # 检索管道（七阶段，可独立测试）
        self.pipeline = RetrievalPipeline(
            self.indexer,
            self.dense_retriever,
            self.sparse_retriever,
            reranker=self.reranker,
            query_transformer=self.query_transformer,
            graph_retriever=self.graph_retriever,
            parent_child_retriever=self.parent_child_retriever,
            crag_evaluator=self.crag_evaluator,
            query_router=self.query_router,
        )

        # F13 Agentic RAG：ReAct agent 复用管道阶段作为工具；use_agentic 开时启用
        self.agentic_retriever: AgenticRetriever | None = (
            AgenticRetriever(self.pipeline, llm=self.llm)
            if settings.use_agentic else None
        )
        self.pipeline.agentic = self.agentic_retriever

        logger.info(
            f"RAGChain 初始化: query_transform={use_query_transform}, "
            f"rerank={use_rerank}, strategy={query_strategy}, "
            f"parent_child={self.parent_child_retriever is not None}, "
            f"crag={self.crag_evaluator is not None}, "
            f"cache={self.semantic_cache is not None}"
        )

    def rebuild_parent_child_index(self) -> int:
        """F6a：从已索引分块重建 Parent-Child 索引（一次性，用于补齐历史文档）。

        按 doc_id 把分块拼回文档级文本，再交给 ParentChildRetriever 重新切分 parent/child。
        无 parent_child_retriever 或任何异常都返回 0（优雅降级）。
        """
        if not self.parent_child_retriever:
            return 0
        try:
            all_chunks = self.indexer.get_all_chunks()
            grouped: dict[str, Document] = {}
            for ch in all_chunks:
                doc_id = ch.metadata.get("doc_id", "unknown")
                if doc_id not in grouped:
                    grouped[doc_id] = Document(
                        page_content="",
                        metadata={"doc_id": doc_id, "source": ch.metadata.get("source", "")},
                    )
                grouped[doc_id].page_content += "\n" + ch.page_content
            docs = list(grouped.values())
            if docs:
                self.parent_child_retriever.index_documents(docs)
            logger.info(f"F6a Parent-Child 重建: {len(docs)} 个文档")
            return len(docs)
        except Exception as e:
            logger.warning(f"F6a Parent-Child 重建失败: {e}")
            return 0

    def retrieve(
        self, question: str, top_k: int | None = None, trace_id: str | None = None
    ) -> RetrievalResult:
        """执行完整检索流程（委托给 RetrievalPipeline）"""
        return self.pipeline.run(
            question,
            top_k=top_k,
            query_strategy=self.query_strategy,
            use_query_transform=self.use_query_transform,
            use_rerank=self.use_rerank,
            trace_id=trace_id,
        )

    @staticmethod
    def _tokenize_for_match(text: str) -> List[str]:
        """词级分词，用于 query↔句子重合匹配（jieba，对齐 sparse/metrics 过滤口径）。

        中文保留 ≥2 字词、英文/数字保留完整 token；过滤单字噪声（的/了/在 等）。
        """
        import jieba
        import re as _re
        out = []
        for t in jieba.lcut(text.lower()):
            t = t.strip()
            if not t:
                continue
            if _re.fullmatch(r'[a-zA-Z0-9]+(?:\.[0-9]+)?', t):
                out.append(t)
            elif len(t) >= 2 and any('\u4e00' <= c <= '\u9fff' for c in t):
                out.append(t)
        return out

    def compress_context(
        self,
        question: str,
        documents: List[Document],
        max_sentences_per_doc: int = 6,
    ) -> List[Document]:
        """上下文压缩：词级重合抽句（零LLM调用）。

        用 jieba 词级分词计算 query 与每个句子的词重合度，**保留所有与问题相关的
        句子**（重合>0，按原文顺序拼接）；无相关句时整篇保留兜底。早期版本用「整段
        中文 token + 硬留 top-3」，token 粒度过粗抓不到子串/词重合（如「双月十六日」），
        会把答案句误删——实测 11/11 拒答样本的答案句曾被旧版丢弃，是端到端正确率偏低
        （语义 0.61）的生成侧主因。词级分词 + 相关句必留修复该自伤。
        """
        import re as _re
        q_tokens = set(self._tokenize_for_match(question))
        if not q_tokens:
            return documents

        compressed = []
        for doc in documents:
            sentences = _re.split(r'(?<=[。！？.!?\n])', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                compressed.append(doc)
                continue

            # 每个句子与 query 的词重合度
            scored = [
                (len(q_tokens & set(self._tokenize_for_match(sent))), sent)
                for sent in sentences
            ]
            relevant = [(sc, s) for sc, s in scored if sc > 0]
            if not relevant:
                # 无相关句：整篇保留兜底（宁可不压缩，也不盲删潜在答案）
                compressed.append(doc)
                continue

            # 相关句超上限时按重合度取 top-N（其余全留）；kept 维持原文顺序
            if len(relevant) > max_sentences_per_doc:
                keep_set = {s for _, s in sorted(relevant, key=lambda x: -x[0])[:max_sentences_per_doc]}
            else:
                keep_set = {s for _, s in relevant}
            kept = [s for s in sentences if s in keep_set]

            compressed.append(Document(
                page_content="".join(kept), metadata=doc.metadata.copy(),
            ))

        return compressed

    def generate(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
        strict: bool = False,
    ) -> str:
        """基于检索结果生成回答。strict=True 时用更严格的 prompt（F3 重生成）。"""
        if not documents:
            return FALLBACK_RESPONSE

        documents = self.compress_context(question, documents)
        context = self._format_context(documents)

        if chat_history:
            template = STRICT_RAG_CHAT_PROMPT if strict else RAG_CHAT_PROMPT
            chain = template | self.llm | StrOutputParser()
            return chain.invoke({
                "context": context, "question": question,
                "chat_history": chat_history,
            })
        template = STRICT_RAG_PROMPT if strict else RAG_SIMPLE_PROMPT
        chain = template | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
    ) -> Generator[str, None, None]:
        """流式生成回答"""
        if not documents:
            yield FALLBACK_RESPONSE
            return

        documents = self.compress_context(question, documents)
        context = self._format_context(documents)
        template = RAG_CHAT_PROMPT if chat_history else RAG_SIMPLE_PROMPT
        payload = {"context": context, "question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        chain = template | self.llm | StrOutputParser()
        for chunk in chain.stream(payload):
            yield chunk

    def generate_direct(self, question: str, chat_history: List | None = None) -> str:
        """门控判定无需检索时的直接回答（无上下文）"""
        chain = DIRECT_ANSWER_PROMPT | self.llm | StrOutputParser()
        payload = {"question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        return chain.invoke(payload)

    def generate_direct_stream(
        self, question: str, chat_history: List | None = None
    ) -> Generator[str, None, None]:
        """门控跳过时的流式直接回答"""
        chain = DIRECT_ANSWER_PROMPT | self.llm | StrOutputParser()
        payload = {"question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        for chunk in chain.stream(payload):
            yield chunk

    def _generate_faithful(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
        deadline=None,
    ) -> tuple[str, bool | None, float, bool]:
        """
        F3 生成 + 忠实度自检 + 有界严格重生成（invoke / invoke_stream 共用）。

        流程：生成答案 → 忠实度检查 → 若不忠实则用 strict prompt 重生成，
        最多 faithfulness_max_regen 次（成本兜底）。检查器关闭或异常时放行。
        check+regen 循环体复用 faithfulness.regen_until_faithful（F8 流式同源）。

        Returns:
            (answer, faithful, faithfulness_score, regenerated)
        """
        answer = self.generate(question, documents, chat_history)
        if not self.faithfulness_checker:
            return answer, None, 0.0, False
        return regen_until_faithful(
            self.faithfulness_checker, question, documents, answer,
            produce_fn=lambda: self.generate(
                question, documents, chat_history, strict=True
            ),
            max_regen=self._settings.faithfulness_max_regen,
            deadline=deadline,
        )

    def invoke(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """完整 RAG 调用：[缓存检查] + 检索 + 生成 + [缓存写入]（自动记录 Trace）"""
        tracer = get_tracer()
        metrics = get_metrics()
        trace_id = tracer.start_trace(question)
        start_time = time.time()
        metrics.inc("requests_total", endpoint="chat", mode="invoke")

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            metrics.inc("cache_hits_total", level="semantic")
            metrics.observe("request_latency_ms", cached.total_time_ms)
            return cached

        rewritten_query, retrieval_result = self._rewrite_and_retrieve(
            question, chat_history, top_k, trace_id
        )

        tracer.start_span(trace_id, "generation")
        if retrieval_result.gate_skipped:
            outcome = _GenOutcome(answer=self.generate_direct(question, chat_history))
        elif self.faithfulness_checker and retrieval_result.documents:
            outcome = _GenOutcome(*self._generate_faithful(
                question, retrieval_result.documents, chat_history,
                deadline=retrieval_result.deadline,
            ))
        else:
            # 无文档（召回为空）或自检关闭：直接生成，不做忠实度校验
            outcome = _GenOutcome(
                answer=self.generate(question, retrieval_result.documents, chat_history)
            )
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(outcome.answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
            "faithful": outcome.faithful,
            "faithfulness_score": outcome.fb_score,
            "regenerated": outcome.regenerated,
        })

        return self._finalize(
            question, chat_history, q_embedding, outcome,
            retrieval_result, rewritten_query, trace_id, start_time,
        )

    def invoke_stream(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        流式 RAG 调用（含缓存检查 + 自动记录 Trace）。

        Yields:
            {"type": "cache_hit", "data": RAGResponse}
            {"type": "retrieving", "data": str}
            {"type": "retrieval", "data": RetrievalResult}
            {"type": "token", "data": str}
            {"type": "correction", "data": str}   # F8: 不忠实时的严格重生成答案
            {"type": "done", "data": RAGResponse}
        """
        tracer = get_tracer()
        metrics = get_metrics()
        trace_id = tracer.start_trace(question)
        start_time = time.time()
        metrics.inc("requests_total", endpoint="chat", mode="stream")

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            metrics.inc("cache_hits_total", level="semantic")
            metrics.observe("request_latency_ms", cached.total_time_ms)
            yield {"type": "cache_hit", "data": cached}
            yield {"type": "done", "data": cached}
            return

        # SSE 顺序约束：retrieving 事件必须先于检索发出，故检索步骤在事件之间调用
        yield {"type": "retrieving", "data": "正在检索相关文档..."}
        rewritten_query, retrieval_result = self._rewrite_and_retrieve(
            question, chat_history, top_k, trace_id
        )
        yield {"type": "retrieval", "data": retrieval_result}

        tracer.start_span(trace_id, "generation")
        full_answer = ""
        faithful, fb_score, regenerated = None, 0.0, False
        use_speculative = (
            self._settings.use_speculative_streaming
            and self.faithfulness_checker
            and retrieval_result.documents
            and not retrieval_result.gate_skipped
        )
        if retrieval_result.gate_skipped:
            for token in self.generate_direct_stream(question, chat_history):
                full_answer += token
                yield {"type": "token", "data": token}
        elif use_speculative:
            # F8 投机流式：先逐 token 吐字（快 TTFT），流末自检，不忠实追加 correction
            docs = retrieval_result.documents
            for event in speculative_faithful_stream(
                stream_fn=lambda: self.generate_stream(question, docs, chat_history),
                question=question,
                documents=docs,
                chat_history=chat_history,
                checker=self.faithfulness_checker,
                regen_fn=lambda: self.generate(question, docs, chat_history, strict=True),
                max_regen=self._settings.faithfulness_max_regen,
                deadline=retrieval_result.deadline,  # 修复：流式路径同受时延预算约束
            ):
                if event["type"] == "token":
                    full_answer += event["data"]
                    yield {"type": "token", "data": event["data"]}
                elif event["type"] == "correction":
                    full_answer = event["data"]  # 最终答案以重生成结果为准
                    yield {"type": "correction", "data": event["data"]}
                elif event["type"] == "final":
                    d = event["data"]
                    full_answer = d["answer"]
                    faithful, fb_score, regenerated = d["faithful"], d["score"], d["regenerated"]
        elif self.faithfulness_checker and retrieval_result.documents:
            # 投机流式关闭：回退旧的阻塞式（先非流式生成+自检，再整体输出）
            full_answer, faithful, fb_score, regenerated = self._generate_faithful(
                question, retrieval_result.documents, chat_history,
                deadline=retrieval_result.deadline,
            )
            yield {"type": "token", "data": full_answer}
        else:
            for token in self.generate_stream(
                question, retrieval_result.documents, chat_history
            ):
                full_answer += token
                yield {"type": "token", "data": token}
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(full_answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
            "faithful": faithful,
            "faithfulness_score": fb_score,
            "regenerated": regenerated,
            "speculative": use_speculative,
        })

        outcome = _GenOutcome(full_answer, faithful, fb_score, regenerated)
        yield {"type": "done", "data": self._finalize(
            question, chat_history, q_embedding, outcome,
            retrieval_result, rewritten_query, trace_id, start_time,
        )}

    # ---- RAG 3.0 增强辅助方法（F7/F9/F10/F12，异常均优雅降级） ----

    def _embed_query(self, question: str):
        """F9 L1：查询编码，优先走 embedding 缓存。"""
        if self.embedding_cache is not None:
            return self.embedding_cache.embed_query(question)
        return self.indexer.embeddings.embed_query(question)

    def _rewrite_query(self, question: str, chat_history: List | None) -> tuple[str, str]:
        """F12：带历史时重写查询。返回 (检索用查询, 重写后的查询字符串或"")。"""
        if not (self.conversation_rewriter and chat_history):
            return question, ""
        try:
            rewritten = self.conversation_rewriter.rewrite(question, chat_history)
            if rewritten and rewritten != question:
                return rewritten, rewritten
        except Exception as e:
            logger.warning(f"F12 查询重写失败，沿用原问题: {e}")
        return question, ""

    def _rewrite_and_retrieve(
        self,
        question: str,
        chat_history: List | None,
        top_k: int | None,
        trace_id: str,
    ) -> tuple[str, RetrievalResult]:
        """检索步骤（invoke / invoke_stream 共用）：F12 重写 → 完整检索管道。

        Returns:
            (重写后的查询或 "", 检索结果)
        """
        retrieval_q, rewritten_query = self._rewrite_query(question, chat_history)
        retrieval_result = self.retrieve(retrieval_q, top_k=top_k, trace_id=trace_id)
        return rewritten_query, retrieval_result

    def _build_citations(
        self, question: str, answer: str, documents: List[Document]
    ) -> List[Citation]:
        """F7：构建引用溯源。任何异常返回空列表。"""
        if not self.citation_builder:
            return []
        try:
            return self.citation_builder.build(question, answer, documents)
        except Exception as e:
            logger.warning(f"F7 引用溯源失败: {e}")
            return []

    def _boost_answer(self, question: str, answer: str, query_type: str):
        """F10：答案增强（抽取短答案 + 可选自一致性）。异常返回 None。"""
        try:
            return self.answer_booster.boost(question, answer, query_type=query_type)
        except Exception as e:
            logger.warning(f"F10 答案增强失败: {e}")
            return None

    def _finalize(
        self,
        question: str,
        chat_history: List | None,
        q_embedding: np.ndarray | None,
        outcome: _GenOutcome,
        retrieval_result: RetrievalResult,
        rewritten_query: str,
        trace_id: str,
        start_time: float,
    ) -> RAGResponse:
        """收尾步骤（invoke / invoke_stream 共用）：
        F7 引用 + F10 增强 + trace 收尾 + 缓存写入 + F11 指标 + 响应组装。
        """
        citations = self._build_citations(
            question, outcome.answer, retrieval_result.documents
        )
        boost = self._boost_answer(
            question, outcome.answer, retrieval_result.query_type
        )
        short_answer = boost.short_answer if boost else ""
        sc_used = bool(boost and boost.self_consistency_used)

        total_time = (time.time() - start_time) * 1000
        get_tracer().end_trace(trace_id, answer_preview=outcome.answer)
        self._write_cache(
            question, chat_history, q_embedding, outcome.answer,
            retrieval_result.documents,
        )

        # F11 指标：时延 / 忠实度分布
        metrics = get_metrics()
        metrics.observe("request_latency_ms", total_time)
        if outcome.faithful is True:
            metrics.inc("faithfulness_total", verdict="faithful")
        elif outcome.faithful is False:
            metrics.inc("faithfulness_total", verdict="unfaithful")

        return RAGResponse(
            answer=outcome.answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
            faithful=outcome.faithful,
            faithfulness_score=outcome.fb_score,
            regenerated=outcome.regenerated,
            citations=citations,
            short_answer=short_answer,
            self_consistency_used=sc_used,
            rewritten_query=rewritten_query,
        )

    # ---- 缓存公共逻辑（invoke / invoke_stream 共用） ----

    def _check_cache(
        self,
        question: str,
        chat_history: List | None,
        trace_id: str,
        start_time: float,
    ) -> tuple[RAGResponse | None, np.ndarray | None]:
        """
        语义缓存检查。

        Returns:
            (命中时的 RAGResponse 或 None, 查询 embedding 或 None)
        """
        tracer = get_tracer()
        q_embedding = None
        if not (self.semantic_cache and not chat_history):
            return None, None
        try:
            q_embedding = np.array(self._embed_query(question))
            cached = self.semantic_cache.get(q_embedding)
            if cached:
                tracer.start_span(trace_id, "cache_hit")
                tracer.end_span(trace_id, "cache_hit", {
                    "question": cached.question[:50],
                })
                total_time = (time.time() - start_time) * 1000
                tracer.end_trace(trace_id, answer_preview=cached.answer)
                source_docs = [
                    Document(
                        page_content=s.get("content", ""),
                        metadata={"source": s.get("source", "")},
                    )
                    for s in cached.sources
                ]
                return RAGResponse(
                    answer=cached.answer,
                    sources=source_docs,
                    retrieval_result=RetrievalResult(documents=source_docs),
                    total_time_ms=total_time,
                    cache_hit=True,
                ), q_embedding
        except Exception as e:
            logger.warning(f"缓存检查失败: {e}")
        return None, q_embedding

    def _write_cache(
        self,
        question: str,
        chat_history: List | None,
        q_embedding: np.ndarray | None,
        answer: str,
        documents: List[Document],
    ) -> None:
        """语义缓存写入（复用已计算的 embedding；带对话历史时不写）"""
        if not self.semantic_cache or chat_history:
            return
        try:
            embedding = q_embedding
            if embedding is None:
                embedding = np.array(self._embed_query(question))
            sources_data = [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", ""),
                }
                for doc in documents
            ]
            self.semantic_cache.put(question, embedding, answer, sources_data)
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")

    def _format_context(self, documents: List[Document]) -> str:
        """格式化检索文档为上下文字符串"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"[文档 {i}] (来源: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)
