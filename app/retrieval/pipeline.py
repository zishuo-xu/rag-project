"""检索管道 - 门控/改写/多路召回/融合/重排/评估/补救 七阶段"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from config import get_settings
from app.retrieval.autocut import autocut_truncate
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.deadline import Deadline
from app.retrieval.fusion import reciprocal_rank_fusion, chunk_key, dedup_by_chunk_id
from app.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

ALL_CHANNELS = ("dense", "sparse", "graph", "parent_child", "summary")


@dataclass
class RetrievalResult:
    """检索结果封装"""
    documents: List[Document]
    dense_results: List[Document] = field(default_factory=list)
    sparse_results: List[Document] = field(default_factory=list)
    graph_results: List[Document] = field(default_factory=list)
    summary_results: List[Document] = field(default_factory=list)
    fused_results: List[Document] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0
    crag_grade: str = ""    # correct / ambiguous / incorrect / recovered
    crag_action: str = ""   # 采取的动作描述
    gate_skipped: bool = False  # 门控判定无需检索
    # RAG 2.0 观测字段
    pre_autocut_count: int = 0       # F1: Autocut 截断前的候选数（降噪幅度）
    query_type: str = ""             # F4: 路由判定的查询类型
    iterations_used: int = 0         # F2: 迭代检索实际迭代次数
    iterative_stop_reason: str = ""  # F2: sufficient / converged / max_iterations / disabled
    # F6 答案定位增强观测字段
    decomposed_subqueries: List[str] = field(default_factory=list)  # F6b: 分解出的子问题
    decomposition_chain: bool = False                              # F6b: 是否依赖链
    # F13 Agentic RAG 观测字段
    agent_steps: List[dict] = field(default_factory=list)  # F13: ReAct 决策轨迹
    agent_stop_reason: str = ""                            # F13: agent_done / max_steps / decision_error
    # 延迟治理观测字段（2026-07-26）
    budget_skipped: List[str] = field(default_factory=list)  # 时延预算熔断跳过的阶段
    deadline: object = None                                  # Deadline 实例（供生成层 F3 复用）


class RetrievalPipeline:
    """
    检索管道：[门控] -> 改写 -> 五路召回 -> RRF融合 -> 重排 -> CRAG评估 -> [补救]

    每个阶段是独立方法，可单独测试；remediate() 复用 recall/fuse/rerank
    组成完整 mini-pipeline，保证补救结果同样经过融合与精排。
    """

    def __init__(
        self,
        indexer,
        dense_retriever,
        sparse_retriever,
        *,
        reranker=None,
        query_transformer=None,
        graph_retriever=None,
        parent_child_retriever=None,
        crag_evaluator=None,
        query_router=None,
        settings=None,
        agentic=None,
    ):
        self.indexer = indexer
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.graph_retriever = graph_retriever
        self.parent_child_retriever = parent_child_retriever
        self.crag_evaluator = crag_evaluator
        self.query_router = query_router
        self._settings = settings or get_settings()
        # F13 Agentic RAG（ReAct agent，由 chain 在构建后注入；use_agentic 开时启用）
        self.agentic = agentic

    # ---- 阶段 1: 门控 ----

    def gate(self, question: str) -> tuple[bool, str]:
        """判断是否需要检索。失败时默认检索（保守方向，不漏检索）"""
        if not (self._settings.use_crag_gate and self.crag_evaluator):
            return True, "门控未启用"
        try:
            return self.crag_evaluator.should_retrieve(question)
        except Exception as e:
            logger.warning(f"门控判断失败，默认检索: {e}")
            return True, f"门控异常: {e}"

    # ---- 阶段 2: 查询改写 ----

    def transform(
        self, question: str, strategy: str, use_query_transform: bool = True
    ) -> List[str]:
        if use_query_transform and self.query_transformer:
            return self.query_transformer.transform(question, strategy)
        return [question]

    # ---- 阶段 3: 多路召回 ----

    def recall(
        self,
        question: str,
        queries: List[str],
        top_n: int | None = None,
        channels=ALL_CHANNELS,
        trace_id: str | None = None,
    ) -> dict:
        """
        并行执行各召回路，结果按 channel 聚合。

        Returns:
            {channel: [Document, ...]}，channels 中每个 key 都存在
        """
        settings = self._settings
        top_n = top_n or settings.rerank_top_n
        results: dict[str, List[Document]] = {c: [] for c in channels}

        # 批量预计算 query embedding，分发给 dense 各查询变体（每变体省 1 次编码）
        query_embeddings: list = [None] * len(queries)
        if "dense" in channels and len(queries) > 1:
            try:
                query_embeddings = self.indexer.embeddings.embed_documents(queries)
            except Exception as e:
                logger.debug(f"批量 embedding 失败，回退到逐个计算: {e}")

        def _dense(q, emb):
            return self.dense_retriever.retrieve(q, top_k=top_n, embedding=emb)

        def _sparse(q):
            return self.sparse_retriever.retrieve(q, top_k=top_n)

        def _graph():
            if self.graph_retriever:
                return self.graph_retriever.retrieve(question, top_k=3)
            return []

        def _pc():
            if self.parent_child_retriever and self.parent_child_retriever.has_index():
                return self.parent_child_retriever.retrieve(question, top_k=3)
            return []

        def _summary():
            return self.indexer.hierarchical_search(
                question, top_k=settings.retrieval_top_k
            )

        with ThreadPoolExecutor(max_workers=settings.recall_max_workers) as executor:
            futures = {}
            if "dense" in channels:
                for i, q in enumerate(queries):
                    futures[executor.submit(_dense, q, query_embeddings[i])] = "dense"
            if "sparse" in channels:
                for q in queries:
                    futures[executor.submit(_sparse, q)] = "sparse"
            if "graph" in channels:
                futures[executor.submit(_graph)] = "graph"
            if "parent_child" in channels:
                futures[executor.submit(_pc)] = "parent_child"
            if "summary" in channels and settings.use_summary_recall:
                futures[executor.submit(_summary)] = "summary"

            for future in as_completed(futures):
                channel = futures[future]
                try:
                    results[channel].extend(future.result())
                except Exception as e:
                    logger.warning(f"{channel} 召回失败: {e}")

        for ch in ("dense", "sparse"):
            if ch in results:
                results[ch] = dedup_by_chunk_id(results[ch])
        return results

    # ---- 阶段 4: RRF 融合 ----

    def fuse(self, recall_results: dict) -> List[Document]:
        fusion_inputs = [
            recall_results.get("dense", []),
            recall_results.get("sparse", []),
        ]
        for ch in ("graph", "parent_child", "summary"):
            if recall_results.get(ch):
                fusion_inputs.append(recall_results[ch])
        return reciprocal_rank_fusion(fusion_inputs)

    # ---- 阶段 5: 重排 ----

    def rerank(
        self,
        question: str,
        documents: List[Document],
        top_k: int,
        use_rerank: bool = True,
        use_autocut: bool = False,
        autocut_min_docs: int = 2,
    ) -> List[Document]:
        if use_rerank and self.reranker:
            # 重排预筛（延迟治理）：cross-encoder 只精排 RRF 融合分 top-N（N=rerank_top_n），
            # 不对全部融合候选（5 路各召回 rerank_top_n，去重后 ~50-70 篇）逐一打分——CPU
            # 上全量打分是 ~3s 的最大单点。documents 已由 RRF 按融合分降序排列（fusion 保证），
            # 截前 N 即 top-N；autocut 膝点与最终 top_k 均 ≪ N，预筛不损失最终召回
            # （答案恒在 RRF 头部，answer_in_top_context 实测保持 1.0）。
            candidates = documents[: self._settings.rerank_top_n]
            if use_autocut:
                # 打分+排序预筛候选（CrossEncoder 计算量不变，仅多返回），
                # 再用 Kneedle 膝点动态截断噪声尾巴，上界 top_k 保证不扩容。
                scored = self.reranker.rerank(
                    question, candidates, top_k=len(candidates)
                )
                return autocut_truncate(
                    scored, top_k=top_k, min_docs=autocut_min_docs,
                )
            return self.reranker.rerank(question, candidates, top_k=top_k)
        return documents[:top_k]

    # ---- 复合原语: 召回 → RRF 融合 → 重排（检索最小闭环） ----

    def search(
        self,
        question: str,
        queries: List[str],
        top_k: int,
        channels=ALL_CHANNELS,
        use_autocut: bool = False,
        autocut_min_docs: int = 2,
        trace_id: str | None = None,
    ) -> List[Document]:
        """召回 → 融合 → 重排 的复合原语，返回最终文档。

        供只需最终文档的路径共用（补救 / 迭代补召 / agent 检索）；
        主路径 run() 需要逐通道中间结果填充 trace 与观测字段，继续内联三阶段。
        """
        recall_results = self.recall(
            question, queries, top_n=top_k, channels=channels, trace_id=trace_id,
        )
        return self.rerank(
            question, self.fuse(recall_results), top_k,
            use_autocut=use_autocut, autocut_min_docs=autocut_min_docs,
        )

    # ---- 阶段 6: CRAG 评估（数字型问题走零 LLM 快速路径） ----

    def evaluate(
        self, question: str, documents: List[Document]
    ) -> tuple[str, List[int], str]:
        if not documents:
            return "incorrect", [], "无检索结果"
        if not CRAGEvaluator.validate_numeric_answer(question, documents):
            return "incorrect", [], "数字型问题但检索结果缺少数字信息（零LLM校验）"
        return self.crag_evaluator.evaluate_relevance(question, documents)

    # ---- 阶段 7: 补救（完整 mini-pipeline：HyDE + 双路召回 + RRF + 重排） ----

    def remediate(
        self, question: str, top_k: int, trace_id: str | None = None
    ) -> List[Document]:
        if not self.query_transformer:
            return []
        hyde_queries = self.query_transformer.transform(question, "hyde")
        return self.search(
            question, hyde_queries, top_k,
            channels=("dense", "sparse"), trace_id=trace_id,
        )

    # ---- 阶段 7b: Self-RAG 迭代检索（F2，质量驱动终止） ----

    def _iterative_retrieve(
        self,
        question: str,
        initial_docs: List[Document],
        top_k: int,
        grade: str,
        reason: str,
        trace_id: str | None = None,
    ) -> tuple[List[Document], int, str, str]:
        """
        Self-RAG 风格迭代检索：证据不足时精化查询、补充召回，直到满足终止条件。

        终止判断标志（质量驱动，非资源/token 兜底）：
        ① 充分性 sufficient：CRAG 评为 correct（证据已足以回答）
        ② 收敛性 converged：本轮精化召回不到任何新增相关文档（精化已无效）
        ③ 安全兜底 max_iterations：max_retrieval_iterations 硬上限（仅防失控）

        Returns:
            (final_docs, iterations_used, stop_reason, final_grade)
        """
        settings = self._settings
        accumulated = list(initial_docs)
        acc_ids = {chunk_key(d) for d in accumulated}
        iterations_used = 0
        stop_reason = "max_iterations"

        for _ in range(settings.max_retrieval_iterations):
            # ① 充分性终止
            if grade == "correct":
                stop_reason = "sufficient"
                break

            # 精化查询（基于已有证据 + 缺口理由）
            refined_q = self._refine_query(question, accumulated, reason)
            new_ranked = self.search(
                question, [refined_q], top_k,
                channels=("dense", "sparse"),
                use_autocut=settings.use_autocut,
                autocut_min_docs=settings.autocut_min_docs,
                trace_id=trace_id,
            )

            # ② 收敛性终止：无新增相关文档
            new_docs = [d for d in new_ranked if chunk_key(d) not in acc_ids]
            if not new_docs:
                stop_reason = "converged"
                break

            # 合并去重 + 重排，重新评估
            accumulated = self.rerank(
                question, accumulated + new_docs, top_k,
                use_autocut=settings.use_autocut,
                autocut_min_docs=settings.autocut_min_docs,
            )
            acc_ids = {chunk_key(d) for d in accumulated}
            iterations_used += 1
            grade, _, reason = self.evaluate(question, accumulated)

        # 边界：最后一轮重评恰好转为 correct，但循环已耗尽
        if grade == "correct" and stop_reason == "max_iterations":
            stop_reason = "sufficient"

        return accumulated, iterations_used, stop_reason, grade

    def _refine_query(
        self, question: str, docs: List[Document], reason: str
    ) -> str:
        """调用查询精化；任何异常都优雅降级到原问题，不中断迭代。"""
        if not self.query_transformer:
            return question
        try:
            return self.query_transformer.refine(
                question, self._evidence_summary(docs), reason
            )
        except Exception as e:
            logger.warning(f"查询精化异常，回退原问题: {e}")
            return question

    @staticmethod
    def _evidence_summary(
        docs: List[Document], max_docs: int = 3, max_chars: int = 200
    ) -> str:
        """拼接前若干篇文档的截断内容，作为精化查询的证据上下文。"""
        return "\n".join(d.page_content[:max_chars] for d in docs[:max_docs])

    # ---- 阶段 3b: F6b 多跳查询分解（并行优先，依赖时链式） ----

    def _retrieve_subquery(
        self, question: str, subq: str, top_n: int
    ) -> List[Document]:
        """子问题轻量检索：跳过门控/改写，直接 dense+sparse+graph 召回 + fuse。

        graph 通道以子问题（而非原问题）做实体匹配——多跳分解的价值正在于
        每个子问题独立命中各自的关系链（2026-07-26 接入）。
        """
        recall_results = self.recall(
            subq, [subq], top_n=top_n, channels=("dense", "sparse", "graph")
        )
        return self.fuse(recall_results)

    def _decompose_retrieve(
        self, question: str, decomposition, top_k: int, trace_id: str | None = None
    ) -> List[Document]:
        """按分解结果检索并合并：无依赖→并行各子问题；有依赖→链式（hop 答案构造下一跳）。"""
        settings = self._settings
        subs = decomposition.sub_questions

        if not decomposition.chain:
            # 并行：各子问题独立轻量检索，统一 RRF 合并
            per_sub: List[List[Document]] = []
            with ThreadPoolExecutor(max_workers=settings.recall_max_workers) as executor:
                futures = [
                    executor.submit(self._retrieve_subquery, question, sq, top_k)
                    for sq in subs
                ]
                for fut in as_completed(futures):
                    try:
                        per_sub.append(fut.result())
                    except Exception as e:
                        logger.warning(f"子问题检索失败: {e}")
            return reciprocal_rank_fusion(per_sub) if per_sub else []

        # 链式：逐跳检索，用上一跳的压缩证据辅助构造下一跳查询
        accumulated: List[Document] = []
        current_subs = list(subs)[: settings.decomposition_max_hops]
        for i, sq in enumerate(current_subs):
            docs = self._retrieve_subquery(question, sq, top_k)
            accumulated = dedup_by_chunk_id(accumulated + docs)
            # 若不是最后一跳，用已检索证据精化下一跳（复用 F2 精化，异常回退原子问题）
            if i < len(current_subs) - 1 and self.query_transformer:
                try:
                    current_subs[i + 1] = self.query_transformer.refine(
                        current_subs[i + 1], self._evidence_summary(accumulated), "需结合上一跳结果"
                    )
                except Exception as e:
                    logger.warning(f"链式精化失败，沿用原子问题: {e}")
        return accumulated

    # ---- 主编排 ----

    def run(
        self,
        question: str,
        top_k: int | None = None,
        *,
        query_strategy: str = "multi_query",
        use_query_transform: bool = True,
        use_rerank: bool = True,
        trace_id: str | None = None,
    ) -> RetrievalResult:
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k
        start_time = time.time()
        tracer = get_tracer()

        result = RetrievalResult(documents=[])
        # 全局时延预算：可选阶段（F2 迭代/F3 重生成）超预算即跳过，熔断离群尾
        result.deadline = Deadline(getattr(settings, "latency_budget_ms", 0))
        result.budget_skipped = result.deadline.skipped  # 同一列表，生成层跳过也记入

        # F13 Agentic RAG：开启时由 ReAct agent 自主决策检索（替代固定七阶段）。
        # 任何异常或空证据都降级回下方固定管道，保证回归安全。
        # 注：agentic 分支不做门控（agent 总是检索），闲聊直接回答由 chain 层语义缓存/门控外的路径兜底。
        if settings.use_agentic and getattr(self, "agentic", None) is not None:
            if trace_id:
                tracer.start_span(trace_id, "agentic")
            agent_result = None
            try:
                agent_result = self.agentic.run(question, top_k, trace_id=trace_id)
            except Exception as e:
                logger.warning(f"F13 agentic 检索异常，降级回七阶段管道: {e}")
            if agent_result is not None and agent_result.documents:
                result.documents = agent_result.documents
                result.queries_used = agent_result.queries_used
                result.decomposed_subqueries = agent_result.decomposed_subqueries
                result.decomposition_chain = agent_result.decomposition_chain
                result.crag_grade = agent_result.crag_grade
                result.agent_steps = [
                    {"thought": s.thought, "action": s.action,
                     "args": s.args, "observation": s.observation}
                    for s in agent_result.steps
                ]
                result.agent_stop_reason = agent_result.stop_reason
                result.retrieval_time_ms = (time.time() - start_time) * 1000
                if trace_id:
                    tracer.end_span(trace_id, "agentic", {
                        "steps": len(agent_result.steps),
                        "stop_reason": agent_result.stop_reason,
                        "docs": len(result.documents),
                    })
                logger.info(
                    f"F13 agentic 检索: {len(agent_result.steps)} 步, "
                    f"stop={agent_result.stop_reason}, docs={len(result.documents)}"
                )
                return result
            logger.warning("F13 agentic 无有效证据，降级回七阶段管道")
            if trace_id:
                tracer.end_span(trace_id, "agentic", {"fallback": True})

        # F4 查询路由（零 LLM，前置）：按查询类型自适应调整检索深度与降噪强度。
        # 前置动机（2026-07-26 延迟治理）：multi_hop + 分解开启时投机改写的结果会被
        # 分解子问题取代（纯浪费的 LLM 调用），路由先行可短路该调用。
        effective_top_k = top_k
        effective_autocut_min = settings.autocut_min_docs
        if settings.use_query_router and self.query_router:
            decision = self.query_router.route(question)
            result.query_type = decision.query_type
            if decision.top_k is not None:
                effective_top_k = decision.top_k
            if decision.autocut_min_docs is not None:
                effective_autocut_min = decision.autocut_min_docs
            if trace_id:
                tracer.start_span(trace_id, "query_routing")
                tracer.end_span(trace_id, "query_routing", {
                    "query_type": decision.query_type,
                    "effective_top_k": effective_top_k,
                    "autocut_min_docs": effective_autocut_min,
                    "reason": decision.reason,
                })

        # ① 门控 + ② 改写：投机并行（门控 false 时丢弃改写结果，换 2-5s 延迟）。
        # multi_hop + 分解开启时跳过投机改写（分解路径用子问题，改写结果本就被丢弃）；
        # 若分解最终未触发，在召回前延迟补跑改写（见下方 not decomposed 分支）。
        skip_transform = (
            result.query_type == "multi_hop"
            and settings.use_decomposition
            and self.query_transformer is not None
            and use_query_transform
        )
        if trace_id:
            tracer.start_span(trace_id, "gate_transform")
        need_retrieval, gate_reason = True, "门控未启用"
        speculative = (
            settings.use_crag_gate
            and self.crag_evaluator
            and use_query_transform
            and self.query_transformer
        )
        if skip_transform:
            need_retrieval, gate_reason = self.gate(question)
            queries = []
        elif speculative:
            with ThreadPoolExecutor(max_workers=2) as executor:
                gate_future = executor.submit(self.gate, question)
                transform_future = executor.submit(
                    self.transform, question, query_strategy
                )
                need_retrieval, gate_reason = gate_future.result()
                queries = transform_future.result()
        else:
            need_retrieval, gate_reason = self.gate(question)
            queries = self.transform(question, query_strategy, use_query_transform)
        result.queries_used = queries
        if trace_id:
            tracer.end_span(trace_id, "gate_transform", {
                "need_retrieval": need_retrieval,
                "gate_reason": gate_reason,
                "speculative": speculative and not skip_transform,
                "skip_transform": skip_transform,
                "num_queries": len(queries),
            })

        if not need_retrieval:
            result.gate_skipped = True
            result.crag_action = f"门控跳过检索: {gate_reason}"
            result.retrieval_time_ms = (time.time() - start_time) * 1000
            logger.info(f"门控跳过检索: {gate_reason}")
            return result

        # F6b 多跳查询分解：仅 multi_hop 且开启分解时触发；分解出 >1 子问题才走分解合并
        use_decomp = (
            result.query_type == "multi_hop"
            and settings.use_decomposition
            and self.query_transformer is not None
        )
        decomposed = False
        if use_decomp:
            decomposition = self.query_transformer.decompose(question)
            if len(decomposition.sub_questions) > 1:
                if trace_id:
                    tracer.start_span(trace_id, "decomposition")
                result.decomposed_subqueries = decomposition.sub_questions
                result.decomposition_chain = decomposition.chain
                result.fused_results = self._decompose_retrieve(
                    question, decomposition, effective_top_k, trace_id=trace_id
                )
                decomposed = True
                if trace_id:
                    tracer.end_span(trace_id, "decomposition", {
                        "subquestions": decomposition.sub_questions,
                        "chain": decomposition.chain,
                        "merged": len(result.fused_results),
                    })

        if not decomposed:
            # multi_hop 跳过了投机改写但分解未触发 → 延迟补跑，不丢召回质量
            if not queries:
                queries = self.transform(question, query_strategy, use_query_transform)
                result.queries_used = queries
            # ③ 多路召回
            if trace_id:
                tracer.start_span(trace_id, "multi_recall")
            recall_results = self.recall(question, queries, trace_id=trace_id)
            result.dense_results = recall_results["dense"]
            result.sparse_results = recall_results["sparse"]
            result.graph_results = recall_results.get("graph", [])
            result.summary_results = recall_results.get("summary", [])
            if trace_id:
                tracer.end_span(trace_id, "multi_recall", {
                    "dense_hits": len(result.dense_results),
                    "sparse_hits": len(result.sparse_results),
                    "graph_hits": len(result.graph_results),
                    "pc_hits": len(recall_results.get("parent_child", [])),
                    "summary_hits": len(result.summary_results),
                })

            # ④ RRF 融合
            if trace_id:
                tracer.start_span(trace_id, "rrf_fusion")
            result.fused_results = self.fuse(recall_results)
            if trace_id:
                tracer.end_span(trace_id, "rrf_fusion", {"fused": len(result.fused_results)})

        # ⑤ 重排（+ F1 Autocut 自适应截断降噪）
        if trace_id:
            tracer.start_span(trace_id, "rerank")
        result.pre_autocut_count = len(result.fused_results)
        result.documents = self.rerank(
            question, result.fused_results, effective_top_k,
            use_rerank=use_rerank,
            use_autocut=settings.use_autocut,
            autocut_min_docs=effective_autocut_min,
        )
        if trace_id:
            tracer.end_span(trace_id, "rerank", {
                "enabled": use_rerank,
                "autocut": settings.use_autocut,
                "pre_autocut": result.pre_autocut_count,
                "final": len(result.documents),
            })

        # ⑥ CRAG 评估 + ⑦ 补救
        if trace_id:
            tracer.start_span(trace_id, "crag_evaluation")
        if self.crag_evaluator:
            grade, relevant_indices, reason = self.evaluate(question, result.documents)
            result.crag_grade = grade

            if settings.use_iterative_retrieval and self.query_transformer:
                # F2 Self-RAG 迭代检索：correct/ambiguous/incorrect 统一走质量驱动迭代。
                # 时延预算熔断：超预算整体跳过（迭代是最多 4 次串行 LLM 的可选增强）
                if result.deadline.check_skip("F2_iterative"):
                    result.iterative_stop_reason = "budget_skipped"
                    result.crag_action = "时延预算耗尽，跳过迭代检索"
                    logger.info("F2 迭代检索被时延预算熔断跳过")
                else:
                    final_docs, iters, stop_reason, final_grade = self._iterative_retrieve(
                        question, result.documents, effective_top_k,
                        grade, reason, trace_id=trace_id,
                    )
                    result.documents = final_docs
                    result.iterations_used = iters
                    result.iterative_stop_reason = stop_reason
                    # 仅当确实由非 correct 转为 correct 才标 recovered；首检即 correct 保持 correct
                    result.crag_grade = (
                        "recovered" if (final_grade == "correct" and grade != "correct")
                        else final_grade
                    )
                    result.crag_action = (
                        f"Self-RAG 迭代检索({stop_reason}, {iters}轮)"
                        if iters > 0 else f"Self-RAG 评估({stop_reason})"
                    )
            elif grade == "incorrect":
                logger.info(f"CRAG: 检索不相关（{reason}），触发 HyDE 完整管道重检索")
                retry_docs = self.remediate(question, effective_top_k, trace_id=trace_id)
                if retry_docs:
                    result.documents = retry_docs
                    result.crag_grade = "recovered"
                    result.crag_action = "HyDE 完整管道重检索"
                else:
                    result.crag_action = "补救失败，保留原结果"
            elif grade == "ambiguous":
                result.crag_action = "过滤不相关文档"
                result.documents = self.crag_evaluator.filter_relevant_docs(
                    result.documents, relevant_indices
                )
            else:
                result.crag_action = "直接使用"

        if trace_id:
            tracer.end_span(trace_id, "crag_evaluation", {
                "enabled": self.crag_evaluator is not None,
                "grade": result.crag_grade,
                "action": result.crag_action,
                "iterative": settings.use_iterative_retrieval,
                "iterations_used": result.iterations_used,
                "stop_reason": result.iterative_stop_reason,
            })

        result.retrieval_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"检索完成: {result.retrieval_time_ms:.0f}ms, "
            f"dense={len(result.dense_results)}, sparse={len(result.sparse_results)}, "
            f"fused={len(result.fused_results)}, final={len(result.documents)}, "
            f"crag={result.crag_grade or 'off'}"
        )
        return result

