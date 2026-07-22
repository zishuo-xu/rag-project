# 检索管道接线补全 + 并发性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 5 个已实现但未接线的功能（L1 摘要召回、CRAG 门控、CRAG 补救完整链路、query_embeddings 复用、语义分块开关）接入检索主链路，同时完成 chain.py 管道重构与中度并发性能优化。

**Architecture:** 新建 `app/retrieval/pipeline.py`（RetrievalPipeline 七阶段：门控→改写→召回→融合→重排→评估→补救），`RAGChain.retrieve()` 变为薄委托，对外签名不变。并发闸门用 `asyncio.Semaphore` 加在 `/api/chat` 入口，同步阻塞调用用 `asyncio.to_thread` 移出事件循环。

**Tech Stack:** Python 3.11 / FastAPI / LangChain / ChromaDB / pytest（全部单测 mock LLM 与 embedding，离线可跑）

**Spec:** `docs/superpowers/specs/2026-07-22-pipeline-wiring-and-concurrency-design.md`

## Global Constraints

- 所有新增单元测试必须离线运行：不初始化真实 `RAGChain`、不加载真实模型、不需要 API key（用 `unittest.mock` 注入）
- 管道行为开关全部走 `config.py` Settings，默认值：`use_summary_recall=True`、`use_crag_gate=True`、`recall_max_workers=6`、`max_concurrent_requests=4`、`request_queue_timeout=30.0`
- `RAGChain.retrieve(question, top_k=None, trace_id=None) -> RetrievalResult` 签名不变；`routes.py`、前端、评估脚本零适配
- 每个 Task 完成后运行 `pytest tests/ -x -q` 确认全绿再提交
- 提交信息用中文，格式 `type: 描述`（沿用仓库现有风格）

---

### Task 1: 基线确认与归档

**Files:**
- Modify: `data/eval_report_baseline.json`（由现有报告复制而来）

**Interfaces:**
- Consumes: 现有 `data/eval_report.json`（faithfulness 0.9819 / relevancy 0.90 / precision 0.9495 / recall 0.8822）
- Produces: 基线测试全绿状态；归档基线报告供 Task 11 对比

- [ ] **Step 1: 跑现有测试确认基线**

Run: `uv run pytest tests/ -q`
Expected: 全部通过。若 `tests/test_api.py` 因真实初始化失败，记录失败详情，该文件将在 Task 8 修复，不阻塞基线。

- [ ] **Step 2: 归档基线评估报告**

```bash
cp data/eval_report.json data/eval_report_baseline.json
```

- [ ] **Step 3: Commit**

```bash
git add data/eval_report_baseline.json
git commit -m "chore: 归档改动前评估基线报告"
```

---

### Task 2: 新增配置项

**Files:**
- Modify: `config.py:55-63`（在 CRAG / 缓存配置区附近插入）
- Test: `tests/test_config.py`（新建）

**Interfaces:**
- Consumes: 无
- Produces: `Settings.use_summary_recall: bool`、`Settings.use_crag_gate: bool`、`Settings.recall_max_workers: int`、`Settings.max_concurrent_requests: int`、`Settings.request_queue_timeout: float`，后续所有 Task 依赖这些字段名

- [ ] **Step 1: 写失败测试**

创建 `tests/test_config.py`：

```python
"""配置项测试"""
from config import Settings


def test_new_pipeline_settings_defaults():
    s = Settings()
    assert s.use_summary_recall is True
    assert s.use_crag_gate is True
    assert s.recall_max_workers == 6
    assert s.max_concurrent_requests == 4
    assert s.request_queue_timeout == 30.0
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL `AttributeError: 'Settings' object has no attribute 'use_summary_recall'`

- [ ] **Step 3: 实现配置**

`config.py` 在 `# CRAG 自纠正检索` 区块后插入：

```python
    # 检索管道接线
    use_summary_recall: bool = True  # L1 摘要索引作为第5路召回
    use_crag_gate: bool = True       # CRAG 门控：判断是否需要检索
    recall_max_workers: int = 6      # 多路召回线程池大小

    # 并发控制
    max_concurrent_requests: int = 4   # /api/chat 并发闸门
    request_queue_timeout: float = 30.0  # 排队超时秒数，超时返回503
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: 新增管道接线与并发控制配置项"
```

---

### Task 3: DenseRetriever 预计算 embedding 复用

**Files:**
- Modify: `app/retrieval/dense.py:24-37`
- Test: `tests/test_dense_embedding.py`（新建）

**Interfaces:**
- Consumes: `HierarchicalIndexer.chunk_store`（Chroma，`similarity_search_by_vector` 是 langchain-chroma 标准方法）、`HierarchicalIndexer.search_chunks(query, top_k)`
- Produces: `DenseRetriever.retrieve(query: str, top_k: int = 10, embedding: list | None = None) -> List[Document]`；Task 4 的 pipeline 召回阶段传入预计算 embedding

- [ ] **Step 1: 写失败测试**

创建 `tests/test_dense_embedding.py`：

```python
"""DenseRetriever 预计算 embedding 复用测试"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.retrieval.dense import DenseRetriever


def _make_retriever():
    indexer = MagicMock()
    indexer.search_chunks.return_value = [Document(page_content="by_query")]
    indexer.chunk_store.similarity_search_by_vector.return_value = [
        Document(page_content="by_vector")
    ]
    return DenseRetriever(indexer), indexer


def test_retrieve_with_precomputed_embedding():
    """传入预计算 embedding 时走向量检索，不再按文本编码"""
    retriever, indexer = _make_retriever()
    results = retriever.retrieve("问题", top_k=5, embedding=[0.1, 0.2, 0.3])
    indexer.chunk_store.similarity_search_by_vector.assert_called_once_with(
        [0.1, 0.2, 0.3], k=5
    )
    indexer.search_chunks.assert_not_called()
    assert results[0].page_content == "by_vector"


def test_retrieve_without_embedding_fallback():
    """不传 embedding 时保持原有文本检索行为"""
    retriever, indexer = _make_retriever()
    results = retriever.retrieve("问题", top_k=5)
    indexer.search_chunks.assert_called_once_with("问题", top_k=5)
    assert results[0].page_content == "by_query"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_dense_embedding.py -q`
Expected: FAIL `TypeError: retrieve() got an unexpected keyword argument 'embedding'`

- [ ] **Step 3: 实现**

替换 `app/retrieval/dense.py:24-37` 的 `retrieve` 方法：

```python
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        embedding: list | None = None,
    ) -> List[Document]:
        """
        稠密向量检索。

        Args:
            query: 查询文本
            top_k: 返回 top-K 结果
            embedding: 可选的预计算查询向量（多查询变体场景下批量预算后复用，
                       避免每个变体重复调用 embedding 模型）

        Returns:
            按相似度排序的文档列表
        """
        if embedding is not None:
            results = self.indexer.chunk_store.similarity_search_by_vector(
                embedding, k=top_k
            )
        else:
            results = self.indexer.search_chunks(query, top_k=top_k)
        logger.debug(f"稠密检索: query='{query[:50]}...', 返回 {len(results)} 条")
        return results
```

- [ ] **Step 4: 运行确认通过**

Run: `uv run pytest tests/test_dense_embedding.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/dense.py tests/test_dense_embedding.py
git commit -m "feat: DenseRetriever 支持预计算 embedding 复用"
```

---

### Task 4: RetrievalPipeline 重构（行为不变）

**Files:**
- Create: `app/retrieval/pipeline.py`
- Modify: `app/generation/chain.py`（整体重写，见 Step 3）
- Test: `tests/test_pipeline.py`（新建，本 Task 只含基础阶段测试，接线行为测试在后续 Task 追加）

**Interfaces:**
- Consumes: `DenseRetriever.retrieve(query, top_k, embedding)`（Task 3）、`SparseRetriever.retrieve(query, top_k)`、`GraphRetriever.retrieve(question, top_k)`、`ParentChildRetriever.retrieve(query, top_k)` / `.has_index()`、`HierarchicalIndexer.hierarchical_search(query, top_k)`、`QueryTransformer.transform(question, strategy)`、`CRAGEvaluator.evaluate_relevance(question, documents)` / `.filter_relevant_docs(docs, indices)` / `.should_retrieve(question)`、`Reranker.rerank(query, documents, top_k)`、`reciprocal_rank_fusion(result_lists)`、`get_tracer()`
- Produces:
  - `RetrievalResult`（dataclass，**迁移到 pipeline.py**，新增字段 `summary_results: List[Document]`、`gate_skipped: bool = False`）
  - `RetrievalPipeline(indexer, dense_retriever, sparse_retriever, *, reranker=None, query_transformer=None, graph_retriever=None, parent_child_retriever=None, crag_evaluator=None, settings=None)`
  - 阶段方法：`gate(question) -> tuple[bool, str]`、`transform(question, strategy, use_query_transform=True) -> List[str]`、`recall(question, queries, top_n=None, channels=ALL_CHANNELS, trace_id=None) -> dict`、`fuse(recall_results) -> List[Document]`、`rerank(question, documents, top_k, use_rerank=True) -> List[Document]`、`evaluate(question, documents) -> tuple[str, List[int], str]`、`remediate(question, top_k, trace_id=None) -> List[Document]`、`run(question, top_k=None, *, query_strategy, use_query_transform, use_rerank, trace_id) -> RetrievalResult`
  - `ALL_CHANNELS = ("dense", "sparse", "graph", "parent_child", "summary")`
  - `RAGChain.retrieve()` 委托给 pipeline；`chain.py` 继续 re-export `RetrievalResult`（routes.py 等 import 不变）
  - `RAGChain.generate_direct(question, chat_history=None) -> str` 与 `generate_direct_stream(...)`：门控跳过时无上下文直接回答（本 Task 实现方法体，接线在 Task 6）
  - `RAGChain._check_cache(question, chat_history, trace_id, start_time) -> tuple[RAGResponse | None, np.ndarray | None]` 与 `_write_cache(question, q_embedding, answer, documents)`：消除 invoke/invoke_stream 重复

- [ ] **Step 1: 写失败测试**

创建 `tests/test_pipeline.py`：

```python
"""RetrievalPipeline 阶段测试（全 mock，离线运行）"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.retrieval.pipeline import RetrievalPipeline, ALL_CHANNELS


def _doc(content, chunk_id=None):
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id or content, "source": "test.md"},
    )


def _make_pipeline(**overrides):
    """构造全 mock 的 pipeline，默认各组件行为可用"""
    indexer = MagicMock()
    indexer.embeddings.embed_documents.return_value = [[0.1], [0.2]]
    indexer.hierarchical_search.return_value = [_doc("summary_hit")]

    dense = MagicMock()
    dense.retrieve.side_effect = lambda q, top_k=10, embedding=None: [_doc(f"dense:{q}")]
    sparse = MagicMock()
    sparse.retrieve.side_effect = lambda q, top_k=10: [_doc(f"sparse:{q}")]

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, docs, top_k=None: docs[: top_k or 5]

    transformer = MagicMock()
    transformer.transform.side_effect = lambda q, strategy="multi_query": [q, f"{q}变体"]

    graph = MagicMock()
    graph.retrieve.return_value = [_doc("graph_hit")]

    pc = MagicMock()
    pc.has_index.return_value = True
    pc.retrieve.return_value = [_doc("pc_hit")]

    crag = MagicMock()
    crag.should_retrieve.return_value = (True, "需要检索")
    crag.evaluate_relevance.return_value = ("correct", [1], "相关")

    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 20
    settings.recall_max_workers = 6
    settings.use_summary_recall = True
    settings.use_crag_gate = True

    kwargs = dict(
        indexer=indexer, dense_retriever=dense, sparse_retriever=sparse,
        reranker=reranker, query_transformer=transformer,
        graph_retriever=graph, parent_child_retriever=pc,
        crag_evaluator=crag, settings=settings,
    )
    kwargs.update(overrides)
    pipe = RetrievalPipeline(**kwargs)
    return pipe, kwargs


def test_recall_five_channels_aggregated():
    """五路召回结果按 channel 聚合"""
    pipe, mocks = _make_pipeline()
    results = pipe.recall("问题", ["问题", "问题变体"])
    assert set(results.keys()) == set(ALL_CHANNELS)
    assert len(results["dense"]) == 2      # 2 个查询变体各 1 条
    assert len(results["sparse"]) == 2
    assert results["graph"] == mocks["graph_retriever"].retrieve.return_value
    assert results["parent_child"] == mocks["parent_child_retriever"].retrieve.return_value
    assert results["summary"] == mocks["indexer"].hierarchical_search.return_value


def test_recall_single_channel_failure_degrades():
    """单路召回失败不影响其他路"""
    pipe, mocks = _make_pipeline()
    mocks["sparse_retriever"].retrieve.side_effect = RuntimeError("BM25 炸了")
    results = pipe.recall("问题", ["问题"])
    assert results["sparse"] == []
    assert len(results["dense"]) == 1


def test_recall_embedding_precomputed_once():
    """多查询变体时 embedding 批量预算一次并逐变体分发"""
    pipe, mocks = _make_pipeline()
    queries = ["问题", "问题变体"]
    pipe.recall("问题", queries)
    mocks["indexer"].embeddings.embed_documents.assert_called_once_with(queries)
    calls = mocks["dense_retriever"].retrieve.call_args_list
    assert calls[0].kwargs["embedding"] == [0.1]
    assert calls[1].kwargs["embedding"] == [0.2]


def test_run_happy_path_correct_grade():
    """完整管道：correct 评级直接使用"""
    pipe, mocks = _make_pipeline()
    result = pipe.run("问题")
    assert result.crag_grade == "correct"
    assert result.crag_action == "直接使用"
    assert result.gate_skipped is False
    assert len(result.documents) > 0
    assert result.queries_used == ["问题", "问题变体"]
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'app.retrieval.pipeline'`

- [ ] **Step 3: 实现 pipeline.py**

创建 `app/retrieval/pipeline.py`：

```python
"""检索管道 - 门控/改写/多路召回/融合/重排/评估/补救 七阶段"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from config import get_settings
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.fusion import reciprocal_rank_fusion
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
    reranked_results: List[Document] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0
    crag_grade: str = ""    # correct / ambiguous / incorrect / recovered
    crag_action: str = ""   # 采取的动作描述
    gate_skipped: bool = False  # 门控判定无需检索


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
        settings=None,
    ):
        self.indexer = indexer
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.graph_retriever = graph_retriever
        self.parent_child_retriever = parent_child_retriever
        self.crag_evaluator = crag_evaluator
        self._settings = settings or get_settings()

    # ---- 阶段 1: 门控 ----

    def gate(self, question: str) -> tuple[bool, str]:
        """判断是否需要检索。失败时默认检索（保守方向，不漏检索）"""
        if not (self._settings.use_crag_gate and self.crag_evaluator):
            return True, "门控未启用"
        return self.crag_evaluator.should_retrieve(question)

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
                results[ch] = self._deduplicate(results[ch])
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
    ) -> List[Document]:
        if use_rerank and self.reranker:
            return self.reranker.rerank(question, documents, top_k=top_k)
        return documents[:top_k]

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
        recall_results = self.recall(
            question, hyde_queries, top_n=top_k,
            channels=("dense", "sparse"), trace_id=trace_id,
        )
        fused = self.fuse(recall_results)
        return self.rerank(question, fused, top_k)

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

        # ① 门控 + ② 改写：投机并行（门控 false 时丢弃改写结果，换 2-5s 延迟）
        if trace_id:
            tracer.start_span(trace_id, "gate_transform")
        need_retrieval, gate_reason = True, "门控未启用"
        speculative = (
            settings.use_crag_gate
            and self.crag_evaluator
            and use_query_transform
            and self.query_transformer
        )
        if speculative:
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
                "speculative": speculative,
                "num_queries": len(queries),
            })

        if not need_retrieval:
            result.gate_skipped = True
            result.crag_action = f"门控跳过检索: {gate_reason}"
            result.retrieval_time_ms = (time.time() - start_time) * 1000
            logger.info(f"门控跳过检索: {gate_reason}")
            return result

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

        # ⑤ 重排
        if trace_id:
            tracer.start_span(trace_id, "rerank")
        result.reranked_results = self.rerank(
            question, result.fused_results, top_k, use_rerank=use_rerank
        )
        result.documents = result.reranked_results
        if trace_id:
            tracer.end_span(trace_id, "rerank", {
                "enabled": use_rerank, "final": len(result.documents),
            })

        # ⑥ CRAG 评估 + ⑦ 补救
        if trace_id:
            tracer.start_span(trace_id, "crag_evaluation")
        if self.crag_evaluator:
            grade, relevant_indices, reason = self.evaluate(question, result.documents)
            result.crag_grade = grade

            if grade == "incorrect":
                logger.info(f"CRAG: 检索不相关（{reason}），触发 HyDE 完整管道重检索")
                retry_docs = self.remediate(question, top_k, trace_id=trace_id)
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
            })

        result.retrieval_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"检索完成: {result.retrieval_time_ms:.0f}ms, "
            f"dense={len(result.dense_results)}, sparse={len(result.sparse_results)}, "
            f"fused={len(result.fused_results)}, final={len(result.documents)}, "
            f"crag={result.crag_grade or 'off'}"
        )
        return result

    @staticmethod
    def _deduplicate(documents: List[Document]) -> List[Document]:
        """基于 chunk_id 去重"""
        seen = set()
        unique = []
        for doc in documents:
            key = doc.metadata.get("chunk_id", id(doc))
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique
```

- [ ] **Step 4: 重写 chain.py 为薄编排层**

整体替换 `app/generation/chain.py` 为以下内容（保留对外 API：`RetrievalResult` / `RAGResponse` / `RAGChain` 及其全部现有方法签名）：

```python
"""RAG Chain - 编排层：缓存 + 检索管道委托 + 生成"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Generator

import numpy as np
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config import get_settings
from app.ingestion.indexer import HierarchicalIndexer
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from app.retrieval.reranker import Reranker
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.parent_child import ParentChildRetriever
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.cache import get_semantic_cache
from app.generation.prompts import (
    RAG_SIMPLE_PROMPT, RAG_CHAT_PROMPT, DIRECT_ANSWER_PROMPT, FALLBACK_RESPONSE,
)
from app.observability.tracing import get_tracer

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

        self.semantic_cache = get_semantic_cache() if settings.cache_enabled else None

        self.use_query_transform = use_query_transform
        self.use_rerank = use_rerank
        self.use_graph = use_graph and settings.graph_enabled
        self.query_strategy = query_strategy

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            streaming=True,
            request_timeout=60,
            max_retries=2,
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
        )

        logger.info(
            f"RAGChain 初始化: query_transform={use_query_transform}, "
            f"rerank={use_rerank}, strategy={query_strategy}, "
            f"parent_child={self.parent_child_retriever is not None}, "
            f"crag={self.crag_evaluator is not None}, "
            f"cache={self.semantic_cache is not None}"
        )

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

    def compress_context(
        self,
        question: str,
        documents: List[Document],
        max_sentences_per_doc: int = 3,
    ) -> List[Document]:
        """上下文压缩：关键词重叠抽句（零LLM调用）"""
        import re as _re
        q_tokens = set(_re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]+', question.lower()))
        if not q_tokens:
            return documents

        compressed = []
        for doc in documents:
            sentences = _re.split(r'(?<=[。！？.!?\n])', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= max_sentences_per_doc:
                compressed.append(doc)
                continue

            scored = []
            for sent in sentences:
                s_tokens = set(_re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]+', sent.lower()))
                scored.append((len(q_tokens & s_tokens), sent))

            scored.sort(key=lambda x: -x[0])
            top_sentences = set(s for _, s in scored[:max_sentences_per_doc])
            kept = [s for s in sentences if s in top_sentences]

            if kept:
                compressed.append(Document(
                    page_content="".join(kept), metadata=doc.metadata.copy(),
                ))
            else:
                compressed.append(doc)

        return compressed

    def generate(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
    ) -> str:
        """基于检索结果生成回答"""
        if not documents:
            return FALLBACK_RESPONSE

        documents = self.compress_context(question, documents)
        context = self._format_context(documents)

        if chat_history:
            chain = RAG_CHAT_PROMPT | self.llm | StrOutputParser()
            return chain.invoke({
                "context": context, "question": question,
                "chat_history": chat_history,
            })
        chain = RAG_SIMPLE_PROMPT | self.llm | StrOutputParser()
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

    def invoke(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """完整 RAG 调用：[缓存检查] + 检索 + 生成 + [缓存写入]（自动记录 Trace）"""
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            return cached

        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)

        tracer.start_span(trace_id, "generation")
        if retrieval_result.gate_skipped:
            answer = self.generate_direct(question, chat_history)
        else:
            answer = self.generate(question, retrieval_result.documents, chat_history)
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=answer)
        self._write_cache(question, chat_history, q_embedding, answer, retrieval_result.documents)

        return RAGResponse(
            answer=answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
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
            {"type": "done", "data": RAGResponse}
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            yield {"type": "cache_hit", "data": cached}
            yield {"type": "done", "data": cached}
            return

        yield {"type": "retrieving", "data": "正在检索相关文档..."}

        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)
        yield {"type": "retrieval", "data": retrieval_result}

        tracer.start_span(trace_id, "generation")
        full_answer = ""
        if retrieval_result.gate_skipped:
            token_stream = self.generate_direct_stream(question, chat_history)
        else:
            token_stream = self.generate_stream(
                question, retrieval_result.documents, chat_history
            )
        for token in token_stream:
            full_answer += token
            yield {"type": "token", "data": token}
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(full_answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=full_answer)
        self._write_cache(
            question, chat_history, q_embedding, full_answer, retrieval_result.documents
        )

        yield {"type": "done", "data": RAGResponse(
            answer=full_answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
        )}

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
            q_embedding = np.array(self.indexer.embeddings.embed_query(question))
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
                embedding = np.array(self.indexer.embeddings.embed_query(question))
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
```

注意：原 `_deduplicate` 已迁移到 `RetrievalPipeline`，chain 不再保留；原文件头部 `from app.retrieval.fusion import reciprocal_rank_fusion` 等不再使用的 import 一并移除。

- [ ] **Step 5: prompts.py 增加 DIRECT_ANSWER_PROMPT**

`app/generation/prompts.py` 中 `FALLBACK_RESPONSE` 定义之后追加：

```python
# 门控判定无需检索时的直接回答 Prompt
DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的 AI 助手。用户的问题不依赖知识库文档，请基于通用知识直接回答，保持简洁友好。如果问题涉及你不知道的特定文档内容，请说明。"),
    ("human", "{question}"),
])
```

- [ ] **Step 6: 运行测试确认通过**

Run: `uv run pytest tests/test_pipeline.py tests/test_dense_embedding.py -q`
Expected: PASS（4 个 pipeline 测试 + 2 个 dense 测试）

- [ ] **Step 7: 全量回归**

Run: `uv run pytest tests/ -q`
Expected: 除 Task 1 已记录的 test_api.py 既有问题外全绿；`test_retrieval.py` 的 RRF/BM25 测试不受影响

- [ ] **Step 8: Commit**

```bash
git add app/retrieval/pipeline.py app/generation/chain.py app/generation/prompts.py tests/test_pipeline.py
git commit -m "refactor: 拆出 RetrievalPipeline 七阶段管道 + 缓存逻辑去重 + 门控直答 Prompt"
```

---

### Task 5: L1 摘要召回路接线验证

**Files:**
- Modify: `tests/test_pipeline.py`（追加测试）
- Modify: `app/api/routes.py:104-118`（retrieval SSE 事件补充 summary 计数）

**Interfaces:**
- Consumes: Task 4 的 `recall()` summary channel、`RetrievalResult.summary_results`
- Produces: SSE `retrieval` 事件新增 `summary_count` 字段（前端可忽略未知字段，无破坏性）

说明：摘要召回的管道代码已在 Task 4 随 `recall()` 落地（`settings.use_summary_recall` 开关控制），本 Task 补行为测试与可观测性。

- [ ] **Step 1: 追加失败测试**

`tests/test_pipeline.py` 追加：

```python
def test_summary_channel_disabled_by_config():
    """use_summary_recall=False 时摘要路不召回"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_summary_recall = False
    results = pipe.recall("问题", ["问题"])
    assert results["summary"] == []
    mocks["indexer"].hierarchical_search.assert_not_called()


def test_summary_results_feed_fusion():
    """摘要召回结果进入 RRF 融合"""
    pipe, _ = _make_pipeline()
    recall_results = pipe.recall("问题", ["问题"])
    fused = pipe.fuse(recall_results)
    contents = [d.page_content for d in fused]
    assert "summary_hit" in contents


def test_run_populates_summary_results():
    """run() 结果包含 summary_results 字段"""
    pipe, _ = _make_pipeline()
    result = pipe.run("问题")
    assert len(result.summary_results) == 1
    assert result.summary_results[0].page_content == "summary_hit"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL（`test_summary_channel_disabled_by_config` 中 `hierarchical_search` 被调用 / `run()` 结果无 summary_results —— 若 Task 4 实现已正确则可能直接 PASS，此时跳过 Step 3）

- [ ] **Step 3: 修复实现**

按失败信息修正 `pipeline.py`（典型问题：`recall()` 中 summary 分支未检查 `settings.use_summary_recall`，或 `run()` 未填充 `result.summary_results`）。Task 4 给出的参考实现已含正确逻辑，本步以测试为准。

- [ ] **Step 4: SSE 事件补充 summary 计数**

`app/api/routes.py` 的 `_stream_response` 中 `retrieval` 事件的 data 字典（约 :104-118），在 `"crag_action"` 前插入一行：

```python
                    "summary_count": len(retrieval.summary_results),
```

- [ ] **Step 5: 运行测试 + Commit**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: PASS

```bash
git add tests/test_pipeline.py app/api/routes.py
git commit -m "feat: L1 摘要召回路接线验证 + SSE retrieval 事件补充 summary 计数"
```

---

### Task 6: CRAG 门控接线验证

**Files:**
- Modify: `tests/test_pipeline.py`（追加门控测试）

**Interfaces:**
- Consumes: Task 4 的 `gate()` / `run()` 投机并行逻辑、`RAGChain.generate_direct()`
- Produces: 门控三分支行为（跳过 / 通过 / 失败降级）测试保障

说明：门控代码已在 Task 4 落地，本 Task 补齐行为测试。

- [ ] **Step 1: 追加测试**

`tests/test_pipeline.py` 追加：

```python
def test_gate_skip_returns_empty_with_flag():
    """门控判定无需检索：跳过召回，gate_skipped=True"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].should_retrieve.return_value = (False, "闲聊")
    result = pipe.run("你好")
    assert result.gate_skipped is True
    assert result.documents == []
    assert "门控跳过检索" in result.crag_action
    # 投机并行下改写已执行但结果被丢弃，召回不应发生
    mocks["dense_retriever"].retrieve.assert_not_called()


def test_gate_failure_defaults_to_retrieve():
    """门控调用异常时默认检索"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].should_retrieve.side_effect = RuntimeError("LLM 超时")
    # should_retrieve 内部已有 try/except，但即便异常穿透，gate() 也不应让管道崩溃
    try:
        result = pipe.run("问题")
        gate_raised = False
    except RuntimeError:
        gate_raised = True
    # CRAGEvaluator.should_retrieve 自身吞异常返回 (True, ...)；
    # 若异常穿透则 gate() 必须改为内部 try/except（见 Step 2）
    assert not gate_raised, "gate() 应吞掉异常并默认检索"


def test_gate_disabled_no_speculation():
    """use_crag_gate=False 时不调用门控，直接改写+检索"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_crag_gate = False
    result = pipe.run("问题")
    mocks["crag_evaluator"].should_retrieve.assert_not_called()
    assert result.gate_skipped is False
    assert len(result.documents) > 0
```

- [ ] **Step 2: 运行并按需修复**

Run: `uv run pytest tests/test_pipeline.py -q`

若 `test_gate_failure_defaults_to_retrieve` 失败（异常穿透），将 `pipeline.py` 的 `gate()` 改为：

```python
    def gate(self, question: str) -> tuple[bool, str]:
        """判断是否需要检索。失败时默认检索（保守方向，不漏检索）"""
        if not (self._settings.use_crag_gate and self.crag_evaluator):
            return True, "门控未启用"
        try:
            return self.crag_evaluator.should_retrieve(question)
        except Exception as e:
            logger.warning(f"门控判断失败，默认检索: {e}")
            return True, f"门控异常: {e}"
```

- [ ] **Step 3: 运行确认通过 + Commit**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: PASS

```bash
git add tests/test_pipeline.py app/retrieval/pipeline.py
git commit -m "feat: CRAG 门控三分支行为测试（跳过/通过/失败降级）"
```

---

### Task 7: CRAG 补救链路 + 数字校验快速路径验证

**Files:**
- Modify: `tests/test_pipeline.py`（追加补救与数字校验测试）

**Interfaces:**
- Consumes: Task 4 的 `remediate()` / `evaluate()`、`CRAGEvaluator.validate_numeric_answer`（静态方法，已实现）
- Produces: 补救链路行为测试保障

- [ ] **Step 1: 追加测试**

`tests/test_pipeline.py` 追加：

```python
def test_remediate_full_pipeline_on_incorrect():
    """incorrect 时补救走完整 mini-pipeline：HyDE + dense/sparse + RRF + rerank"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("incorrect", [], "无关")
    result = pipe.run("问题")
    # HyDE 改写被调用
    hyde_calls = [
        c for c in mocks["query_transformer"].transform.call_args_list
        if c.args[1] == "hyde"
    ]
    assert len(hyde_calls) == 1
    # 补救结果经过 rerank（reranker 至少被调用 2 次：主检索 1 次 + 补救 1 次）
    assert mocks["reranker"].rerank.call_count >= 2
    assert result.crag_grade == "recovered"
    assert result.crag_action == "HyDE 完整管道重检索"


def test_remediate_not_invoked_for_correct():
    """correct 时不触发补救"""
    pipe, mocks = _make_pipeline()
    pipe.run("问题")
    hyde_calls = [
        c for c in mocks["query_transformer"].transform.call_args_list
        if len(c.args) > 1 and c.args[1] == "hyde"
    ]
    assert hyde_calls == []


def test_ambiguous_filters_irrelevant_docs():
    """ambiguous 时按 relevant_indices 过滤"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("ambiguous", [1], "部分相关")
    mocks["crag_evaluator"].filter_relevant_docs.side_effect = (
        lambda docs, idx: docs[:1]
    )
    result = pipe.run("问题")
    assert result.crag_action == "过滤不相关文档"
    assert len(result.documents) == 1


def test_numeric_fastpath_skips_llm_judge():
    """数字型问题检索结果缺数字：零 LLM 直接判 incorrect，不调 LLM 评估"""
    pipe, mocks = _make_pipeline()
    # 所有召回结果都不含数字
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: [_doc("没有数字的内容")]
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: [_doc("还是没有数字")]
    )
    mocks["indexer"].hierarchical_search.return_value = [_doc("摘要也没数字")]
    mocks["graph_retriever"].retrieve.return_value = []
    mocks["parent_child_retriever"].retrieve.return_value = []
    result = pipe.run("范廷颂是哪一年被任命的？")
    mocks["crag_evaluator"].evaluate_relevance.assert_not_called()
    assert result.crag_grade == "recovered"  # incorrect -> 补救成功（mock 下必有结果）


def test_remediate_failure_keeps_original():
    """补救返回空时保留原结果"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("incorrect", [], "无关")
    mocks["reranker"].rerank.side_effect = lambda q, docs, top_k=None: (
        docs[: top_k or 5] if not getattr(mocks["reranker"].rerank, "_called", False) else []
    )
    # 简化：直接让 remediate 阶段 fuse 结果为空
    mocks["query_transformer"].transform.side_effect = (
        lambda q, strategy="multi_query": [q] if strategy == "hyde" else [q, f"{q}变体"]
    )
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: [] if q == "问题" else [_doc(f"dense:{q}")]
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: [] if q == "问题" else [_doc(f"sparse:{q}")]
    )
    # 主检索：queries=["问题","问题变体"]，"问题"变体返回空但"问题变体"有结果
    # HyDE 重检索 queries=["问题"] -> 全空 -> remediate 返回空
    result = pipe.run("问题")
    if result.crag_grade != "recovered":
        assert result.crag_action == "补救失败，保留原结果"
        assert len(result.documents) > 0
```

注：`test_remediate_failure_keeps_original` 依赖 mock 细节较脆弱，实现时若行为断言不稳定，可改为直接单测 `remediate()` 返回空列表的场景 + `run()` 中 `retry_docs` 为空分支的集成断言二选一，以保持测试可靠。

- [ ] **Step 2: 运行并按需修复**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: PASS；若失败按断言信息修正 `pipeline.py` 的 `evaluate()` / `remediate()` / `run()` 补救分支

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py app/retrieval/pipeline.py
git commit -m "feat: CRAG 补救完整链路 + 数字校验零LLM快速路径行为测试"
```

---

### Task 8: 上传 API chunk_strategy 参数 + 修复 test_api 真实初始化

**Files:**
- Modify: `app/api/routes.py:143-184`（upload_document）
- Modify: `tests/test_api.py`（改为 mock RAGChain，离线运行）
- Test: `tests/test_upload_chunk_strategy.py`（新建）

**Interfaces:**
- Consumes: `smart_chunk(documents, embeddings=None, use_semantic=False, short_doc_threshold=1000)`、`routes.set_rag_chain(chain)`
- Produces: `POST /api/documents/upload` 接受表单字段 `chunk_strategy`（`"recursive"` 默认 | `"semantic"`）；`tests/test_api.py` 不再触发真实 RAGChain 初始化

- [ ] **Step 1: 写失败测试**

创建 `tests/test_upload_chunk_strategy.py`：

```python
"""上传 API chunk_strategy 参数测试（mock RAGChain，离线运行）"""
import io
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router, set_rag_chain


def _make_client():
    app = FastAPI()
    app.include_router(router)
    chain = MagicMock()
    chain.indexer.index_documents.return_value = None
    chain.sparse_retriever.add_documents.return_value = None
    set_rag_chain(chain)
    return TestClient(app), chain


def test_upload_default_recursive():
    """默认走递归分块"""
    client, chain = _make_client()
    with patch("app.api.routes.smart_chunk") as mock_chunk, \
         patch("app.api.routes.load_document") as mock_load:
        mock_load.return_value = [MagicMock(metadata={"doc_id": "d1"})]
        mock_chunk.return_value = [MagicMock()]
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
        )
    assert resp.status_code == 200
    assert mock_chunk.call_args.kwargs.get("use_semantic", False) is False


def test_upload_semantic_strategy():
    """chunk_strategy=semantic 时启用语义分块"""
    client, chain = _make_client()
    with patch("app.api.routes.smart_chunk") as mock_chunk, \
         patch("app.api.routes.load_document") as mock_load:
        mock_load.return_value = [MagicMock(metadata={"doc_id": "d1"})]
        mock_chunk.return_value = [MagicMock()]
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
            data={"chunk_strategy": "semantic"},
        )
    assert resp.status_code == 200
    assert mock_chunk.call_args.kwargs["use_semantic"] is True


def test_upload_invalid_strategy_400():
    """非法 chunk_strategy 返回 400"""
    client, _ = _make_client()
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
        data={"chunk_strategy": "banana"},
    )
    assert resp.status_code == 400
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_upload_chunk_strategy.py -q`
Expected: FAIL（`chunk_strategy=semantic` 时 `use_semantic` 不是 True；非法值未返回 400）

- [ ] **Step 3: 实现**

`app/api/routes.py` 修改 upload 端点签名与分块调用：

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

@router.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
):
    """
    上传文档并建立索引。

    支持格式：PDF, TXT, Markdown
    chunk_strategy: recursive（默认，递归字符分块）| semantic（语义分块）
    """
    chain = get_rag_chain()

    if chunk_strategy not in ("recursive", "semantic"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 chunk_strategy: {chunk_strategy}，支持: recursive, semantic",
        )

    suffix = Path(file.filename).suffix
    if suffix.lower() not in (".pdf", ".txt", ".md", ".markdown"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持: .pdf, .txt, .md"
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        docs = load_document(tmp_path)
        chunks = smart_chunk(
            docs,
            embeddings=chain.indexer.embeddings if chunk_strategy == "semantic" else None,
            use_semantic=(chunk_strategy == "semantic"),
        )
        chain.indexer.index_documents(chunks)
        chain.sparse_retriever.add_documents(chunks)

        return UploadResponse(
            message=f"文档 '{file.filename}' 上传并索引成功",
            filename=file.filename,
            num_chunks=len(chunks),
            doc_id=docs[0].metadata.get("doc_id", "") if docs else "",
        )

    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
```

- [ ] **Step 4: 修复 test_api.py 真实初始化**

阅读 `tests/test_api.py`，将其中通过 `TestClient(app)` 触发 lifespan（真实初始化 RAGChain）的写法，替换为 `_make_client()` 同款模式：独立 `FastAPI()` 实例 + `include_router(router)` + `set_rag_chain(MagicMock())`，断言逻辑不变。目标：`uv run pytest tests/test_api.py -q` 在无 API key、无本地模型环境下通过。

- [ ] **Step 5: 运行测试 + Commit**

Run: `uv run pytest tests/test_upload_chunk_strategy.py tests/test_api.py -q`
Expected: PASS

```bash
git add app/api/routes.py tests/test_upload_chunk_strategy.py tests/test_api.py
git commit -m "feat: 上传API支持chunk_strategy语义分块开关 + test_api改为离线mock"
```

---

### Task 9: 并发闸门 + 事件循环解阻

**Files:**
- Modify: `main.py:25-53`（lifespan 创建 Semaphore）
- Modify: `app/api/routes.py:46-138`（chat 端点 + `_stream_response`）
- Test: `tests/test_concurrency_gate.py`（新建）

**Interfaces:**
- Consumes: `settings.max_concurrent_requests` / `settings.request_queue_timeout`（Task 2）
- Produces:
  - `routes.set_concurrency_gate(gate: asyncio.Semaphore | None)` / `routes.get_concurrency_gate() -> asyncio.Semaphore | None`
  - `/api/chat` 非流式：`asyncio.to_thread` 执行 `chain.invoke`，闸门内运行
  - `/api/chat` 流式：闸门在 `_stream_response` 生成器内获取/释放，排队超时返回 SSE `error` 事件
  - 非流式排队超时：HTTP 503 `{"detail": "服务繁忙，请求排队超时，请稍后重试"}`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_concurrency_gate.py`：

```python
"""并发闸门测试（mock RAGChain，离线运行）"""
import asyncio
import io
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes import router, set_rag_chain, set_concurrency_gate
from app.generation.chain import RAGResponse
from app.retrieval.pipeline import RetrievalResult


def _make_app(concurrency: int):
    app = FastAPI()
    app.include_router(router)

    active = {"cur": 0, "max": 0}

    def slow_invoke(question, chat_history=None, top_k=None):
        active["cur"] += 1
        active["max"] = max(active["max"], active["cur"])
        time.sleep(0.2)  # 模拟阻塞的 LLM 调用
        active["cur"] -= 1
        return RAGResponse(
            answer="ok", sources=[],
            retrieval_result=RetrievalResult(documents=[]),
        )

    chain = MagicMock()
    chain.invoke.side_effect = slow_invoke
    set_rag_chain(chain)
    set_concurrency_gate(asyncio.Semaphore(concurrency))
    return app, active


@pytest.mark.asyncio
async def test_gate_limits_concurrency():
    """并发 5 个请求，闸门=2 时同时在执行的 invoke 不超过 2"""
    app, active = _make_app(concurrency=2)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await asyncio.gather(*[
            client.post("/api/chat", json={"question": f"q{i}", "stream": False})
            for i in range(5)
        ])
    assert active["max"] <= 2


@pytest.mark.asyncio
async def test_gate_timeout_returns_503(monkeypatch):
    """排队超时返回 503"""
    app, _ = _make_app(concurrency=1)
    settings = MagicMock()
    settings.request_queue_timeout = 0.05  # 50ms 必超时
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        results = await asyncio.gather(*[
            client.post("/api/chat", json={"question": f"q{i}", "stream": False})
            for i in range(3)
        ])
    statuses = [r.status_code for r in results]
    assert 503 in statuses
```

注意：`set_concurrency_gate` 是全局状态，测试文件末尾或 fixture 中需 `set_concurrency_gate(None)` 复位，避免污染其他测试。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_concurrency_gate.py -q`
Expected: FAIL `ImportError: cannot import name 'set_concurrency_gate'`

- [ ] **Step 3: 实现 routes.py 闸门与解阻**

`app/api/routes.py` 在 `set_rag_chain` 之后追加：

```python
# 并发闸门（在 main.py lifespan 中初始化）
_concurrency_gate: asyncio.Semaphore | None = None


def get_concurrency_gate() -> asyncio.Semaphore | None:
    """获取全局并发闸门"""
    return _concurrency_gate


def set_concurrency_gate(gate: asyncio.Semaphore | None):
    """设置全局并发闸门"""
    global _concurrency_gate
    _concurrency_gate = gate


async def _acquire_gate(gate: asyncio.Semaphore) -> bool:
    """获取闸门许可，超时返回 False"""
    settings = get_settings()
    try:
        await asyncio.wait_for(
            gate.acquire(), timeout=settings.request_queue_timeout
        )
        return True
    except asyncio.TimeoutError:
        return False
```

替换 `chat` 端点与 `_stream_response`：

```python
@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口 - 支持普通和流式两种模式。

    并发闸门限制同时处理的请求数，超出排队，排队超时返回 503。
    同步阻塞的 chain 调用通过 asyncio.to_thread 移出事件循环。
    """
    chain = get_rag_chain()
    chat_history = _build_chat_history(request.chat_history)

    if request.stream:
        return EventSourceResponse(
            _stream_response(chain, request, chat_history)
        )

    gate = get_concurrency_gate()
    if gate is not None:
        if not await _acquire_gate(gate):
            raise HTTPException(
                status_code=503, detail="服务繁忙，请求排队超时，请稍后重试"
            )
        try:
            response = await asyncio.to_thread(
                chain.invoke,
                question=request.question,
                chat_history=chat_history,
                top_k=request.top_k,
            )
        finally:
            gate.release()
    else:
        response = await asyncio.to_thread(
            chain.invoke,
            question=request.question,
            chat_history=chat_history,
            top_k=request.top_k,
        )

    return _build_chat_response(response)


async def _stream_response(
    chain: RAGChain,
    request: ChatRequest,
    chat_history: list,
) -> AsyncGenerator[dict, None]:
    """SSE 流式响应生成器（闸门在流结束后释放）"""
    gate = get_concurrency_gate()
    if gate is not None:
        if not await _acquire_gate(gate):
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "服务繁忙，请求排队超时，请稍后重试"},
                    ensure_ascii=False,
                ),
            }
            return

    try:
        event_iter = chain.invoke_stream(
            question=request.question,
            chat_history=chat_history,
            top_k=request.top_k,
        )
        while True:
            # 同步生成器逐事件移出事件循环，None 为耗尽哨兵
            event = await asyncio.to_thread(next, event_iter, None)
            if event is None:
                break
            if event["type"] == "cache_hit":
                # ……以下事件格式化逻辑与原实现完全一致，逐分支保留……
```

（`_stream_response` 内 4 个事件分支的格式化代码保持原样，仅把外层 `for event in chain.invoke_stream(...)` 改为上面的 `while True` + `asyncio.to_thread(next, event_iter, None)` 模式，并在 `finally` 中释放闸门：）

```python
    finally:
        if gate is not None:
            gate.release()
```

- [ ] **Step 4: main.py lifespan 创建闸门**

`main.py` 顶部 import 处把 `from app.api.routes import router, set_rag_chain` 改为：

```python
from app.api.routes import router, set_rag_chain, set_concurrency_gate
```

lifespan 中 `set_rag_chain(rag_chain)` 之后插入：

```python
    # 并发闸门：限制同时处理的 chat 请求数，防止高并发打爆 LLM
    import asyncio
    set_concurrency_gate(asyncio.Semaphore(settings.max_concurrent_requests))
    logger.info(f"并发闸门: max_concurrent_requests={settings.max_concurrent_requests}")
```

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/test_concurrency_gate.py -q`
Expected: PASS

- [ ] **Step 6: 全量回归 + Commit**

Run: `uv run pytest tests/ -q`
Expected: 全绿

```bash
git add app/api/routes.py main.py tests/test_concurrency_gate.py
git commit -m "feat: /api/chat 并发闸门(Semaphore+503) + to_thread 事件循环解阻"
```

---

### Task 10: CMRC 检索命中率评估脚本

**Files:**
- Create: `run_retrieval_eval.py`

**Interfaces:**
- Consumes: `data/eval_dataset_cmrc.json`（结构：`{"samples": [{"id", "question", "ground_truth", "metadata": {"source", ...}}]}`）、`RAGChain.retrieve(question)`
- Produces: `data/eval_report_cmrc.json`（结构对齐现有报告：`hit_count/hit_rate/avg_keyword_coverage/avg_retrieval_ms/details[]`）

- [ ] **Step 1: 实现脚本**

创建 `run_retrieval_eval.py`：

```python
"""检索级评估：CMRC 测试集命中率 + 关键词覆盖率（无 LLM judge，只评估检索）

用法: uv run python run_retrieval_eval.py [--dataset data/eval_dataset_cmrc.json]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from app.generation.chain import RAGChain

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def keyword_coverage(question: str, ground_truth: str, documents) -> float:
    """ground_truth 中的关键词在检索结果中的覆盖率"""
    keywords = set(re.findall(r"[一-鿿]{2,}|\d+", ground_truth))
    if not keywords:
        return 1.0
    context = " ".join(d.page_content for d in documents)
    hits = sum(1 for kw in keywords if kw in context)
    return hits / len(keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset_cmrc.json")
    parser.add_argument("--output", default="data/eval_report_cmrc.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = dataset["samples"]

    chain = RAGChain()
    try:
        chain.sparse_retriever.build_index()
    except Exception as e:
        logger.warning(f"BM25 索引构建跳过: {e}")

    details = []
    hit_count = 0
    total_coverage = 0.0
    total_ms = 0.0

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        expected_source = sample.get("metadata", {}).get("source", "")

        t0 = time.time()
        result = chain.retrieve(question)
        ms = (time.time() - t0) * 1000

        coverage = keyword_coverage(question, ground_truth, result.documents)
        source_hit = any(
            expected_source and expected_source in d.metadata.get("source", "")
            for d in result.documents
        )
        ok = coverage >= 0.5 or source_hit

        hit_count += int(ok)
        total_coverage += coverage
        total_ms += ms
        details.append({"q": question, "rate": round(coverage, 4), "ok": ok, "ms": ms})
        print(f"[{i}/{len(samples)}] {'✓' if ok else '✗'} {question[:40]} "
              f"coverage={coverage:.2f} {ms:.0f}ms")

    n = len(samples)
    report = {
        "dataset": "cmrc2018",
        "num_samples": n,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / n, 4),
        "avg_keyword_coverage": round(total_coverage / n, 4),
        "avg_retrieval_ms": round(total_ms / n, 1),
        "details": details,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n命中率: {report['hit_rate']:.2%}  覆盖率: {report['avg_keyword_coverage']:.2%}  "
          f"平均耗时: {report['avg_retrieval_ms']:.0f}ms -> {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟运行（需要真实模型 + API key）**

Run: `uv run python run_retrieval_eval.py`
Expected: 输出 31 题逐题结果与汇总，写入 `data/eval_report_cmrc.json`。此步需要本地模型缓存与 `.env` API key；若环境不满足则标记跳过并记录原因。

- [ ] **Step 3: Commit**

```bash
git add run_retrieval_eval.py
git commit -m "feat: CMRC检索命中率评估脚本（补回丢失的评估工具）"
```

---

### Task 11: 全量质量验证（里程碑评估）

**Files:**
- Modify: `data/eval_report.json`、`data/eval_report_cmrc.json`（重新生成）
- Create: `data/concurrency_bench_after.log`

**Interfaces:**
- Consumes: `run_eval.py`、`run_retrieval_eval.py`（Task 10）、`run_concurrency_bench.py`、运行中的服务
- Produces: 改动后 RAGAS 四维报告 + CMRC 命中率报告 + 并发压测对比数据

- [ ] **Step 1: 快速回归（零 LLM）**

确认 `quick_evaluate()` 可用于迭代回归（`app/evaluation/metrics.py:416`）。用 Python REPL 对 3-5 个典型问题跑 retrieve + quick_evaluate，确认无异常。

- [ ] **Step 2: 全量 RAGAS 评估**

Run: `uv run python run_eval.py`（约 30 分钟，消耗 qwen API 额度）
Expected: `data/eval_report.json` 中 **context_recall ≥ 0.90**，faithfulness / precision 相对基线（`data/eval_report_baseline.json`：0.9819 / 0.9495）不回退超过 0.01。若不达标：用报告中的 details 定位失败题目归因（漏召回 / 门控误判 / 补救帮倒忙），修复后重跑。

- [ ] **Step 3: CMRC 检索评估**

Run: `uv run python run_retrieval_eval.py`
Expected: hit_rate ≥ 基线 0.9032

- [ ] **Step 4: 并发压测对比**

```bash
# 终端1: 启动服务
uv run python main.py
# 终端2: 压测
uv run python run_concurrency_bench.py | tee data/concurrency_bench_after.log
```

Expected: 与 `concurrency_output.log`（基线：并发=10 错误率 50%、P50 80s）对比，错误率显著下降（503 快速失败 + 排队替代超时）；将前后数据整理进面试文档（Task 12）。

- [ ] **Step 5: Commit**

```bash
git add data/eval_report.json data/eval_report_cmrc.json data/concurrency_bench_after.log
git commit -m "test: 里程碑全量评估 - RAGAS四维 + CMRC命中率 + 并发压测对比"
```

---

### Task 12: 文档与面试素材更新

**Files:**
- Modify: `docs/interview_guide.md`、`README.md`、`.env.example`
- Modify: `.gitignore`（若 `.env` 未被忽略）

**Interfaces:**
- Consumes: Task 11 的验证数据
- Produces: 4 条设计决策记录；同步的 README/.env.example；密钥安全确认

- [ ] **Step 1: 密钥安全检查**

```bash
git ls-files | grep -x ".env" && echo "危险: .env 已入库" || echo "安全: .env 未入库"
grep -qx ".env" .gitignore || echo ".env" >> .gitignore
```

若 `.env` 已入库：`git rm --cached .env` 并提醒用户轮换 API key。

- [ ] **Step 2: 更新 interview_guide.md**

- 把 Graph RAG 从"下一步/扩展路线"挪到已实现特性
- 新增「设计决策记录」一节，写入 4 条（每条含：决策、备选方案、权衡、验证数据）：
  1. 为什么摘要召回进 RRF 而不是前置路由（路由是硬决策会丢信息，RRF 让摘要路作为软投票参与，单路失败可降级；验证：Task 11 消融对比）
  2. 门控为什么投机并行（门控与改写无数据依赖，并行浪费一次改写 LLM 调用换 2-5s；闲聊场景占比低，期望收益为正）
  3. 补救链路为什么必须过 rerank（HyDE 假设文档召回的结果噪声大，不过 rerank 会把噪声直接灌进生成；补救最多 1 次防循环）
  4. 并发闸门为什么用 Semaphore 而不是队列中间件（单机部署、学习项目，Semaphore 零依赖够用；附压测前后数据）

- [ ] **Step 3: 更新 README.md 与 .env.example**

- README：模型栈改为 qwen3.8-max-preview + bge-small-zh-v1.5 + bge-reranker-base；前端 5 Tab；模块结构补充 pipeline.py / graph_* / parent_child / crag / cache / observability；新增配置开关说明
- `.env.example`：与 `config.py` 全部字段对齐（含新增的 5 个配置项）

- [ ] **Step 4: Commit**

```bash
git add docs/interview_guide.md README.md .env.example .gitignore
git commit -m "docs: 面试指南设计决策记录 + README/.env.example 与实现同步"
```

---

## Self-Review 记录

- **Spec 覆盖**：接线 4 项 → Task 4-8；并发中度方案 → Task 9（Semaphore + to_thread + 投机并行在 Task 4 落地、Task 6 验证）；测试 → 各 Task TDD + Task 8 test_api 修复；评估 → Task 10/11；文档 → Task 12。✅
- **类型一致性**：`RetrievalResult` 在 pipeline.py 定义、chain.py re-export；`summary_results` / `gate_skipped` 字段在 Task 4 定义、Task 5/6 消费；`set_concurrency_gate` Task 9 定义并消费。✅
- **Placeholder 扫描**：Task 7 的 `test_remediate_failure_keeps_original` 标注了可替换方案（脆弱 mock 的备选），属有意说明而非占位。✅
