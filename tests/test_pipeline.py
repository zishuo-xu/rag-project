"""RetrievalPipeline 阶段测试（全 mock，离线运行）"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.retrieval.pipeline import RetrievalPipeline, ALL_CHANNELS
from app.retrieval.router import RoutingDecision


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

    router = MagicMock()
    router.route.return_value = RoutingDecision("factual", reason="默认")

    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 20
    settings.recall_max_workers = 6
    settings.use_summary_recall = True
    settings.use_crag_gate = True
    # RAG 2.0 新特性：基线 mock 默认全关，保护既有行为；专项测试再单独打开
    settings.use_autocut = False
    settings.autocut_min_docs = 2
    settings.use_iterative_retrieval = False
    settings.max_retrieval_iterations = 2
    settings.use_query_router = False

    kwargs = dict(
        indexer=indexer, dense_retriever=dense, sparse_retriever=sparse,
        reranker=reranker, query_transformer=transformer,
        graph_retriever=graph, parent_child_retriever=pc,
        crag_evaluator=crag, query_router=router, settings=settings,
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


def test_remediate_full_pipeline_on_incorrect():
    """incorrect 时补救走完整 mini-pipeline：HyDE + dense/sparse + RRF + rerank"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("incorrect", [], "无关")
    result = pipe.run("问题")
    # HyDE 改写被调用
    hyde_calls = [
        c for c in mocks["query_transformer"].transform.call_args_list
        if len(c.args) > 1 and c.args[1] == "hyde"
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


def test_remediate_returns_empty_when_recall_empty():
    """remediate() 单测：HyDE 双路召回均为空时返回空列表"""
    pipe, mocks = _make_pipeline()
    mocks["query_transformer"].transform.side_effect = (
        lambda q, strategy="multi_query": [f"hyde:{q}"]
    )
    mocks["dense_retriever"].retrieve.side_effect = lambda *a, **k: []
    mocks["sparse_retriever"].retrieve.side_effect = lambda *a, **k: []
    assert pipe.remediate("问题", 5) == []


def test_autocut_truncates_on_clear_cliff():
    """F1 use_autocut=True：重排分数断崖时动态截断，保留少于 top_k"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_autocut = True
    mocks["settings"].autocut_min_docs = 2

    def _scored_rerank(q, docs, top_k=None):
        scores = [0.95, 0.90, 0.85, 0.30, 0.25, 0.20]  # 3 高 3 低断崖
        out = []
        for i, d in enumerate(docs[:6]):
            d.metadata["rerank_score"] = scores[i]
            out.append(d)
        return out[: top_k or len(out)]

    mocks["reranker"].rerank.side_effect = _scored_rerank
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: [_doc(f"d{q}{j}") for j in range(3)]
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: [_doc(f"s{q}{j}") for j in range(3)]
    )
    result = pipe.run("问题")
    assert len(result.documents) == 3        # 断崖在第3篇后 → 保留3篇 (< top_k=5)
    assert result.pre_autocut_count >= 6     # 截断前候选 ≥6


def test_autocut_flat_scores_fallback_top_k():
    """F1 use_autocut=True：分数全等（无膝点）→ 回退 top_k"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_autocut = True
    mocks["settings"].autocut_min_docs = 2

    def _flat_rerank(q, docs, top_k=None):
        for d in docs:
            d.metadata["rerank_score"] = 0.5
        return docs[: top_k or len(docs)]

    mocks["reranker"].rerank.side_effect = _flat_rerank
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: [_doc(f"d{q}{j}") for j in range(4)]
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: [_doc(f"s{q}{j}") for j in range(4)]
    )
    result = pipe.run("问题")
    assert 2 <= len(result.documents) <= 5   # 回退 top_k=5，满足下界


def test_autocut_disabled_keeps_fixed_top_k():
    """F1 use_autocut=False：保持固定 top_k 截断（回归保护）"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_autocut = False
    result = pipe.run("问题")
    assert len(result.documents) <= 5


def test_query_router_sets_query_type():
    """F4 use_query_router=True：记录路由判定的 query_type"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_query_router = True
    result = pipe.run("什么是缓存穿透？")
    mocks["query_router"].route.assert_called_once_with("什么是缓存穿透？")
    assert result.query_type == "factual"


def test_query_router_widens_top_k():
    """F4 路由返回更大 top_k → 最终文档数上限提升"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_query_router = True
    mocks["query_router"].route.return_value = RoutingDecision(
        "comparative", top_k=8, autocut_min_docs=3, reason="对比型"
    )
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: [_doc(f"d{q}{j}") for j in range(6)]
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: [_doc(f"s{q}{j}") for j in range(6)]
    )
    result = pipe.run("A和B的区别")
    assert result.query_type == "comparative"
    assert len(result.documents) <= 8


def test_query_router_disabled_no_route_call():
    """F4 use_query_router=False：不调用路由器，query_type 为空（回归保护）"""
    pipe, mocks = _make_pipeline()
    mocks["settings"].use_query_router = False
    result = pipe.run("问题")
    mocks["query_router"].route.assert_not_called()
    assert result.query_type == ""


def test_remediate_failure_keeps_original():
    """补救返回空时保留原结果（run() 中 retry_docs 为空分支）"""
    pipe, mocks = _make_pipeline()
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("incorrect", [], "无关")
    # 主检索改写正常返回变体；HyDE 改写返回带标记的查询
    mocks["query_transformer"].transform.side_effect = (
        lambda q, strategy="multi_query": (
            [f"hyde:{q}"] if strategy == "hyde" else [q, f"{q}变体"]
        )
    )
    # HyDE 查询在 dense/sparse 两路均召回为空 -> remediate 返回空
    mocks["dense_retriever"].retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: (
            [] if q.startswith("hyde:") else [_doc(f"dense:{q}")]
        )
    )
    mocks["sparse_retriever"].retrieve.side_effect = (
        lambda q, top_k=10: (
            [] if q.startswith("hyde:") else [_doc(f"sparse:{q}")]
        )
    )
    result = pipe.run("问题")
    assert result.crag_grade == "incorrect"
    assert result.crag_action == "补救失败，保留原结果"
    assert len(result.documents) > 0
