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
