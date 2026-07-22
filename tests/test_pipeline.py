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
