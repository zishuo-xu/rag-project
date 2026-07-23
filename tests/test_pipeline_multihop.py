"""F6b pipeline 多跳分支（mock，离线；新开关默认关以保护既有测试）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_transform import Decomposition


def _doc(cid):
    return Document(page_content=f"内容{cid}", metadata={"chunk_id": cid})


def _make_pipeline(settings_overrides=None):
    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 20
    settings.recall_max_workers = 4
    settings.use_summary_recall = False
    settings.use_crag_gate = False          # 关门控，简化
    settings.use_query_router = True
    settings.use_autocut = False
    settings.autocut_min_docs = 2
    settings.use_iterative_retrieval = False  # 关 F2，隔离 F6b
    settings.use_decomposition = True
    settings.decomposition_max_subquestions = 4
    settings.decomposition_max_hops = 3
    for k, v in (settings_overrides or {}).items():
        setattr(settings, k, v)

    p = RetrievalPipeline.__new__(RetrievalPipeline)
    p.indexer = MagicMock()
    p.dense_retriever = MagicMock()
    p.sparse_retriever = MagicMock()
    p.reranker = None
    p.query_transformer = MagicMock()
    p.graph_retriever = None
    p.parent_child_retriever = None
    p.crag_evaluator = None               # 关 CRAG，隔离
    p.query_router = MagicMock()
    p.query_router.route.return_value = MagicMock(
        query_type="multi_hop", top_k=None, autocut_min_docs=None, reason="多跳"
    )
    p._settings = settings
    return p


def test_multihop_parallel_decomposition_merges_subqueries():
    p = _make_pipeline()
    p.query_transformer.decompose.return_value = Decomposition(
        sub_questions=["子问题1", "子问题2"], chain=False
    )
    # recall 按子问题返回不同文档
    p.recall = MagicMock(side_effect=[
        {"dense": [_doc("a")], "sparse": []},
        {"dense": [_doc("b")], "sparse": []},
    ])
    result = p.run("范廷颂担任总主教的那个教区在哪里？")
    assert result.decomposed_subqueries == ["子问题1", "子问题2"]
    assert result.decomposition_chain is False
    fused_ids = {d.metadata["chunk_id"] for d in result.fused_results}
    assert {"a", "b"} <= fused_ids


def test_multihop_single_subquery_falls_back_to_normal_recall():
    p = _make_pipeline()
    p.query_transformer.decompose.return_value = Decomposition(
        sub_questions=["原问题"], chain=False
    )
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("简单问题")
    assert result.decomposed_subqueries == []  # 未触发分解合并
    assert p.recall.call_count == 1


def test_multihop_disabled_goes_normal():
    p = _make_pipeline({"use_decomposition": False})
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("多跳问题")
    assert result.decomposed_subqueries == []
    p.query_transformer.decompose.assert_not_called()


def test_non_multihop_does_not_decompose():
    p = _make_pipeline()
    p.query_router.route.return_value = MagicMock(
        query_type="factual", top_k=None, autocut_min_docs=None, reason="事实"
    )
    p.recall = MagicMock(return_value={"dense": [_doc("x")], "sparse": []})
    result = p.run("事实问题")
    p.query_transformer.decompose.assert_not_called()
    assert result.decomposed_subqueries == []
