"""Self-RAG 迭代检索测试（全 mock，离线运行）

终止判断标志（非资源兜底）：
① 充分性：CRAG grade==correct（证据足以回答）→ sufficient
② 收敛性：本轮无新增相关文档（精化已无效）→ converged
③ 安全兜底：max_retrieval_iterations 硬上限 → max_iterations
"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.retrieval.pipeline import RetrievalPipeline


def _doc(content):
    return Document(page_content=content, metadata={"chunk_id": content, "source": "t.md"})


def _make_iter_pipeline(evaluate_seq, dense_map, refine_seq=None, max_iter=2):
    """构造迭代检索测试用全 mock pipeline。

    evaluate_seq: crag.evaluate_relevance 的返回序列（控制评级演变）
    dense_map: {query: [doc,...]} 精化查询→新文档 的映射
    refine_seq: query_transformer.refine 的返回序列
    """
    indexer = MagicMock()
    indexer.embeddings.embed_documents.return_value = [[0.1]]
    indexer.hierarchical_search.return_value = []

    dense = MagicMock()
    dense.retrieve.side_effect = (
        lambda q, top_k=10, embedding=None: dense_map.get(q, [])
    )
    sparse = MagicMock()
    sparse.retrieve.return_value = []  # 简化：sparse 不贡献

    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, docs, top_k=None: docs[: top_k or 5]

    transformer = MagicMock()
    transformer.transform.side_effect = lambda q, strategy="multi_query": [q]
    if refine_seq is not None:
        transformer.refine.side_effect = refine_seq
    else:
        transformer.refine.return_value = "refined_q"

    graph = MagicMock()
    graph.retrieve.return_value = []
    pc = MagicMock()
    pc.has_index.return_value = False

    crag = MagicMock()
    crag.should_retrieve.return_value = (True, "需要检索")
    crag.evaluate_relevance.side_effect = evaluate_seq

    settings = MagicMock()
    settings.retrieval_top_k = 5
    settings.rerank_top_n = 20
    settings.recall_max_workers = 6
    settings.use_summary_recall = False
    settings.use_crag_gate = False           # 跳过门控简化
    settings.use_autocut = False
    settings.autocut_min_docs = 2
    settings.use_query_router = False
    settings.use_iterative_retrieval = True  # 打开迭代检索
    settings.max_retrieval_iterations = max_iter

    pipe = RetrievalPipeline(
        indexer=indexer, dense_retriever=dense, sparse_retriever=sparse,
        reranker=reranker, query_transformer=transformer,
        graph_retriever=graph, parent_child_retriever=pc,
        crag_evaluator=crag, settings=settings,
    )
    return pipe, {"transformer": transformer, "crag": crag, "dense": dense}


def test_first_pass_correct_no_iteration():
    """首检 correct → sufficient，0 迭代，不精化"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("correct", [1], "相关")],
        dense_map={"问题": [_doc("a")]},
    )
    result = pipe.run("问题")
    assert result.iterative_stop_reason == "sufficient"
    assert result.iterations_used == 0
    assert result.crag_grade == "correct"  # 首检即 correct，不应误标 recovered
    m["transformer"].refine.assert_not_called()


def test_ambiguous_then_new_evidence_then_correct():
    """ambiguous → 精化召回到新证据 → correct：sufficient，1 轮"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("ambiguous", [1], "部分"), ("correct", [1, 2], "够了")],
        dense_map={"问题": [_doc("orig")], "refined_q": [_doc("new")]},
        refine_seq=["refined_q"],
    )
    result = pipe.run("问题")
    assert result.iterative_stop_reason == "sufficient"
    assert result.iterations_used == 1
    assert result.crag_grade == "recovered"
    # 新证据被并入
    assert any(d.page_content == "new" for d in result.documents)


def test_converged_when_no_new_evidence():
    """精化召回不到新文档（全是已有）→ converged，0 轮"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("ambiguous", [1], "部分")],
        # 精化查询召回的文档 chunk_id 与初始相同 → 无新增
        dense_map={"问题": [_doc("orig")], "refined_q": [_doc("orig")]},
        refine_seq=["refined_q"],
    )
    result = pipe.run("问题")
    assert result.iterative_stop_reason == "converged"
    assert result.iterations_used == 0


def test_max_iterations_safety_bound():
    """一直 ambiguous 且每轮都有新证据 → 撞硬上限 max_iterations"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("ambiguous", [1], "部分")] * 5,  # 始终 ambiguous
        dense_map={
            "问题": [_doc("orig")],
            "rq1": [_doc("n1")],
            "rq2": [_doc("n2")],
        },
        refine_seq=["rq1", "rq2"],
        max_iter=2,
    )
    result = pipe.run("问题")
    assert result.iterative_stop_reason == "max_iterations"
    assert result.iterations_used == 2


def test_iterative_disabled_uses_legacy_crag():
    """关闭迭代检索 → 走原 CRAG 单次补救路径（回归保护）"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("incorrect", [], "无关")],
        dense_map={"问题": [_doc("orig")]},
    )
    pipe._settings.use_iterative_retrieval = False
    # 原路径：incorrect → HyDE 补救（transform hyde）
    m["transformer"].transform.side_effect = (
        lambda q, strategy="multi_query": ([f"hyde:{q}"] if strategy == "hyde" else [q])
    )
    result = pipe.run("问题")
    # 迭代字段未被使用
    assert result.iterative_stop_reason == ""
    assert result.iterations_used == 0
    # 走了 HyDE 补救（transform 被以 hyde 策略调用）
    hyde_calls = [c for c in m["transformer"].transform.call_args_list
                  if len(c.args) > 1 and c.args[1] == "hyde"]
    assert len(hyde_calls) == 1


def test_refine_failure_degrades_gracefully():
    """精化抛异常 → 优雅降级（用原问题兜底），不崩溃"""
    pipe, m = _make_iter_pipeline(
        evaluate_seq=[("ambiguous", [1], "部分"), ("correct", [1], "够")],
        dense_map={"问题": [_doc("orig")], "fallback_q": [_doc("new")]},
    )
    m["transformer"].refine.side_effect = RuntimeError("LLM 超时")
    # _refine_query 捕获异常返回原问题 "问题"；dense("问题")→orig（已存在）→ converged
    result = pipe.run("问题")
    assert result.iterative_stop_reason in ("converged", "max_iterations")
    assert result.documents  # 不崩溃且有结果
