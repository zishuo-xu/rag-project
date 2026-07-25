"""延迟治理测试（2026-07-26）：Deadline 时延预算 / F2/F3 预算熔断 / router 前置短路 / max_tokens 封顶

背景：full 模式 42.6s 均值实为单点 486s 离群拖拽（14/15 样本均值 10.9s）。
治理主线：消灭离群尾（全局预算 + 超时收紧）+ 结构优化（router 前置跳过分解路径无用 transform）。
全离线 mock。
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.retrieval.deadline import Deadline

from test_pipeline import _make_pipeline, _doc


# ============ Deadline 单元 ============

def test_deadline_not_exceeded_within_budget():
    t = {"v": 0.0}
    d = Deadline(25000, clock=lambda: t["v"])
    t["v"] = 1.0
    assert not d.exceeded()
    assert d.elapsed_ms() == 1000.0


def test_deadline_exceeded_after_budget():
    calls = iter([0.0, 30.0])
    d = Deadline(25000, clock=lambda: next(calls))
    assert d.exceeded()


def test_deadline_disabled_with_nonpositive_budget():
    d = Deadline(0, clock=lambda: 9999.0)
    assert not d.exceeded()
    assert Deadline(-1, clock=lambda: 9999.0).check_skip("X") is False


def test_check_skip_records_stage():
    calls = iter([0.0, 30.0, 30.0])
    d = Deadline(25000, clock=lambda: next(calls))
    assert d.check_skip("F2_iterative") is True
    assert d.check_skip("F3_regen") is True
    assert d.skipped == ["F2_iterative", "F3_regen"]


# ============ F2 迭代检索预算熔断 ============

def _f2_pipeline(budget_ms=25000):
    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_iterative_retrieval = True
    s.latency_budget_ms = budget_ms
    s.use_query_router = False
    s.use_decomposition = False
    s.use_contextual_chunks = False
    # CRAG 首评 ambiguous → 本应触发迭代精化
    mocks["crag_evaluator"].evaluate_relevance.return_value = ("ambiguous", [1], "部分相关")
    mocks["query_transformer"].refine.return_value = "精化查询"
    return pipe, mocks


def test_f2_skipped_when_budget_exceeded(monkeypatch):
    """超预算时 F2 迭代整体跳过，记 budget_skipped，不消耗 refine LLM。"""
    calls = iter([0.0] + [30.0] * 50)  # start=0，其后 30s → 超 25s 预算
    monkeypatch.setattr("app.retrieval.deadline.time.monotonic", lambda: next(calls))
    pipe, mocks = _f2_pipeline()
    result = pipe.run("问题")
    assert result.iterative_stop_reason == "budget_skipped"
    assert "F2_iterative" in result.budget_skipped
    mocks["query_transformer"].refine.assert_not_called()


def test_f2_runs_when_budget_ok(monkeypatch):
    """预算内 F2 正常迭代（对照）。"""
    monkeypatch.setattr("app.retrieval.deadline.time.monotonic", lambda: 0.0)
    pipe, mocks = _f2_pipeline()
    result = pipe.run("问题")
    assert result.iterative_stop_reason != "budget_skipped"
    assert result.budget_skipped == []


# ============ router 前置：multi_hop 分解路径跳过无用 transform ============

def test_router_first_skips_transform_for_multihop():
    """multi_hop + 分解开启：transform（multi_query）被跳过，decompose 照常触发。"""
    from app.retrieval.router import QueryRouter
    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_query_router = True
    s.use_decomposition = True
    s.use_crag_gate = False
    s.latency_budget_ms = 0
    mocks["query_router"] = QueryRouter(settings=s)
    pipe.query_router = mocks["query_router"]
    mocks["query_transformer"].decompose.return_value = SimpleNamespace(
        sub_questions=["子问题1", "子问题2"], chain=False
    )
    result = pipe.run("范廷颂担任总主教的那个教区在哪里？")
    assert result.query_type == "multi_hop"
    mocks["query_transformer"].transform.assert_not_called()   # 投机改写被短路
    mocks["query_transformer"].decompose.assert_called_once()
    assert result.decomposed_subqueries == ["子问题1", "子问题2"]


def test_router_first_backfills_transform_when_decompose_fails():
    """分解未产出 >1 子问题时回退普通召回，transform 延迟补跑（不丢召回质量）。"""
    from app.retrieval.router import QueryRouter
    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_query_router = True
    s.use_decomposition = True
    s.use_crag_gate = False
    s.latency_budget_ms = 0
    pipe.query_router = QueryRouter(settings=s)
    mocks["query_transformer"].decompose.return_value = SimpleNamespace(
        sub_questions=["单一问题"], chain=False
    )
    pipe.run("范廷颂担任总主教的那个教区在哪里？")
    mocks["query_transformer"].transform.assert_called_once()  # 延迟补跑


def test_router_first_keeps_transform_for_single_hop():
    """非 multi_hop 查询行为不变：transform 照常执行。"""
    from app.retrieval.router import QueryRouter
    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_query_router = True
    s.use_decomposition = True
    s.use_crag_gate = False
    s.latency_budget_ms = 0
    pipe.query_router = QueryRouter(settings=s)
    pipe.run("什么是缓存穿透？")
    mocks["query_transformer"].transform.assert_called_once()


# ============ max_tokens / 超时收紧 ============

def test_query_transform_llm_budget_capped():
    """改写 LLM：max_tokens=512 封顶 + 超时 20s + 重试 1 次（原 30s/2 次/无界）。"""
    with patch("app.retrieval.query_transform.ChatOpenAI") as m:
        from app.retrieval.query_transform import QueryTransformer
        QueryTransformer()
        kw = m.call_args.kwargs
        assert kw["max_tokens"] == 512
        assert kw["request_timeout"] == 20
        assert kw["max_retries"] == 1


# ============ F3 重生成预算熔断 ============

def _bare_chain(faithful=False):
    """RAGChain.__new__ 裸构造（绕开重组件初始化），仅测 _generate_faithful 逻辑。"""
    from app.generation.chain import RAGChain
    chain = RAGChain.__new__(RAGChain)
    chain._settings = SimpleNamespace(faithfulness_max_regen=1)
    chain.faithfulness_checker = MagicMock()
    chain.faithfulness_checker.check.return_value = SimpleNamespace(
        faithful=faithful, score=0.3
    )
    chain.generate = MagicMock(return_value="答案")
    return chain


def test_f3_regen_skipped_when_budget_exceeded():
    """超预算时跳过严格重生成（省 2 次串行 LLM），记 budget_skipped。"""
    chain = _bare_chain()
    calls = iter([0.0, 99.0, 99.0])  # start=0，其后 99s → 超 1s 预算
    deadline = Deadline(1000, clock=lambda: next(calls))
    answer, faithful, score, regenerated = chain._generate_faithful(
        "问题", [_doc("证据")], deadline=deadline
    )
    assert regenerated is False
    assert "F3_regen" in deadline.skipped
    assert chain.generate.call_count == 1  # 仅首次生成，无重生成


def test_f3_regen_runs_when_budget_ok():
    """预算内忠实度不足仍触发重生成（对照）。"""
    chain = _bare_chain()
    deadline = Deadline(100000, clock=lambda: 0.0)
    _, _, _, regenerated = chain._generate_faithful(
        "问题", [_doc("证据")], deadline=deadline
    )
    assert regenerated is True
    assert chain.generate.call_count == 2
    assert deadline.skipped == []
