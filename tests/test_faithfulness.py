"""生成忠实度自检测试（全 mock，离线运行）

覆盖两层：
1. FaithfulnessChecker.check：LLM-judge 解析 / 异常放行 / 空答案 / score 截断
2. RAGChain._generate_faithful：不忠实→严格重生成 / 重生成上限 / 关闭开关回归
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.generation.faithfulness import FaithfulnessChecker, FaithfulnessResult
from app.generation.chain import RAGChain


def _mock_llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def _doc(c):
    return Document(page_content=c, metadata={"source": "t.md"})


# ---- FaithfulnessChecker.check ----

def test_full_supported():
    checker = FaithfulnessChecker(
        llm=_mock_llm('{"score": 1.0, "unsupported": [], "reason": "全部支撑"}'),
        threshold=0.7,
    )
    r = checker.check("问题", [_doc("上下文")], "答案")
    assert r.faithful is True
    assert r.score == 1.0


def test_unsupported_below_threshold():
    checker = FaithfulnessChecker(
        llm=_mock_llm('{"score": 0.3, "unsupported": ["编造的细节"], "reason": "含幻觉"}'),
        threshold=0.7,
    )
    r = checker.check("问题", [_doc("上下文")], "答案含编造")
    assert r.faithful is False
    assert r.score == 0.3
    assert "编造的细节" in r.unsupported


def test_llm_exception_returns_none_passthrough():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("timeout")
    checker = FaithfulnessChecker(llm=llm, threshold=0.7)
    r = checker.check("问题", [_doc("上下文")], "答案")
    assert r.faithful is None  # 未知 → 放行，不阻断


def test_unparseable_returns_none():
    checker = FaithfulnessChecker(llm=_mock_llm("这不是JSON"), threshold=0.7)
    r = checker.check("问题", [_doc("上下文")], "答案")
    assert r.faithful is None


def test_empty_answer_faithful():
    checker = FaithfulnessChecker(llm=_mock_llm("{}"), threshold=0.7)
    r = checker.check("问题", [_doc("上下文")], "")
    assert r.faithful is True


def test_score_clamped_to_unit_interval():
    checker = FaithfulnessChecker(
        llm=_mock_llm('{"score": 1.5, "unsupported": [], "reason": ""}'),
        threshold=0.7,
    )
    r = checker.check("问题", [_doc("上下文")], "答案")
    assert r.score == 1.0


# ---- RAGChain._generate_faithful（用 __new__ 绕过重型 __init__） ----

def _bare_chain(max_regen=1):
    chain = RAGChain.__new__(RAGChain)
    chain._settings = MagicMock(faithfulness_max_regen=max_regen)
    chain.faithfulness_checker = MagicMock()
    chain.generate = MagicMock()
    return chain


def test_generate_faithful_no_regen_when_faithful():
    chain = _bare_chain()
    chain.generate.return_value = "好答案"
    chain.faithfulness_checker.check.return_value = FaithfulnessResult(
        faithful=True, score=0.9
    )
    ans, faithful, score, regen = chain._generate_faithful("q", [_doc("c")], None)
    assert ans == "好答案"
    assert faithful is True
    assert regen is False
    chain.generate.assert_called_once()  # 未触发重生成


def test_generate_faithful_regen_on_unfaithful():
    chain = _bare_chain()
    chain.generate.side_effect = ["幻觉答案", "严格答案"]
    chain.faithfulness_checker.check.side_effect = [
        FaithfulnessResult(faithful=False, score=0.3),
        FaithfulnessResult(faithful=True, score=0.9),
    ]
    ans, faithful, score, regen = chain._generate_faithful("q", [_doc("c")], None)
    assert ans == "严格答案"
    assert regen is True
    assert faithful is True
    # 重生成使用 strict=True
    assert chain.generate.call_args_list[1].kwargs.get("strict") is True


def test_generate_faithful_regen_bounded_by_max():
    chain = _bare_chain(max_regen=1)
    chain.generate.return_value = "答案"
    chain.faithfulness_checker.check.return_value = FaithfulnessResult(
        faithful=False, score=0.2
    )
    ans, faithful, score, regen = chain._generate_faithful("q", [_doc("c")], None)
    # 初次 + 1 次重生成 = 2 次 generate（被 max_regen 兜住）
    assert chain.generate.call_count == 2
    assert regen is True
    assert faithful is False  # 达上限仍不忠实，如实返回


def test_generate_faithful_checker_disabled():
    chain = _bare_chain()
    chain.faithfulness_checker = None
    chain.generate.return_value = "答案"
    ans, faithful, score, regen = chain._generate_faithful("q", [_doc("c")], None)
    assert ans == "答案"
    assert faithful is None
    assert regen is False
    chain.generate.assert_called_once()
