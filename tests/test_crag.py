"""CRAG 评估器测试（零 LLM 分级 + 门控规则 + 数字校验）

修复背景：crag_relevance_threshold 曾是死旋钮（赋值后判级用字面量 0.5/0.3），
本组测试锁住「配置阈值真实生效」与分级边界。
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.crag import CRAGEvaluator


def _docs_with_score(*scores):
    return [
        Document(page_content="内容", metadata={"rerank_score": s})
        for s in scores
    ]


# ============ 分级：sigmoid(top1) 判级 ============

def test_grade_correct_default_threshold():
    ev = CRAGEvaluator()
    grade, _, reason = ev.evaluate_relevance("q", _docs_with_score(1.0))
    assert grade == "correct"  # sigmoid(1.0)=0.731 ≥ 0.5
    assert "rerank_sigmoid" in reason


def test_grade_ambiguous_default_threshold():
    ev = CRAGEvaluator()
    grade, _, _ = ev.evaluate_relevance("q", _docs_with_score(-0.5))
    assert grade == "ambiguous"  # sigmoid(-0.5)=0.378 ∈ [0.3, 0.5)


def test_grade_incorrect_default_threshold():
    ev = CRAGEvaluator()
    grade, _, _ = ev.evaluate_relevance("q", _docs_with_score(-2.0))
    assert grade == "incorrect"  # sigmoid(-2.0)=0.119 < 0.3


def test_configured_threshold_is_live(monkeypatch):
    """配置阈值真实生效（修复死旋钮）：0.7 阈值下 0.5 判级从 correct 降为 ambiguous。"""
    monkeypatch.setattr(
        "app.retrieval.crag.get_settings",
        lambda: MagicMock(crag_relevance_threshold=0.7),
    )
    ev = CRAGEvaluator()
    assert ev.threshold == 0.7
    assert ev.incorrect_threshold == 0.5
    # sigmoid(0.2)=0.55：默认 0.5 阈值判 correct，0.7 阈值判 ambiguous
    grade, _, _ = ev.evaluate_relevance("q", _docs_with_score(0.2))
    assert grade == "ambiguous"
    # sigmoid(1.0)=0.731 ≥ 0.7 仍为 correct
    grade2, _, _ = ev.evaluate_relevance("q", _docs_with_score(1.0))
    assert grade2 == "correct"


def test_grade_no_scores_defaults_pass():
    """无 rerank 分数默认通过（分解路径纯 RRF 结果等场景）。"""
    ev = CRAGEvaluator()
    docs = [Document(page_content="无分数", metadata={})]
    grade, indices, _ = ev.evaluate_relevance("q", docs)
    assert grade == "correct"
    assert indices == [0]


def test_grade_empty_docs_incorrect():
    ev = CRAGEvaluator()
    grade, _, _ = ev.evaluate_relevance("q", [])
    assert grade == "incorrect"  # noqa: 空输入判 incorrect


def test_grade_uses_top1_not_average():
    """分级看最好的一篇（top1），不看平均——已降序输入取首篇。"""
    ev = CRAGEvaluator()
    grade, _, _ = ev.evaluate_relevance("q", _docs_with_score(1.0, -5.0, -5.0))
    assert grade == "correct"


# ============ 门控：should_retrieve（零 LLM 规则） ============

def test_should_retrieve_chitchat_skipped():
    ev = CRAGEvaluator()
    assert ev.should_retrieve("你好")[0] is False
    assert ev.should_retrieve("谢谢")[0] is False
    assert ev.should_retrieve("您好，请问")[0] is False


def test_should_retrieve_real_question():
    ev = CRAGEvaluator()
    assert ev.should_retrieve("范廷颂哪一年成为主教？")[0] is True


def test_should_retrieve_too_short():
    ev = CRAGEvaluator()
    assert ev.should_retrieve(" ")[0] is False


# ============ 数字型答案校验 ============

def test_numeric_validation_requires_digits():
    ev = CRAGEvaluator()
    q = "这座桥建于哪一年？"
    no_digits = [Document(page_content="这座桥历史悠久，建于古代。", metadata={})]
    has_digits = [Document(page_content="这座桥建于1963年。", metadata={})]
    assert ev.validate_numeric_answer(q, no_digits) is False
    assert ev.validate_numeric_answer(q, has_digits) is True


def test_numeric_validation_skips_non_numeric_question():
    ev = CRAGEvaluator()
    docs = [Document(page_content="没有任何数字", metadata={})]
    assert ev.validate_numeric_answer("这座桥在哪里？", docs) is True
