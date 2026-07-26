"""语义正确率 answer_correctness 测试（离线，零真实 LLM）

answer_correctness 是端到端的语义裁判（LLM-as-judge），纠偏词法 F1/子串 hit 对
「长句型 gold vs 凝练/改写答案」的结构性苛刻——允许同义改写，只看语义是否等价于参考答案。
本组测试通过 monkeypatch 单一 choke point `_llm_judge` 喂罐头 JSON，覆盖：
解析 / 空输入 / 解析失败兜底 / 异常兜底 / prompt 契约（锁定纠偏意图，防回退）。
"""
import pytest

from app.evaluation import metrics


def test_parses_score(monkeypatch):
    monkeypatch.setattr(metrics, "_llm_judge", lambda p: '{"reasoning": "语义一致", "score": 0.8}')
    assert metrics.answer_correctness("q", "答案A", "参考答案A") == 0.8


def test_empty_answer_returns_zero(monkeypatch):
    called = {"n": 0}

    def _fake(p):
        called["n"] += 1
        return '{"score": 1.0}'

    monkeypatch.setattr(metrics, "_llm_judge", _fake)
    assert metrics.answer_correctness("q", "", "gold") == 0.0
    assert metrics.answer_correctness("q", "   ", "gold") == 0.0
    assert called["n"] == 0  # 空输入直接判 0，不浪费 LLM 调用


def test_empty_gold_returns_zero(monkeypatch):
    monkeypatch.setattr(metrics, "_llm_judge", lambda p: '{"score": 1.0}')
    assert metrics.answer_correctness("q", "答案", "") == 0.0
    assert metrics.answer_correctness("q", "答案", "  ") == 0.0


def test_parse_failure_falls_back_to_half(monkeypatch):
    # 无 JSON、无 score 字段 → _extract_json 返回 None → 兜底 0.5
    monkeypatch.setattr(metrics, "_llm_judge", lambda p: "完全不是 JSON 的乱码输出")
    assert metrics.answer_correctness("q", "答案", "gold") == 0.5


def test_exception_falls_back_to_half(monkeypatch):
    def _boom(p):
        raise RuntimeError("llm down")

    monkeypatch.setattr(metrics, "_llm_judge", _boom)
    assert metrics.answer_correctness("q", "答案", "gold") == 0.5


def test_prompt_contract(monkeypatch):
    """prompt 必须带上 question/gold/answer，并明确『允许改写、不苛求字面』的纠偏指令。"""
    seen = {}

    def _capture(p):
        seen["prompt"] = p
        return '{"score": 0.7}'

    monkeypatch.setattr(metrics, "_llm_judge", _capture)
    metrics.answer_correctness("范廷颂何时出生？", "他生于1963年。", "1963年")
    p = seen["prompt"]
    # 三个输入都进了 prompt
    assert "范廷颂何时出生？" in p
    assert "他生于1963年。" in p
    assert "1963年" in p
    # 纠偏意图：明确允许改写、不苛求字面（这正是相对 char-F1/子串 hit 的改进点）
    assert "不苛求字面" in p
    assert "同义改写" in p
    # 输出契约：严格 JSON + score 键
    assert "score" in p
