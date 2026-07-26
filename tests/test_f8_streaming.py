"""F8 低延迟流式 + 投机忠实度测试（离线，全 mock）

覆盖：
1. token 逐字流出且顺序正确、full 累积
2. checker=None 直接放行（faithful=None，无 correction）
3. 忠实 → 无 correction，final faithful=True
4. 不忠实→严格重生成：发 correction，final answer=重生成、regenerated=True
5. 重生成被 max_regen 兜住
6. 事件顺序：token 先于 correction/final（保证首屏快）
7. 时延预算耗尽 → 流式路径同样跳过重生成（与 chain F3 同源）
"""
from unittest.mock import MagicMock

from app.generation.streaming import speculative_faithful_stream
from app.generation.faithfulness import FaithfulnessResult


def _stream(tokens):
    def fn():
        for t in tokens:
            yield t
    return fn


def _collect(**kwargs):
    return list(speculative_faithful_stream(**kwargs))


def test_tokens_streamed_in_order():
    events = _collect(
        stream_fn=_stream(["你", "好", "世界"]), question="q", context="上下文",
        chat_history=None, checker=None, regen_fn=lambda: "",
    )
    tokens = [e["data"] for e in events if e["type"] == "token"]
    assert tokens == ["你", "好", "世界"]
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["answer"] == "你好世界"


def test_no_checker_passthrough():
    events = _collect(
        stream_fn=_stream(["答", "案"]), question="q", context="上下文",
        chat_history=None, checker=None, regen_fn=lambda: "x",
    )
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["faithful"] is None
    assert final["regenerated"] is False
    assert not [e for e in events if e["type"] == "correction"]


def test_faithful_no_correction():
    checker = MagicMock()
    checker.check.return_value = FaithfulnessResult(faithful=True, score=0.9)
    events = _collect(
        stream_fn=_stream(["好", "答案"]), question="q", context="上下文",
        chat_history=None, checker=checker, regen_fn=lambda: "不应调用",
    )
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["faithful"] is True
    assert final["regenerated"] is False
    assert not [e for e in events if e["type"] == "correction"]


def test_unfaithful_triggers_correction():
    checker = MagicMock()
    checker.check.side_effect = [
        FaithfulnessResult(faithful=False, score=0.3),
        FaithfulnessResult(faithful=True, score=0.9),
    ]
    events = _collect(
        stream_fn=_stream(["幻觉", "答案"]), question="q", context="上下文",
        chat_history=None, checker=checker, regen_fn=lambda: "严格答案", max_regen=1,
    )
    corrections = [e["data"] for e in events if e["type"] == "correction"]
    assert corrections == ["严格答案"]
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["answer"] == "严格答案"
    assert final["regenerated"] is True
    assert final["faithful"] is True


def test_regen_bounded_by_max():
    checker = MagicMock()
    checker.check.return_value = FaithfulnessResult(faithful=False, score=0.2)
    regen_calls = {"n": 0}
    def regen():
        regen_calls["n"] += 1
        return f"重生成{regen_calls['n']}"
    events = _collect(
        stream_fn=_stream(["答案"]), question="q", context="上下文",
        chat_history=None, checker=checker, regen_fn=regen, max_regen=1,
    )
    assert regen_calls["n"] == 1  # 被 max_regen 兜住
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["regenerated"] is True
    assert final["faithful"] is False  # 达上限仍不忠实，如实返回


def test_token_before_correction_ordering():
    """首屏体验：所有 token 事件必须先于 correction/final。"""
    checker = MagicMock()
    checker.check.side_effect = [
        FaithfulnessResult(faithful=False, score=0.3),
        FaithfulnessResult(faithful=True, score=0.9),
    ]
    events = _collect(
        stream_fn=_stream(["a", "b", "c"]), question="q", context="上下文",
        chat_history=None, checker=checker, regen_fn=lambda: "fix", max_regen=1,
    )
    types = [e["type"] for e in events]
    first_correction = types.index("correction")
    last_token = max(i for i, t in enumerate(types) if t == "token")
    assert last_token < first_correction  # token 全部先于 correction
    assert types[-1] == "final"


def test_deadline_exhausted_skips_regen():
    """时延预算耗尽：流式路径跳过严格重生成（修复此前流式忽略预算的隐性分歧）。"""
    from app.retrieval.deadline import Deadline

    checker = MagicMock()
    checker.check.return_value = FaithfulnessResult(faithful=False, score=0.2)
    t = {"v": 0.0}
    deadline = Deadline(1000, clock=lambda: t["v"])
    t["v"] = 9999.0  # 预算已耗尽

    regen_calls = {"n": 0}
    def regen():
        regen_calls["n"] += 1
        return "严格答案"

    events = _collect(
        stream_fn=_stream(["答案"]), question="q", context="上下文",
        chat_history=None, checker=checker, regen_fn=regen, max_regen=1,
        deadline=deadline,
    )
    assert regen_calls["n"] == 0  # 预算耗尽，不重生成
    assert "F3_regen" in deadline.skipped
    final = [e for e in events if e["type"] == "final"][0]["data"]
    assert final["regenerated"] is False
    assert final["faithful"] is False  # 如实返回未校验通过
    assert not [e for e in events if e["type"] == "correction"]
