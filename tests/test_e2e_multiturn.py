"""端到端 harness 多轮扩展测试（离线，stub chain）

覆盖：
1. eval_sample 将样本 history 透传给 chain.invoke(chat_history=...)
2. 无 history 样本传 None（行为与旧版一致）
3. 结果记录 rewritten_query 字符串（F12 归因用）
4. 多轮样本异常时优雅降级（ok=False）
5. --slice multiturn 过滤逻辑与数据集 slice 字段一致
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_e2e_eval import eval_sample


def _stub_doc(content="Redis 缓存穿透可用布隆过滤器解决", source="redis.md"):
    d = MagicMock()
    d.page_content = content
    d.metadata = {"source": source}
    return d


def _stub_chain(rewritten_query=""):
    """构造记录调用参数的 stub chain，返回结构完整的假响应。"""
    chain = MagicMock()
    resp = SimpleNamespace(
        answer="布隆过滤器可以拦截不存在的 key。",
        sources=[_stub_doc()],
        retrieval_result=SimpleNamespace(
            pre_autocut_count=3, query_type="factual", iterations_used=1,
            iterative_stop_reason="sufficient", gate_skipped=False,
            decomposed_subqueries=[], decomposition_chain=False,
        ),
        faithfulness_score=0.95, faithful=True, regenerated=False,
        short_answer="布隆过滤器", citations=[], rewritten_query=rewritten_query,
    )
    chain.invoke.return_value = resp
    return chain


def _sample(with_history=True):
    s = {
        "id": "mt1",
        "question": "它的解决方案有哪些？",
        "ground_truth": "布隆过滤器、空值缓存、参数校验",
        "slice": "multiturn",
        "metadata": {"source": "redis.md"},
    }
    if with_history:
        s["history"] = [
            {"role": "user", "content": "什么是 Redis 的缓存穿透？"},
            {"role": "assistant", "content": "查询不存在的数据。"},
        ]
    return s


def test_history_passed_to_chain():
    chain = _stub_chain()
    sample = _sample(with_history=True)
    eval_sample(chain, sample)
    chain.invoke.assert_called_once_with(
        "它的解决方案有哪些？", chat_history=sample["history"]
    )


def test_no_history_passes_none():
    chain = _stub_chain()
    eval_sample(chain, _sample(with_history=False))
    chain.invoke.assert_called_once_with("它的解决方案有哪些？", chat_history=None)


def test_rewritten_query_recorded():
    chain = _stub_chain(rewritten_query="Redis缓存穿透解决方案有哪些？")
    result = eval_sample(chain, _sample())
    assert result["ok"] is True
    assert result["rewritten"] is True
    assert result["rewritten_query"] == "Redis缓存穿透解决方案有哪些？"


def test_rewritten_query_empty_when_not_rewritten():
    chain = _stub_chain(rewritten_query="")
    result = eval_sample(chain, _sample())
    assert result["rewritten"] is False
    assert result["rewritten_query"] == ""


def test_exception_with_history_degrades_gracefully():
    chain = MagicMock()
    chain.invoke.side_effect = RuntimeError("llm down")
    result = eval_sample(chain, _sample())
    assert result["ok"] is False
    assert "llm down" in result["error"]


def test_multiturn_slice_filter_matches_dataset():
    """harness 的 slice 过滤表达式应能选中多轮集全部样本。"""
    dataset = json.loads(Path("data/eval_multiturn.json").read_text(encoding="utf-8"))
    samples = dataset["samples"]
    filtered = [s for s in samples if s.get("slice") == "multiturn"]
    assert len(filtered) == len(samples) >= 10
