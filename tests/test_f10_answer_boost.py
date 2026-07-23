"""F10 答案质量增强测试（离线，零 LLM）

覆盖：
1. extract_short_answer：数字型抽数字 / 首句抽取 / 剔除来源标注与填充词 / 空答案
2. self_consistency_vote：多数投票 / 平票稳定 / 空输入
3. AnswerBooster.boost：抽取开关 / 自一致性默认关 / 类型过滤 / 采样投票覆盖 / 异常降级
"""
from unittest.mock import MagicMock

from app.generation.answer_boost import (
    extract_short_answer, self_consistency_vote, AnswerBooster, BoostResult,
)


def _settings(extract=True, sc=False, samples=3, types="numeric,factual"):
    s = MagicMock()
    s.use_answer_extraction = extract
    s.use_self_consistency = sc
    s.self_consistency_samples = samples
    s.self_consistency_types = types
    return s


# ============ extract_short_answer ============

def test_extract_numeric_year():
    ans = "范廷颂于1963年出生。他后来成为主教。[来源: a.md]"
    assert extract_short_answer("范廷颂是什么时候出生的？", ans) == "1963年"


def test_extract_numeric_count():
    ans = "缓存穿透主要有3种解决方案。分别是布隆过滤器等。"
    out = extract_short_answer("缓存穿透有多少种解决方案？", ans)
    assert "3" in out


def test_extract_first_sentence_strips_filler():
    ans = "答案是跳表。Redis的ZSet使用跳表实现有序集合。"
    out = extract_short_answer("Redis的ZSet底层使用什么数据结构", ans)
    assert out.startswith("跳表")


def test_extract_strips_source_tag():
    ans = "缓存穿透指查询不存在的数据。[来源: redis.md]"
    out = extract_short_answer("什么是缓存穿透？", ans)
    assert "来源" not in out
    assert "缓存穿透" in out


def test_extract_strips_list_marker():
    ans = "1、跳表是核心结构。其它还有压缩列表。"
    out = extract_short_answer("ZSet用什么结构", ans)
    assert out.startswith("跳表")


def test_extract_empty_answer():
    assert extract_short_answer("问题", "") == ""
    assert extract_short_answer("问题", "   ") == ""


# ============ self_consistency_vote ============

def test_vote_majority():
    assert self_consistency_vote(["1963年", "1963年", "1964年"]) == "1963年"


def test_vote_tie_stable_first():
    # 平票时取原始顺序最先出现者
    assert self_consistency_vote(["A", "B", "A", "B"]) == "A"


def test_vote_ignores_empty():
    assert self_consistency_vote(["", "X", "", "X"]) == "X"


def test_vote_all_empty():
    assert self_consistency_vote(["", ""]) == ""


# ============ AnswerBooster.boost ============

def test_boost_extraction_on():
    b = AnswerBooster(settings=_settings(extract=True))
    r = b.boost("范廷颂是什么时候出生的？", "他于1963年出生。后来成为主教。")
    assert r.short_answer == "1963年"
    assert r.self_consistency_used is False


def test_boost_extraction_off():
    b = AnswerBooster(settings=_settings(extract=False))
    r = b.boost("问题", "答案内容。")
    assert r.short_answer == ""


def test_boost_self_consistency_disabled_by_default():
    b = AnswerBooster(settings=_settings(sc=False))
    called = {"n": 0}
    def sample():
        called["n"] += 1
        return "1963年。"
    r = b.boost("什么时候出生？", "1963年出生。", query_type="numeric", sample_fn=sample)
    assert r.self_consistency_used is False
    assert called["n"] == 0  # 未采样


def test_boost_self_consistency_type_filter():
    # conceptual 不在 numeric,factual 之列 → 跳过自一致性
    b = AnswerBooster(settings=_settings(sc=True))
    called = {"n": 0}
    def sample():
        called["n"] += 1
        return "x"
    r = b.boost("什么是X？", "答案。", query_type="conceptual", sample_fn=sample)
    assert r.self_consistency_used is False
    assert called["n"] == 0


def test_boost_self_consistency_votes_overrides():
    b = AnswerBooster(settings=_settings(sc=True, samples=3))
    samples = iter(["1963年。详情", "1963年。其它", "1964年。"])
    r = b.boost(
        "什么时候出生？", "他于1964年出生。",  # 初次抽取为 1964
        query_type="numeric", sample_fn=lambda: next(samples),
    )
    assert r.self_consistency_used is True
    assert r.samples == 3
    assert r.voted_answer == "1963年"   # 2/3 多数
    assert r.short_answer == "1963年"   # 被投票结果覆盖


def test_boost_self_consistency_no_sample_fn():
    b = AnswerBooster(settings=_settings(sc=True))
    r = b.boost("什么时候？", "1963年。", query_type="numeric", sample_fn=None)
    assert r.self_consistency_used is False


def test_boost_sample_exception_degrades():
    b = AnswerBooster(settings=_settings(sc=True, samples=3))
    def bad_sample():
        raise RuntimeError("boom")
    r = b.boost("什么时候？", "他于1963年出生。", query_type="numeric", sample_fn=bad_sample)
    # 异常后保留抽取结果，不抛出
    assert r.short_answer == "1963年"
