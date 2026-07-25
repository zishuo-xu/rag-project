"""查询路由 / 类型自适应测试（纯规则、零 LLM、离线运行）

路由器根据查询类型自适应调整检索深度(top_k)与降噪强度(autocut_min_docs)，
不削减召回路（避免漏召回）。优先级：multi_hop > numeric > comparative > conceptual > factual。
"""
from unittest.mock import MagicMock

from app.retrieval.router import QueryRouter


def _router(top_k=5):
    settings = MagicMock()
    settings.retrieval_top_k = top_k
    return QueryRouter(settings=settings)


# ============ 查询类型判定 ============

def test_numeric_when_question():
    assert _router().route("范廷颂是什么时候被任为主教的？").query_type == "numeric"


def test_numeric_how_many():
    assert _router().route("缓存穿透有多少种解决方案？").query_type == "numeric"


def test_numeric_english():
    assert _router().route("How many types of cache penetration exist?").query_type == "numeric"


def test_comparative_difference():
    assert _router().route("Redis和Memcached的区别是什么？").query_type == "comparative"


def test_comparative_vs():
    assert _router().route("微服务 vs 单体架构").query_type == "comparative"


def test_multi_hop_relation_chain():
    # 含关系链「X的Y的Z」
    assert _router().route("范廷颂所在的教区的名字是什么？").query_type == "multi_hop"


def test_conceptual_what_is():
    assert _router().route("什么是缓存穿透？").query_type == "conceptual"


def test_conceptual_principle():
    assert _router().route("Transformer中Self-Attention的原理").query_type == "conceptual"


def test_factual_default():
    # 不含任何触发信号 → 默认事实型
    assert _router().route("Redis的ZSet底层使用什么数据结构").query_type == "factual"


def test_empty_query_factual():
    assert _router().route("").query_type == "factual"
    assert _router().route("   ").query_type == "factual"


# ============ 优先级 ============

def test_comparative_beats_conceptual():
    # 「区别是什么」同时命中 comparative 与 conceptual，comparative 优先
    assert _router().route("A和B的区别是什么").query_type == "comparative"


def test_numeric_beats_conceptual():
    assert _router().route("GIL是什么，有多少线程受影响").query_type == "numeric"


# ============ multi_hop 判定扩展（2026-07-25 评估闭环暴露：15 条多跳仅判中 1 条） ============

def test_multi_hop_parallel_marker():
    # 并行标记「分别/各自」+ 疑问词 → 多跳
    assert _router().route("大卫·克劳斯和岸本早未分别是哪国人？").query_type == "multi_hop"
    assert _router().route("米象和半领彩鹬各自的食性是什么？").query_type == "multi_hop"


def test_multi_hop_pronoun_chain():
    # 指代链「的那个」+ 疑问词 → 多跳
    assert _router().route("范廷颂担任总主教的那个教区在哪里？").query_type == "multi_hop"


def test_multi_hop_order_comparison():
    # 先后比较「先…还是…先」→ 多跳
    assert _router().route("范廷颂是先成为总主教还是先成为枢机？").query_type == "multi_hop"


def test_multi_hop_relaxed_chain_distance():
    # 关系链距离放宽（>10 字）+ 疑问词 → 多跳
    assert _router().route("岸本早未的出道单曲被用作哪部动画的片头曲？").query_type == "multi_hop"


def test_multi_hop_temporal_sequence():
    # 时间序列双事实「…之后…」+ 疑问词 → 多跳
    assert _router().route("《太平天历》是在太平军攻克哪里之后正式颁行的？").query_type == "multi_hop"


def test_multi_hop_beats_numeric_when_both():
    # 同时命中 numeric（哪一年）与 multi_hop（关系链+兼任第二事实）→ multi_hop 优先
    assert _router().route("范廷颂被擢升为枢机的那一年，他还兼任了哪个教区的宗座署理？").query_type == "multi_hop"


def test_pure_numeric_unchanged_after_priority_change():
    # 纯数字型（无多跳标记）优先级调整后行为不变
    assert _router().route("范廷颂是什么时候被任为主教的？").query_type == "numeric"
    assert _router().route("缓存穿透有多少种解决方案？").query_type == "numeric"


def test_multihop_dataset_recall():
    """多跳评估集路由召回：15 条中 ≥10 条应判为 multi_hop（修复前仅 1 条）。"""
    import json
    from pathlib import Path
    samples = json.loads(Path("data/eval_multihop.json").read_text(encoding="utf-8"))["samples"]
    router = _router()
    hits = [s["id"] for s in samples if router.route(s["question"]).query_type == "multi_hop"]
    assert len(hits) >= 10, f"multi_hop 召回不足: {len(hits)}/15 -> {hits}"


def test_cmrc_no_multihop_overclassification():
    """CMRC 单跳集回归防线：multi_hop 误判不得明显增长（修复前 3/31，容忍 ≤4）。"""
    import json
    from pathlib import Path
    samples = json.loads(Path("data/eval_dataset_cmrc.json").read_text(encoding="utf-8"))["samples"]
    router = _router()
    mh = [s["question"] for s in samples if router.route(s["question"]).query_type == "multi_hop"]
    assert len(mh) <= 4, f"CMRC multi_hop 误判过多: {len(mh)} -> {mh}"


# ============ 策略参数（真实行为效果） ============

def test_numeric_tightens_autocut():
    d = _router().route("哪一年发生的？")
    assert d.autocut_min_docs == 1     # 精确答案，收紧降噪
    assert d.top_k is None             # 不改检索深度


def test_comparative_widens_top_k():
    d = _router(top_k=5).route("Redis和Memcached的区别")
    assert d.top_k == 8                # 对比需更多候选 (5+3)
    assert d.autocut_min_docs == 3


def test_multi_hop_widens_top_k():
    d = _router(top_k=5).route("范廷颂所在的教区的名字是什么？")
    assert d.top_k == 8


def test_conceptual_uses_defaults():
    d = _router().route("什么是缓存穿透？")
    assert d.top_k is None
    assert d.autocut_min_docs is None


def test_factual_uses_defaults():
    d = _router().route("Redis的ZSet底层使用什么数据结构")
    assert d.top_k is None
    assert d.autocut_min_docs is None


def test_decision_has_reason():
    d = _router().route("什么是缓存穿透？")
    assert d.reason  # 可解释性：每个路由决策带理由
