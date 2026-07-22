"""Autocut 自适应截断测试（Kneedle 膝点检测，纯函数零依赖，离线运行）

算法：把已按 rerank_score 降序的候选分数 min-max 归一化，
找曲线到首尾连线垂直距离最大的点（膝点），在膝点处截断噪声尾巴，
并施加 [min_docs, top_k] 上下界；曲线平坦无膝点时回退 top_k。
"""
from langchain_core.documents import Document

from app.retrieval.autocut import autocut_truncate, find_knee


def _docs(scores):
    """按给定分数（降序）构造带 rerank_score 的文档"""
    return [
        Document(page_content=f"doc{i}", metadata={"rerank_score": s, "chunk_id": f"c{i}"})
        for i, s in enumerate(scores)
    ]


def _contents(docs):
    return [d.page_content for d in docs]


# ============ find_knee 单测（核心几何） ============

def test_find_knee_clear_cliff():
    """高相关平台后断崖下跌 → 膝点在平台末尾"""
    # [0.95,0.90,0.85 | 0.30,0.25,0.20] 膝点 index=2
    assert find_knee([0.95, 0.90, 0.85, 0.30, 0.25, 0.20]) == 2


def test_find_knee_linear_returns_none():
    """完美线性（点全在连线上）→ 无膝点"""
    assert find_knee([0.9, 0.8, 0.7, 0.6, 0.5]) is None


def test_find_knee_all_equal_returns_none():
    """全等分（平坦）→ 无膝点"""
    assert find_knee([0.5, 0.5, 0.5, 0.5]) is None


def test_find_knee_convex_single_relevant():
    """仅首篇高相关（凸曲线）→ 膝点靠近开头"""
    # [0.95 | 0.30,0.25,0.20] 膝点 index=1
    assert find_knee([0.95, 0.30, 0.25, 0.20]) == 1


def test_find_knee_negative_logits():
    """CrossEncoder 原始 logit 可为负 → 归一化后仍正确"""
    # 形状等价于 [高,高,低,低]：[2.0,1.8 | -1.0,-1.5]
    assert find_knee([2.0, 1.8, -1.0, -1.5]) == 1


def test_find_knee_two_points_no_interior():
    """只有 2 点无内点 → 无膝点"""
    assert find_knee([0.9, 0.1]) is None


# ============ autocut_truncate 单测（含上下界） ============

def test_autocut_clear_cliff_keeps_plateau():
    """明显断崖 → 保留高相关平台 3 篇"""
    docs = _docs([0.95, 0.90, 0.85, 0.30, 0.25, 0.20])
    out = autocut_truncate(docs, top_k=5, min_docs=2)
    assert _contents(out) == ["doc0", "doc1", "doc2"]


def test_autocut_flat_falls_back_to_top_k():
    """线性无膝点 → 回退 top_k"""
    docs = _docs([0.9, 0.8, 0.7, 0.6, 0.5])
    out = autocut_truncate(docs, top_k=3, min_docs=2)
    assert len(out) == 3
    assert _contents(out) == ["doc0", "doc1", "doc2"]


def test_autocut_all_equal_falls_back_to_top_k():
    """全等分 → 回退 top_k"""
    docs = _docs([0.5, 0.5, 0.5, 0.5, 0.5])
    out = autocut_truncate(docs, top_k=4, min_docs=2)
    assert len(out) == 4


def test_autocut_floor_min_docs():
    """膝点过早（< min_docs）→ 下界兜底保留 min_docs"""
    # 仅首篇高相关，膝点 index=1 → keep=2；min_docs=3 提升到 3
    docs = _docs([0.95, 0.30, 0.25, 0.20, 0.18])
    out = autocut_truncate(docs, top_k=5, min_docs=3)
    assert len(out) == 3


def test_autocut_ceiling_top_k():
    """膝点很晚（> top_k）→ 上界压到 top_k"""
    # 前 4 篇高相关平台，第 5 篇才跌：膝点 index=3 → keep=4，top_k=2 压到 2
    docs = _docs([0.95, 0.93, 0.91, 0.90, 0.20, 0.15])
    out = autocut_truncate(docs, top_k=2, min_docs=1)
    assert len(out) == 2


def test_autocut_never_exceeds_candidates():
    """top_k 大于候选数 → 不超过候选总数"""
    docs = _docs([0.9, 0.85, 0.2, 0.1])
    out = autocut_truncate(docs, top_k=10, min_docs=2)
    assert len(out) <= 4


def test_autocut_fewer_than_min_docs_returns_all():
    """候选数 ≤ min_docs → 全部保留"""
    docs = _docs([0.9, 0.8])
    out = autocut_truncate(docs, top_k=5, min_docs=3)
    assert len(out) == 2


def test_autocut_empty():
    assert autocut_truncate([], top_k=5, min_docs=2) == []


def test_autocut_single_doc():
    docs = _docs([0.9])
    out = autocut_truncate(docs, top_k=5, min_docs=2)
    assert len(out) == 1


def test_autocut_custom_score_key():
    """支持自定义分数字段"""
    docs = [
        Document(page_content=f"d{i}", metadata={"my_score": s})
        for i, s in enumerate([0.95, 0.90, 0.85, 0.20, 0.10])
    ]
    out = autocut_truncate(docs, top_k=5, min_docs=2, score_key="my_score")
    assert len(out) == 3


def test_autocut_missing_score_treated_as_zero():
    """缺失分数字段按 0 处理，不崩溃"""
    docs = [
        Document(page_content="a", metadata={"rerank_score": 0.9}),
        Document(page_content="b", metadata={}),
        Document(page_content="c", metadata={}),
    ]
    out = autocut_truncate(docs, top_k=3, min_docs=2)
    assert len(out) >= 2  # 不崩溃且满足下界


def test_autocut_returns_new_list_not_mutate():
    """返回新列表，不改原输入"""
    docs = _docs([0.95, 0.90, 0.85, 0.30, 0.25])
    original_len = len(docs)
    out = autocut_truncate(docs, top_k=5, min_docs=2)
    assert len(docs) == original_len
    assert out is not docs
