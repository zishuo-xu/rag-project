"""F7 引用溯源测试（离线，embedding 全 mock）

覆盖：
1. split_claims：切句 / 剔除来源标注 / 过滤过短 / max_claims 上限 / 空答案
2. _l2_normalize / _best_snippet 纯函数
3. CitationBuilder.build：claim-块关联、置信度、空输入降级、异常降级
"""
from unittest.mock import MagicMock

import numpy as np
from langchain_core.documents import Document

from app.generation.citation import (
    split_claims, CitationBuilder, Citation, _l2_normalize, _best_snippet,
)


# ============ split_claims ============

def test_split_claims_basic():
    ans = "范廷颂是主教。他在1963年出生。"
    claims = split_claims(ans, max_claims=6)
    assert len(claims) == 2
    assert claims[0].startswith("范廷颂")


def test_split_claims_strips_source_tag():
    ans = "答案是缓存穿透。[来源: redis.md]"
    claims = split_claims(ans, max_claims=6)
    assert claims == ["答案是缓存穿透"]
    assert "来源" not in claims[0]


def test_split_claims_filters_short():
    ans = "是的。这是一个很长的句子用于测试过滤。"
    claims = split_claims(ans, max_claims=6)
    assert all(len(c) >= 4 for c in claims)
    assert "是的" not in claims


def test_split_claims_max_limit():
    ans = "第一个论断句子。第二个论断句子。第三个论断句子。第四个论断句子。"
    claims = split_claims(ans, max_claims=2)
    assert len(claims) == 2


def test_split_claims_strips_list_markers():
    ans = "1、第一点内容足够长。2、第二点内容也够长。"
    claims = split_claims(ans, max_claims=6)
    assert claims[0].startswith("第一点")


def test_split_claims_empty():
    assert split_claims("", 6) == []
    assert split_claims("   ", 6) == []


# ============ 纯函数 ============

def test_l2_normalize_unit_norm():
    mat = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = _l2_normalize(mat)
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-9
    assert np.allclose(out[1], [0.0, 0.0])  # 零向量保持


def test_best_snippet_picks_overlapping_sentence():
    content = "Redis支持多种数据结构。缓存穿透指查询不存在的数据。ZSet用跳表实现。"
    snip = _best_snippet(content, "缓存穿透是什么")
    assert "缓存穿透" in snip


# ============ CitationBuilder.build ============

def _doc(cid, source, content):
    return Document(page_content=content, metadata={"chunk_id": cid, "source": source})


def _mock_embeddings(vectors):
    """embed_documents 跨调用按顺序消费预设向量（先 claims 后 docs）。"""
    emb = MagicMock()
    it = iter(vectors)
    emb.embed_documents.side_effect = lambda texts: [next(it) for _ in texts]
    return emb


def test_build_maps_claim_to_best_doc():
    docs = [_doc("c1", "a.md", "甲文档内容"), _doc("c2", "b.md", "乙文档内容")]
    # claim 向量与 doc1(乙) 最相似
    vectors = [
        [0.0, 1.0],   # claim
        [1.0, 0.0],   # doc0 甲
        [0.1, 0.99],  # doc1 乙 ← 最接近 claim
    ]
    builder = CitationBuilder(embeddings=_mock_embeddings(vectors), threshold=0.5, max_claims=6)
    cites = builder.build("q", "这是一个测试论断句子。", docs)
    assert len(cites) == 1
    assert cites[0].chunk_id == "c2"
    assert cites[0].doc_index == 2
    assert cites[0].source == "b.md"
    assert cites[0].confidence > 0.9


def test_build_confidence_in_unit_interval():
    docs = [_doc("c1", "a.md", "内容甲")]
    vectors = [[1.0, 0.0], [1.0, 0.0]]
    builder = CitationBuilder(embeddings=_mock_embeddings(vectors), max_claims=6)
    cites = builder.build("q", "一个论断句子。", docs)
    assert 0.0 <= cites[0].confidence <= 1.0


def test_build_no_embeddings_degrades():
    builder = CitationBuilder(embeddings=None)
    assert builder.build("q", "论断句子。", [_doc("c1", "a.md", "x")]) == []


def test_build_no_docs_degrades():
    builder = CitationBuilder(embeddings=MagicMock())
    assert builder.build("q", "论断句子。", []) == []


def test_build_empty_answer_degrades():
    builder = CitationBuilder(embeddings=MagicMock())
    assert builder.build("q", "", [_doc("c1", "a.md", "x")]) == []


def test_build_exception_degrades_to_empty():
    emb = MagicMock()
    emb.embed_documents.side_effect = RuntimeError("boom")
    builder = CitationBuilder(embeddings=emb, max_claims=6)
    assert builder.build("q", "论断句子。", [_doc("c1", "a.md", "x")]) == []


def test_build_multiple_claims():
    docs = [_doc("c1", "a.md", "甲"), _doc("c2", "b.md", "乙")]
    vectors = [
        [1.0, 0.0], [0.0, 1.0],   # 2 claims
        [1.0, 0.0],               # doc0 → claim0 最近
        [0.0, 1.0],               # doc1 → claim1 最近
    ]
    builder = CitationBuilder(embeddings=_mock_embeddings(vectors), max_claims=6)
    cites = builder.build("q", "第一个论断句子。第二个论断句子。", docs)
    assert len(cites) == 2
    assert cites[0].chunk_id == "c1"
    assert cites[1].chunk_id == "c2"
