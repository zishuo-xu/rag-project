"""F9 多级缓存测试（离线，无模型依赖）

覆盖：
1. LRUCache：O(1) get/put、淘汰最久未用、命中统计、线程安全基础
2. EmbeddingCache：命中复用、未命中编码、批量保序
3. RerankCache：key 稳定性（顺序无关）、命中/未命中
4. Reranker 集成：缓存命中跳过 cross-encoder、未命中计算并写入
"""
from unittest.mock import MagicMock

import numpy as np
from langchain_core.documents import Document

from app.retrieval.caching import LRUCache, EmbeddingCache, RerankCache
from app.retrieval.reranker import Reranker


# ============ LRUCache ============

def test_lru_basic_get_put():
    c = LRUCache(max_size=2)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None


def test_lru_evicts_least_recently_used():
    c = LRUCache(max_size=2)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")          # a 变最近使用
    c.put("c", 3)       # 应淘汰 b
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3
    assert len(c) == 2


def test_lru_overwrite_existing_key():
    c = LRUCache(max_size=2)
    c.put("a", 1)
    c.put("a", 99)
    assert c.get("a") == 99
    assert len(c) == 1


def test_lru_hit_rate_tracking():
    c = LRUCache(max_size=4)
    c.put("a", 1)
    c.get("a")   # hit
    c.get("a")   # hit
    c.get("x")   # miss
    assert c.hits == 2
    assert c.misses == 1
    assert abs(c.hit_rate - 2 / 3) < 1e-9


def test_lru_clear_resets():
    c = LRUCache(max_size=2)
    c.put("a", 1)
    c.get("a")
    c.clear()
    assert len(c) == 0
    assert c.hits == 0 and c.misses == 0


def test_lru_min_size_one():
    c = LRUCache(max_size=0)  # 被钳到下界 1
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 1


# ============ EmbeddingCache ============

def _mock_embeddings():
    emb = MagicMock()
    _vec = lambda t: [float(len(t)), 1.0]
    emb.embed_query.side_effect = _vec
    # 批处理路径：未命中部分一次批量编码（与单条同口径）
    emb.embed_documents.side_effect = lambda texts: [_vec(t) for t in texts]
    return emb


def test_embedding_cache_miss_then_hit():
    emb = _mock_embeddings()
    cache = EmbeddingCache(emb, max_size=8)
    v1 = cache.embed_query("hello")
    v2 = cache.embed_query("hello")
    assert v1 == v2
    assert emb.embed_query.call_count == 1  # 第二次命中，未再编码


def test_embedding_cache_different_text_encodes():
    emb = _mock_embeddings()
    cache = EmbeddingCache(emb, max_size=8)
    cache.embed_query("a")
    cache.embed_query("bb")
    assert emb.embed_query.call_count == 2


def test_embedding_cache_batch_preserves_order():
    emb = _mock_embeddings()
    cache = EmbeddingCache(emb, max_size=8)
    out = cache.embed_documents(["x", "yy", "x"])
    assert out[0] == [1.0, 1.0]
    assert out[1] == [2.0, 1.0]
    assert out[2] == [1.0, 1.0]
    # 批处理语义：未命中走一次 embed_documents（非逐条 embed_query 循环）
    assert emb.embed_documents.call_count == 1
    assert emb.embed_query.call_count == 0
    # 第二次全命中：不再触发底层编码
    out2 = cache.embed_documents(["x", "yy"])
    assert out2 == [[1.0, 1.0], [2.0, 1.0]]
    assert emb.embed_documents.call_count == 1


# ============ SemanticCache（L3，原零测试 + 无锁竞态修复） ============

def _vec(x):
    import numpy as np
    return np.array([float(x), 0.0])


def _semantic_cache(**kw):
    from app.retrieval.caching import SemanticCache
    defaults = dict(threshold=0.9, max_size=4, ttl=3600)
    defaults.update(kw)
    return SemanticCache(**defaults)


def test_semantic_cache_put_get_hit():
    sc = _semantic_cache()
    sc.put("什么是RAG", _vec(1.0), "答案A", [])
    hit = sc.get(_vec(1.0))  # 完全相同 → 相似度 1.0
    assert hit is not None
    assert hit.answer == "答案A"


def test_semantic_cache_below_threshold_miss():
    sc = _semantic_cache(threshold=0.99)
    sc.put("问题一", _vec(1.0), "答案A", [])
    assert sc.get(_vec(0.0)) is None  # 正交向量，相似度 0


def test_semantic_cache_ttl_expiry():
    import time
    sc = _semantic_cache(ttl=1)
    sc.put("问题", _vec(1.0), "答案", [])
    entry = sc._cache[0]
    entry.timestamp = time.time() - 2  # 伪造过期
    assert sc.get(_vec(1.0)) is None


def test_semantic_cache_eviction_keeps_newest():
    import time
    sc = _semantic_cache(max_size=3)
    for i in range(5):
        sc.put(f"q{i}", _vec(float(i)), f"a{i}", [])
        time.sleep(0.002)  # 保证 timestamp 有序
    assert sc.size == 3
    # 最旧的 q0/q1 被淘汰，q4 还在
    assert sc.get(_vec(4.0)) is not None


def test_semantic_cache_zero_vector_guard():
    import numpy as np
    sc = _semantic_cache()
    sc.put("问题", _vec(1.0), "答案", [])
    assert sc.get(np.array([0.0, 0.0])) is None  # 零向量不崩、不误命中


def test_semantic_cache_concurrent_no_race():
    """并发读写不抛 RuntimeError（修复前淘汰期 sort+重绑与遍历竞态）。"""
    import threading
    sc = _semantic_cache(max_size=8)
    errors = []
    def worker(wid):
        try:
            for i in range(50):
                sc.put(f"w{wid}-q{i}", _vec(float((wid * 50 + i) % 16 + 1)), "a", [])
                sc.get(_vec(float(i % 16 + 1)))
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert sc.size <= 8


# ============ RerankCache ============

def test_rerank_key_order_insensitive():
    k1 = RerankCache.make_key("q", ["c1", "c2", "c3"])
    k2 = RerankCache.make_key("q", ["c3", "c1", "c2"])
    assert k1 == k2


def test_rerank_key_query_sensitive():
    assert RerankCache.make_key("q1", ["c1"]) != RerankCache.make_key("q2", ["c1"])


def test_rerank_cache_miss_then_hit():
    rc = RerankCache(max_size=8)
    assert rc.get_scores("q", ["c1", "c2"]) is None
    rc.put_scores("q", ["c1", "c2"], {"c1": 0.9, "c2": 0.1})
    got = rc.get_scores("q", ["c2", "c1"])  # 顺序无关
    assert got == {"c1": 0.9, "c2": 0.1}


def test_rerank_cache_docset_change_invalidates():
    rc = RerankCache(max_size=8)
    rc.put_scores("q", ["c1", "c2"], {"c1": 0.9, "c2": 0.1})
    # 文档集合变化 → key 变化 → 未命中（不会返回过期排序）
    assert rc.get_scores("q", ["c1", "c3"]) is None


# ============ Reranker 集成 ============

def _bare_reranker(use_cache=True):
    """绕过 CrossEncoder 加载，构造带 mock model 的 Reranker。"""
    r = Reranker.__new__(Reranker)
    r.model = MagicMock()
    r.model_name = "mock"
    r.cache = RerankCache(max_size=16) if use_cache else None
    return r


def _docs():
    return [
        Document(page_content="甲内容", metadata={"chunk_id": "c1"}),
        Document(page_content="乙内容", metadata={"chunk_id": "c2"}),
    ]


def test_reranker_cache_miss_computes_and_stores(monkeypatch):
    monkeypatch.setattr("app.retrieval.reranker.get_settings",
                        lambda: MagicMock(retrieval_top_k=5))
    r = _bare_reranker(use_cache=True)
    r.model.predict.return_value = np.array([0.2, 0.8])
    out = r.rerank("q", _docs(), top_k=2)
    assert r.model.predict.call_count == 1
    assert out[0].metadata["chunk_id"] == "c2"  # 0.8 排前
    # 已写入缓存
    assert r.cache.get_scores("q", ["c1", "c2"]) is not None


def test_reranker_cache_hit_skips_model(monkeypatch):
    monkeypatch.setattr("app.retrieval.reranker.get_settings",
                        lambda: MagicMock(retrieval_top_k=5))
    r = _bare_reranker(use_cache=True)
    r.model.predict.return_value = np.array([0.2, 0.8])
    r.rerank("q", _docs(), top_k=2)          # 第一次：计算
    r.model.predict.reset_mock()
    out = r.rerank("q", _docs(), top_k=2)    # 第二次：命中
    r.model.predict.assert_not_called()      # 跳过 cross-encoder
    assert out[0].metadata["chunk_id"] == "c2"
    assert out[0].metadata["rerank_score"] == 0.8


def test_reranker_no_cache_always_computes(monkeypatch):
    monkeypatch.setattr("app.retrieval.reranker.get_settings",
                        lambda: MagicMock(retrieval_top_k=5))
    r = _bare_reranker(use_cache=False)
    r.model.predict.return_value = np.array([0.2, 0.8])
    r.rerank("q", _docs(), top_k=2)
    r.rerank("q", _docs(), top_k=2)
    assert r.model.predict.call_count == 2


def test_reranker_empty_docs():
    r = _bare_reranker()
    assert r.rerank("q", [], top_k=3) == []


def test_reranker_doc_key_fallback_to_hash():
    r = _bare_reranker()
    d = Document(page_content="无 chunk_id", metadata={})
    key = r._doc_key(d)
    assert key.startswith("h")
