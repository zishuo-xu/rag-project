"""多级缓存 - L1 Embedding / L2 Rerank / L3 语义响应缓存

生产级 RAG 中，相同/相似查询会重复做 embedding 编码（~50-200ms）与
cross-encoder 重排（~100-500ms），抬高 P95；而整条检索+生成链路（~15s）
在相似问题上完全可以复用答案。本模块三级缓存统一收口：

- L1 EmbeddingCache：key=查询文本，value=向量，省掉重复编码。
- L2 RerankCache：key=hash(query + sorted(chunk_ids))，value={chunk_id: score}，
  命中则跳过 cross-encoder，直接按缓存分排序。
- L3 SemanticCache：问题向量余弦相似 > 阈值直接返回缓存答案，跳过全链路
  （延迟从 ~15s 降到 <100ms）。

L1/L2 为进程内线程安全 LRU，O(1) 命中，零外部依赖；L3 为 TTL + LRU 淘汰的
向量扫描（缓存规模小，O(n) 可接受）。任何异常由调用方捕获降级。
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


# ==================== 通用基座 ====================

class LRUCache:
    """线程安全 LRU 缓存（OrderedDict 实现，O(1) get/put，带命中统计）。"""

    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)  # 淘汰最久未用

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ==================== L1 Embedding 缓存 ====================

class EmbeddingCache:
    """L1 Embedding 缓存：包装底层 embeddings，相同文本直接命中，省掉重复编码。"""

    def __init__(self, embeddings, max_size: Optional[int] = None):
        settings = get_settings()
        self.embeddings = embeddings
        self.cache = LRUCache(max_size or settings.embedding_cache_size)

    def embed_query(self, text: str):
        """单条查询编码：命中直接返回，否则编码后写入。"""
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vec = self.embeddings.embed_query(text)
        self.cache.put(text, vec)
        return vec

    def embed_documents(self, texts: List[str]):
        """批量编码：逐条走缓存（保持输入顺序），重复文本复用。"""
        return [self.embed_query(t) for t in texts]


# ==================== L2 Rerank 缓存 ====================

class RerankCache:
    """L2 Rerank 缓存：key=hash(query + sorted(chunk_ids))。

    命中返回 {chunk_id: score}；文档集合变化即 key 变化，不会返回过期排序。
    """

    def __init__(self, max_size: Optional[int] = None):
        settings = get_settings()
        self.cache = LRUCache(max_size or settings.rerank_cache_size)

    @staticmethod
    def make_key(query: str, chunk_ids: List[str]) -> str:
        ids = sorted(str(c) for c in chunk_ids)
        return f"{query}||{'|'.join(ids)}"

    def get_scores(self, query: str, chunk_ids: List[str]) -> Optional[Dict[str, float]]:
        return self.cache.get(self.make_key(query, chunk_ids))

    def put_scores(self, query: str, chunk_ids: List[str], scores: Dict[str, float]) -> None:
        self.cache.put(self.make_key(query, chunk_ids), scores)


_rerank_cache_instance: Optional[RerankCache] = None


def get_rerank_cache() -> RerankCache:
    """获取全局 Rerank 缓存单例。"""
    global _rerank_cache_instance
    if _rerank_cache_instance is None:
        _rerank_cache_instance = RerankCache()
    return _rerank_cache_instance


# ==================== L3 语义响应缓存 ====================

@dataclass
class CachedAnswer:
    """缓存条目"""
    question: str
    answer: str
    sources: List[dict]  # [{content, source, score}]
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    语义缓存：基于问题向量的余弦相似度判断是否命中。

    当新问题与缓存中某问题的 cosine similarity > threshold 时，
    直接返回缓存的答案，跳过检索+生成全链路（延迟从 ~15s 降到 <100ms）。

    淘汰策略：TTL 过期 + LRU（超容量时淘汰最久未命中的条目）。
    """

    def __init__(
        self,
        threshold: float | None = None,
        max_size: int | None = None,
        ttl: int | None = None,
    ):
        settings = get_settings()
        self.threshold = threshold or settings.cache_threshold
        self.max_size = max_size or settings.cache_max_size
        self.ttl = ttl or settings.cache_ttl
        self._cache: List[CachedAnswer] = []
        logger.info(
            f"SemanticCache 初始化: threshold={self.threshold}, "
            f"max_size={self.max_size}, ttl={self.ttl}s"
        )

    def get(self, query_embedding: np.ndarray) -> Optional[CachedAnswer]:
        """
        查询缓存：找到最相似且超过阈值的问题。

        Args:
            query_embedding: 问题的归一化向量

        Returns:
            命中则返回 CachedAnswer，否则 None
        """
        if not self._cache:
            return None

        now = time.time()
        best_score = -1.0
        best_entry: Optional[CachedAnswer] = None

        # 遍历缓存计算相似度（缓存规模小，O(n) 可接受）
        for entry in self._cache:
            # TTL 检查
            if now - entry.timestamp > self.ttl:
                continue
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            best_entry.hit_count += 1
            best_entry.timestamp = now  # LRU 更新
            logger.info(
                f"缓存命中: '{best_entry.question[:30]}...' "
                f"(similarity={best_score:.4f}, hits={best_entry.hit_count})"
            )
            return best_entry

        return None

    def put(
        self,
        question: str,
        embedding: np.ndarray,
        answer: str,
        sources: List[dict],
    ):
        """
        写入缓存。

        Args:
            question: 原始问题
            embedding: 问题向量
            answer: 生成的回答
            sources: 来源列表
        """
        entry = CachedAnswer(
            question=question,
            answer=answer,
            sources=sources,
            embedding=embedding,
        )
        self._cache.append(entry)

        # 超容量淘汰：按 timestamp 排序，移除最旧的
        if len(self._cache) > self.max_size:
            self._cache.sort(key=lambda x: x.timestamp, reverse=True)
            self._cache = self._cache[:self.max_size]
            logger.debug(f"缓存淘汰: 保留 {self.max_size} 条")

        logger.debug(f"缓存写入: '{question[:30]}...' (size={len(self._cache)})")

    def clear(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("语义缓存已清空")

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# 全局单例
_cache_instance: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """获取全局语义缓存单例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
