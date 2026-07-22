"""语义缓存 - 相似问题直接命中缓存，跳过全链路调用"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


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

    def _evict_expired(self):
        """清理过期条目"""
        now = time.time()
        before = len(self._cache)
        self._cache = [e for e in self._cache if now - e.timestamp <= self.ttl]
        evicted = before - len(self._cache)
        if evicted > 0:
            logger.debug(f"缓存过期清理: 移除 {evicted} 条")

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
