"""F9 多级缓存 - L1 Embedding 缓存 / L2 Rerank 缓存（L3 为既有语义响应缓存）

生产级 RAG 中，相同/相似查询会重复做 embedding 编码（~50-200ms）与
cross-encoder 重排（~100-500ms），抬高 P95。本模块在既有语义响应缓存之上新增两级：

- L1 EmbeddingCache：key=查询文本，value=向量，省掉重复编码。
- L2 RerankCache：key=hash(query + sorted(chunk_ids))，value={chunk_id: score}，
  命中则跳过 cross-encoder，直接按缓存分排序。

均为进程内线程安全 LRU，O(1) 命中，零外部依赖。任何异常由调用方捕获降级。
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)


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


# ---- 全局单例（供 Reranker / 管道复用） ----

_rerank_cache_instance: Optional[RerankCache] = None


def get_rerank_cache() -> RerankCache:
    """获取全局 Rerank 缓存单例。"""
    global _rerank_cache_instance
    if _rerank_cache_instance is None:
        _rerank_cache_instance = RerankCache()
    return _rerank_cache_instance
