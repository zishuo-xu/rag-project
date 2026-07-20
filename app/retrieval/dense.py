"""稠密向量检索 - OpenAI Embedding + ChromaDB"""

import logging
from typing import List

from langchain_core.documents import Document

from app.ingestion.indexer import HierarchicalIndexer

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    稠密检索器 - 基于向量相似度搜索。

    使用 OpenAI text-embedding-3-small 将查询编码为向量，
    在 ChromaDB 中进行余弦相似度搜索。
    """

    def __init__(self, indexer: HierarchicalIndexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        稠密向量检索。

        Args:
            query: 查询文本
            top_k: 返回 top-K 结果

        Returns:
            按相似度排序的文档列表
        """
        results = self.indexer.search_chunks(query, top_k=top_k)
        logger.debug(f"稠密检索: query='{query[:50]}...', 返回 {len(results)} 条")
        return results

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[dict]:
        """
        带相似度分数的检索。

        Returns:
            [{"document": Document, "score": float}, ...]
        """
        results_with_scores = self.indexer.chunk_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        return [
            {"document": doc, "score": score}
            for doc, score in results_with_scores
        ]
