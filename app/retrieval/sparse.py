"""稀疏检索 - BM25 关键词匹配"""

import logging
import re
from typing import List

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.ingestion.indexer import HierarchicalIndexer

logger = logging.getLogger(__name__)

# 预编译正则：匹配英文单词和数字
_EN_WORD_RE = re.compile(r'[a-zA-Z0-9]+')


class SparseRetriever:
    """
    稀疏检索器 - 基于 BM25 的关键词匹配。

    优势：精确匹配专有名词、术语、编号等，
    弥补向量检索在精确匹配上的不足。
    """

    def __init__(self, indexer: HierarchicalIndexer):
        self.indexer = indexer
        self.bm25: BM25Okapi | None = None
        self.corpus: List[Document] = []

    def build_index(self):
        """
        构建 BM25 索引。

        从 ChromaDB 中读取所有已索引的分块，
        进行分词后构建 BM25 倒排索引。
        """
        self.corpus = self.indexer.get_all_chunks()
        if not self.corpus:
            logger.warning("BM25 索引构建失败：无文档数据")
            return

        # 简单分词（中文按字符，英文按空格）
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成: {len(self.corpus)} 个文档")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词策略（jieba 词级分词）：
        - 使用 jieba 进行中文词语级切分（而非单字）
        - 英文/数字保持完整单词
        - 过滤停用词和单字噪声
        """
        # jieba 精确模式分词
        tokens = jieba.lcut(text.lower())
        # 过滤：只保留有意义的 token（中文>=2字 或 英文/数字）
        result = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            # 纯英文/数字：保留
            if _EN_WORD_RE.fullmatch(t):
                result.append(t)
            # 中文：至少2个字才有语义（过滤 "的"、"了" 等单字）
            elif len(t) >= 2 and any('\u4e00' <= c <= '\u9fff' for c in t):
                result.append(t)
        return result

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        BM25 关键词检索。

        Args:
            query: 查询文本
            top_k: 返回 top-K 结果

        Returns:
            按 BM25 分数排序的文档列表
        """
        if self.bm25 is None:
            self.build_index()

        if self.bm25 is None or not self.corpus:
            logger.warning("BM25 索引为空，返回空结果")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top-K 索引
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有匹配的结果
                doc = self.corpus[idx]
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)

        logger.debug(f"BM25 检索: query='{query[:50]}...', 返回 {len(results)} 条")
        return results

    def retrieve_with_scores(self, query: str, top_k: int = 10) -> List[dict]:
        """带分数的 BM25 检索"""
        if self.bm25 is None:
            self.build_index()

        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-top_k:][::-1]

        return [
            {"document": self.corpus[idx], "score": float(scores[idx])}
            for idx in top_indices
            if scores[idx] > 0
        ]
