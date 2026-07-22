"""稀疏检索 - BM25 关键词匹配"""

import logging
import pickle
import re
from pathlib import Path
from typing import List

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.ingestion.indexer import HierarchicalIndexer
from config import get_settings

logger = logging.getLogger(__name__)

# 预编译正则：匹配英文单词和数字
_EN_WORD_RE = re.compile(r'[a-zA-Z0-9]+')
# 数字+量词/单位组合（如 "1963年"、"100ms"、"5000万"）
_NUM_UNIT_RE = re.compile(r'\d+[\u4e00-\u9fff]?[a-zA-Z\u4e00-\u9fff]*')


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
        # #4: BM25 索引持久化路径
        settings = get_settings()
        self._persist_path = Path(settings.chroma_persist_dir).parent / "bm25_index.pkl"

    def build_index(self):
        """
        构建 BM25 索引。

        从 ChromaDB 中读取所有已索引的分块，
        进行分词后构建 BM25 倒排索引。
        构建完成后自动持久化到磁盘。
        """
        self.corpus = self.indexer.get_all_chunks()
        if not self.corpus:
            logger.warning("BM25 索引构建失败：无文档数据")
            return

        # 简单分词（中文按字符，英文按空格）
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成: {len(self.corpus)} 个文档")
        # #4: 持久化到磁盘
        self._save_index()

    def add_documents(self, new_docs: List[Document]):
        """
        #11: 增量更新 BM25 索引（追加新文档，避免全量重建）。

        Args:
            new_docs: 新索引的文档分块
        """
        if not new_docs:
            return
        self.corpus.extend(new_docs)
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 增量更新: +{len(new_docs)} -> 共 {len(self.corpus)} 个文档")
        self._save_index()

    def _save_index(self):
        """#4: 持久化 BM25 索引到磁盘"""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "corpus": self.corpus,
                }, f)
            logger.debug(f"BM25 索引已持久化: {self._persist_path}")
        except Exception as e:
            logger.warning(f"BM25 索引持久化失败: {e}")

    def _load_index(self) -> bool:
        """#4: 从磁盘加载 BM25 索引"""
        if not self._persist_path.exists():
            return False
        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]
            logger.info(f"BM25 索引从磁盘加载: {len(self.corpus)} 个文档")
            return True
        except Exception as e:
            logger.warning(f"BM25 索引加载失败: {e}")
            return False

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词策略（jieba 词级分词 + 数字量词保留）：
        - 使用 jieba 进行中文词语级切分（而非单字）
        - 英文/数字保持完整单词
        - 数字+量词组合保留（如 "1963年"、"100ms"）
        - 过滤停用词和单字噪声
        """
        # 先提取数字+单位组合作为额外 token
        num_units = _NUM_UNIT_RE.findall(text.lower())

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

        # 补充数字+量词组合 token（解决 "1963年" 被拆散的问题）
        for nu in num_units:
            if nu not in result and len(nu) >= 2:
                result.append(nu)

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
            # #4: 先尝试从磁盘加载，失败再全量重建
            if not self._load_index():
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
            if not self._load_index():
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
