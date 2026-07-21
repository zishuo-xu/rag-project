"""检索模块单元测试"""

import pytest
from langchain_core.documents import Document

from app.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from app.retrieval.sparse import SparseRetriever


class TestRRFFusion:
    """RRF 融合测试"""

    def test_basic_fusion(self):
        """基本融合功能"""
        list1 = [
            Document(page_content="doc A", metadata={"chunk_id": "a"}),
            Document(page_content="doc B", metadata={"chunk_id": "b"}),
            Document(page_content="doc C", metadata={"chunk_id": "c"}),
        ]
        list2 = [
            Document(page_content="doc B", metadata={"chunk_id": "b"}),
            Document(page_content="doc D", metadata={"chunk_id": "d"}),
            Document(page_content="doc A", metadata={"chunk_id": "a"}),
        ]

        results = reciprocal_rank_fusion([list1, list2], k=60)

        # doc A 和 doc B 在两路都出现，应该排名靠前
        assert len(results) == 4  # A, B, C, D
        top_ids = [r.metadata["chunk_id"] for r in results[:2]]
        assert "a" in top_ids or "b" in top_ids

    def test_single_list(self):
        """单路结果"""
        docs = [
            Document(page_content="doc 1", metadata={"chunk_id": "1"}),
            Document(page_content="doc 2", metadata={"chunk_id": "2"}),
        ]
        results = reciprocal_rank_fusion([docs], k=60)
        assert len(results) == 2
        assert results[0].metadata["chunk_id"] == "1"

    def test_empty_lists(self):
        """空列表"""
        results = reciprocal_rank_fusion([[], []], k=60)
        assert results == []

    def test_rrf_score_attached(self):
        """RRF 分数附加到元数据"""
        docs = [Document(page_content="test", metadata={"chunk_id": "x"})]
        results = reciprocal_rank_fusion([docs], k=60)
        assert "rrf_score" in results[0].metadata
        assert results[0].metadata["rrf_score"] == pytest.approx(1.0 / 61)


class TestWeightedFusion:
    """加权融合测试"""

    def test_weighted(self):
        """加权融合"""
        list1 = [Document(page_content="A", metadata={"chunk_id": "a"})]
        list2 = [Document(page_content="B", metadata={"chunk_id": "b"})]

        # 给第一路更高权重
        results = weighted_fusion([list1, list2], weights=[0.8, 0.2])
        assert results[0].metadata["chunk_id"] == "a"


class TestSparseRetrieverTokenize:
    """BM25 分词测试"""

    def test_english_tokenize(self):
        """英文分词"""
        retriever = SparseRetriever.__new__(SparseRetriever)
        tokens = retriever._tokenize("Hello World test")
        assert tokens == ["hello", "world", "test"]

    def test_chinese_tokenize(self):
        """中文分词"""
        retriever = SparseRetriever.__new__(SparseRetriever)
        tokens = retriever._tokenize("你好世界")
        # jieba 将"你好世界"分为词级 token，而非单字
        assert "你好" in tokens
        assert "世界" in tokens

    def test_mixed_tokenize(self):
        """中英混合分词"""
        retriever = SparseRetriever.__new__(SparseRetriever)
        tokens = retriever._tokenize("RAG技术 retrieval")
        assert "rag" in tokens
        assert "retrieval" in tokens
