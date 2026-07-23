"""DenseRetriever 预计算 embedding 复用测试"""
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.retrieval.dense import DenseRetriever


def _make_retriever():
    indexer = MagicMock()
    indexer.search_chunks.return_value = [Document(page_content="by_query")]
    indexer.detail_store.similarity_search_by_vector.return_value = [
        Document(page_content="by_vector")
    ]
    return DenseRetriever(indexer), indexer


def test_retrieve_with_precomputed_embedding():
    """传入预计算 embedding 时走向量检索，不再按文本编码"""
    retriever, indexer = _make_retriever()
    results = retriever.retrieve("问题", top_k=5, embedding=[0.1, 0.2, 0.3])
    indexer.detail_store.similarity_search_by_vector.assert_called_once_with(
        [0.1, 0.2, 0.3], k=5
    )
    indexer.search_chunks.assert_not_called()
    assert results[0].page_content == "by_vector"


def test_retrieve_without_embedding_fallback():
    """不传 embedding 时保持原有文本检索行为"""
    retriever, indexer = _make_retriever()
    results = retriever.retrieve("问题", top_k=5)
    indexer.search_chunks.assert_called_once_with("问题", top_k=5)
    assert results[0].page_content == "by_query"
