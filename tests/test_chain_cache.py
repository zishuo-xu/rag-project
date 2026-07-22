"""缓存写入防护测试"""
from unittest.mock import MagicMock
import numpy as np
from langchain_core.documents import Document

from app.generation.chain import RAGChain


def _bare_chain():
    chain = RAGChain.__new__(RAGChain)  # 绕过重量级 __init__
    chain.semantic_cache = MagicMock()
    chain.indexer = MagicMock()
    chain.indexer.embeddings.embed_query.return_value = [0.1, 0.2]
    return chain


def test_write_cache_skipped_with_chat_history():
    chain = _bare_chain()
    chain._write_cache("问题", [{"role": "user", "content": "前文"}], None, "答案", [])
    chain.semantic_cache.put.assert_not_called()


def test_write_cache_without_history():
    chain = _bare_chain()
    docs = [Document(page_content="内容", metadata={"source": "a.md"})]
    chain._write_cache("问题", None, np.array([0.1]), "答案", docs)
    chain.semantic_cache.put.assert_called_once()
