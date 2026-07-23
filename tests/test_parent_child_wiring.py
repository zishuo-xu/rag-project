"""F6a Parent-Child 接线：上传索引 + 从已有分块重建（mock，离线）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.generation.chain import RAGChain


def _chain_with_mocks():
    chain = RAGChain.__new__(RAGChain)  # 跳过 __init__，手动装配 mock
    chain.indexer = MagicMock()
    chain.parent_child_retriever = MagicMock()
    return chain


def test_rebuild_parent_child_groups_chunks_by_doc():
    chain = _chain_with_mocks()
    chain.indexer.get_all_chunks.return_value = [
        Document(page_content="a1", metadata={"doc_id": "A", "source": "a.txt"}),
        Document(page_content="a2", metadata={"doc_id": "A", "source": "a.txt"}),
        Document(page_content="b1", metadata={"doc_id": "B", "source": "b.txt"}),
    ]
    n = chain.rebuild_parent_child_index()
    assert n == 2  # 两个文档
    args = chain.parent_child_retriever.index_documents.call_args[0][0]
    doc_ids = sorted(d.metadata["doc_id"] for d in args)
    assert doc_ids == ["A", "B"]
    # A 的内容被拼接
    a_doc = next(d for d in args if d.metadata["doc_id"] == "A")
    assert "a1" in a_doc.page_content and "a2" in a_doc.page_content


def test_rebuild_parent_child_no_retriever_returns_zero():
    chain = _chain_with_mocks()
    chain.parent_child_retriever = None
    assert chain.rebuild_parent_child_index() == 0


def test_rebuild_parent_child_exception_returns_zero():
    chain = _chain_with_mocks()
    chain.indexer.get_all_chunks.side_effect = RuntimeError("boom")
    assert chain.rebuild_parent_child_index() == 0
