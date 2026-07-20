"""分块模块单元测试"""

import pytest
from langchain_core.documents import Document

from app.ingestion.chunker import recursive_chunk, smart_chunk


class TestRecursiveChunk:
    """递归分块测试"""

    def test_basic_chunking(self):
        """基本分块功能"""
        docs = [
            Document(
                page_content="这是第一段内容。" * 50,
                metadata={"doc_id": "test_doc", "source": "test.txt"},
            )
        ]
        chunks = recursive_chunk(docs, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 120  # 允许少量超出
            assert "chunk_id" in chunk.metadata
            assert "position" in chunk.metadata

    def test_short_document(self):
        """短文档不需要分块"""
        docs = [
            Document(
                page_content="很短的文档。",
                metadata={"doc_id": "short", "source": "short.txt"},
            )
        ]
        chunks = recursive_chunk(docs, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1

    def test_metadata_preserved(self):
        """元数据保留"""
        docs = [
            Document(
                page_content="测试内容。" * 100,
                metadata={"doc_id": "meta_test", "source": "meta.txt", "custom": "value"},
            )
        ]
        chunks = recursive_chunk(docs, chunk_size=50, chunk_overlap=10)

        for chunk in chunks:
            assert chunk.metadata["doc_id"] == "meta_test"
            assert chunk.metadata["source"] == "meta.txt"

    def test_overlap(self):
        """重叠区域验证"""
        text = "ABCDEFGHIJ" * 20  # 200 chars
        docs = [Document(page_content=text, metadata={"doc_id": "overlap"})]
        chunks = recursive_chunk(docs, chunk_size=50, chunk_overlap=10)

        # 验证有重叠
        if len(chunks) >= 2:
            end_of_first = chunks[0].page_content[-10:]
            assert end_of_first in chunks[1].page_content or len(chunks) == 1


class TestSmartChunk:
    """智能分块入口测试"""

    def test_default_uses_recursive(self):
        """默认使用递归分块"""
        docs = [
            Document(page_content="内容。" * 100, metadata={"doc_id": "smart"})
        ]
        chunks = smart_chunk(docs, use_semantic=False)
        assert len(chunks) >= 1

    def test_empty_input(self):
        """空输入"""
        chunks = recursive_chunk([], chunk_size=100, chunk_overlap=20)
        assert chunks == []
