# tests/test_contextual_chunking.py
"""F6a 上下文增强分块：上下文生成器（mock LLM，离线）"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.ingestion.contextual import generate_chunk_context, build_chunk_contexts


def _llm_returning(text):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=text)
    return llm


def test_generate_context_normal():
    llm = _llm_returning("本文讲范廷颂生平，本段述其受封主教。")
    ctx = generate_chunk_context("文档全文...", "范廷颂于1963年受封主教。", llm=llm)
    assert ctx == "本文讲范廷颂生平，本段述其受封主教。"


def test_generate_context_truncates_to_max_chars():
    llm = _llm_returning("x" * 200)
    ctx = generate_chunk_context("d", "c", llm=llm, max_chars=10)
    assert len(ctx) == 10


def test_generate_context_takes_first_line():
    llm = _llm_returning("第一行\n第二行")
    assert generate_chunk_context("d", "c", llm=llm) == "第一行"


def test_generate_context_llm_exception_degrades_to_empty():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    assert generate_chunk_context("d", "c", llm=llm) == ""


def test_build_chunk_contexts_groups_by_doc_and_aligns():
    chunks = [
        Document(page_content="a1", metadata={"doc_id": "A"}),
        Document(page_content="a2", metadata={"doc_id": "A"}),
        Document(page_content="b1", metadata={"doc_id": "B"}),
    ]
    llm = _llm_returning("CTX")
    contexts = build_chunk_contexts(chunks, llm=llm)
    assert contexts == ["CTX", "CTX", "CTX"]
    assert len(contexts) == len(chunks)


import tempfile
import uuid

import pytest
from config import get_settings
from app.ingestion.indexer import HierarchicalIndexer


class FakeEmbeddings:
    """确定性假 embedding：向量由文本长度决定（离线、可复现）"""
    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(len(t) % 3), 1.0] for t in texts]
    def embed_query(self, text):
        return [float(len(text) % 7), float(len(text) % 3), 1.0]


@pytest.fixture
def ctx_indexer(monkeypatch):
    settings = get_settings()
    tag = uuid.uuid4().hex[:8]
    monkeypatch.setattr(settings, "chroma_persist_dir", tempfile.mkdtemp())
    monkeypatch.setattr(settings, "chroma_chunk_collection", f"t_chunks_{tag}")
    monkeypatch.setattr(settings, "chroma_contextual_collection", f"t_ctx_{tag}")
    return HierarchicalIndexer(embeddings=FakeEmbeddings(), llm=MagicMock())


def test_index_documents_contextual_stores_original_text(ctx_indexer):
    chunks = [Document(page_content="原文内容", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["这是上下文"])
    data = ctx_indexer.contextual_store._collection.get(include=["documents", "metadatas"])
    # 存的是原文（不含上下文前缀），上下文进 metadata
    assert data["documents"][0] == "原文内容"
    assert data["metadatas"][0]["context"] == "这是上下文"


def test_detail_store_falls_back_to_chunk_store_when_contextual_empty(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", True)
    # contextual 为空 → 回退 chunk_store
    assert ctx_indexer.detail_store is ctx_indexer.chunk_store


def test_detail_store_uses_contextual_when_enabled_and_built(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", True)
    chunks = [Document(page_content="原文", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["ctx"])
    assert ctx_indexer.detail_store is ctx_indexer.contextual_store


def test_detail_store_uses_chunk_store_when_disabled(ctx_indexer, monkeypatch):
    monkeypatch.setattr(get_settings(), "use_contextual_chunks", False)
    chunks = [Document(page_content="原文", metadata={"doc_id": "A", "chunk_id": "A_0"})]
    ctx_indexer.index_documents_contextual(chunks, ["ctx"])
    assert ctx_indexer.detail_store is ctx_indexer.chunk_store
