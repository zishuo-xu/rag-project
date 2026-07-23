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
