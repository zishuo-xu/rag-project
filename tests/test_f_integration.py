"""RAG 3.0 集成测试 - 验证 F7/F8/F10/F12 真正接入 RAGChain（离线，组件全 mock）

单元测试只覆盖各模块自身；本文件用 RAGChain.__new__ 绕过重型 __init__，
注入 mock 组件，验证 invoke / invoke_stream 确实调用了各特性并写入响应字段。
"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.generation.chain import RAGChain, RAGResponse
from app.retrieval.pipeline import RetrievalResult
from app.generation.citation import Citation
from app.generation.faithfulness import FaithfulnessResult
from app.generation.answer_boost import BoostResult


def _doc():
    return Document(page_content="缓存穿透指查询不存在的数据。", metadata={"source": "redis.md", "chunk_id": "c1"})


def _result(query_type="factual", gate_skipped=False):
    return RetrievalResult(documents=[_doc()], query_type=query_type, gate_skipped=gate_skipped)


def _invoke_chain():
    chain = RAGChain.__new__(RAGChain)
    chain._settings = MagicMock(
        use_speculative_streaming=True, faithfulness_max_regen=1,
    )
    chain.semantic_cache = None
    chain.indexer = MagicMock()
    chain.faithfulness_checker = None
    chain.citation_builder = MagicMock()
    chain.citation_builder.build.return_value = [
        Citation(claim="缓存穿透指查询不存在的数据", source="redis.md",
                 chunk_id="c1", doc_index=1, confidence=0.85, snippet="缓存穿透指...")
    ]
    chain.answer_booster = MagicMock()
    chain.answer_booster.boost.return_value = BoostResult(short_answer="查询不存在的数据")
    chain.conversation_rewriter = MagicMock()
    chain.conversation_rewriter.rewrite.return_value = "缓存穿透的原理是什么？"
    chain.retrieve = MagicMock(return_value=_result())
    chain.generate = MagicMock(return_value="缓存穿透指查询不存在的数据。[来源: redis.md]")
    return chain


# ============ invoke 集成 ============

def test_invoke_applies_f12_rewrite_to_retrieval():
    chain = _invoke_chain()
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    resp = chain.invoke("它的原理呢？", chat_history=history)
    # F12：检索使用重写后的查询
    assert chain.retrieve.call_args.args[0] == "缓存穿透的原理是什么？"
    assert resp.rewritten_query == "缓存穿透的原理是什么？"


def test_invoke_no_rewrite_without_history():
    chain = _invoke_chain()
    resp = chain.invoke("什么是缓存穿透？")  # 无历史
    assert chain.retrieve.call_args.args[0] == "什么是缓存穿透？"
    assert resp.rewritten_query == ""


def test_invoke_populates_f7_citations():
    chain = _invoke_chain()
    resp = chain.invoke("什么是缓存穿透？")
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == "c1"
    assert resp.citations[0].confidence == 0.85


def test_invoke_populates_f10_short_answer():
    chain = _invoke_chain()
    resp = chain.invoke("什么是缓存穿透？")
    assert resp.short_answer == "查询不存在的数据"
    assert resp.self_consistency_used is False


def test_invoke_returns_ragresponse_type():
    chain = _invoke_chain()
    resp = chain.invoke("什么是缓存穿透？")
    assert isinstance(resp, RAGResponse)
    assert resp.total_time_ms >= 0


def test_invoke_citation_failure_degrades():
    chain = _invoke_chain()
    chain.citation_builder.build.side_effect = RuntimeError("boom")
    resp = chain.invoke("什么是缓存穿透？")  # 不抛出
    assert resp.citations == []


# ============ invoke_stream 集成（F8 投机流式） ============

def _stream_chain(speculative=True):
    chain = RAGChain.__new__(RAGChain)
    chain._settings = MagicMock(
        use_speculative_streaming=speculative, faithfulness_max_regen=1,
    )
    chain.semantic_cache = None
    chain.indexer = MagicMock()
    chain.conversation_rewriter = None
    chain.citation_builder = None
    chain.answer_booster = MagicMock()
    chain.answer_booster.boost.return_value = BoostResult(short_answer="")
    chain.faithfulness_checker = MagicMock()
    chain.faithfulness_checker.check.side_effect = [
        FaithfulnessResult(faithful=False, score=0.3),
        FaithfulnessResult(faithful=True, score=0.9),
    ]
    chain.retrieve = MagicMock(return_value=_result())
    chain.generate_stream = MagicMock(return_value=iter(["幻觉", "答案"]))
    chain.generate = MagicMock(return_value="严格答案")
    return chain


def test_stream_speculative_emits_correction():
    chain = _stream_chain(speculative=True)
    events = list(chain.invoke_stream("q"))
    types = [e["type"] for e in events]
    assert "token" in types
    assert "correction" in types
    # token 先于 correction（首屏快）
    assert types.index("token") < types.index("correction")
    done = [e for e in events if e["type"] == "done"][0]["data"]
    assert done.answer == "严格答案"
    assert done.regenerated is True
    assert done.faithful is True


def test_stream_speculative_streams_tokens_individually():
    chain = _stream_chain(speculative=True)
    events = list(chain.invoke_stream("q"))
    tokens = [e["data"] for e in events if e["type"] == "token"]
    assert tokens == ["幻觉", "答案"]  # 逐 token 流出


def test_stream_speculative_disabled_falls_back_blocking():
    chain = _stream_chain(speculative=False)
    events = list(chain.invoke_stream("q"))
    types = [e["type"] for e in events]
    # 关闭投机流式 → 走阻塞式 _generate_faithful，无逐字 token（整体输出）
    assert "correction" not in types
    done = [e for e in events if e["type"] == "done"][0]["data"]
    assert done.regenerated is True  # 阻塞路径同样触发重生成


def test_stream_gate_skipped_uses_direct_stream():
    chain = _stream_chain(speculative=True)
    chain.retrieve = MagicMock(return_value=_result(gate_skipped=True))
    chain.generate_direct_stream = MagicMock(return_value=iter(["直接", "回答"]))
    events = list(chain.invoke_stream("q"))
    tokens = [e["data"] for e in events if e["type"] == "token"]
    assert tokens == ["直接", "回答"]
