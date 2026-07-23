"""F6b 多跳查询分解（mock LLM，离线）"""
import json
from unittest.mock import MagicMock

from app.retrieval.query_transform import QueryTransformer, Decomposition


def _transformer_returning(content):
    t = QueryTransformer.__new__(QueryTransformer)  # 跳过 __init__（不连真实 LLM）
    t.llm = MagicMock()
    t.llm.invoke.return_value = MagicMock(content=content)
    t._transform_cache = {}
    t._cache_ttl = 3600
    return t


def test_decompose_parallel():
    payload = json.dumps({"sub_questions": ["范廷颂担任总主教的教区是哪个？", "该教区在哪里？"], "chain": False})
    t = _transformer_returning(payload)
    d = t.decompose("范廷颂担任总主教的那个教区在哪里？")
    assert d.sub_questions == ["范廷颂担任总主教的教区是哪个？", "该教区在哪里？"]
    assert d.chain is False


def test_decompose_chain_flag():
    payload = json.dumps({"sub_questions": ["q1", "q2"], "chain": True})
    t = _transformer_returning(payload)
    assert t.decompose("Q").chain is True


def test_decompose_truncates_to_max_subquestions(monkeypatch):
    from config import get_settings
    monkeypatch.setattr(get_settings(), "decomposition_max_subquestions", 2)
    payload = json.dumps({"sub_questions": ["q1", "q2", "q3", "q4"], "chain": False})
    t = _transformer_returning(payload)
    assert t.decompose("Q").sub_questions == ["q1", "q2"]


def test_decompose_single_subquestion_falls_back_to_original():
    payload = json.dumps({"sub_questions": ["只有一个"], "chain": False})
    t = _transformer_returning(payload)
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False


def test_decompose_invalid_json_falls_back():
    t = _transformer_returning("这不是 JSON")
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False


def test_decompose_llm_exception_falls_back():
    t = QueryTransformer.__new__(QueryTransformer)
    t.llm = MagicMock()
    t.llm.invoke.side_effect = RuntimeError("boom")
    t._transform_cache = {}
    t._cache_ttl = 3600
    d = t.decompose("原问题")
    assert d.sub_questions == ["原问题"]
    assert d.chain is False
