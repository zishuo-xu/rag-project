"""知识图谱升级测试（2026-07-26）：类型化三元组 + chunk 溯源 + 分解路径接入

背景：生产图是 jieba 共现（关系仅「共现/上下文关联」），LLM 三元组路径写了未用，
无实体类型/无 chunk 溯源，分解路径完全跳过 graph 通道。
升级（Option A）：JSON 类型化三元组（person/work/place/org/position/event/other）
+ 边带 chunk_id 溯源 + graph 文档带 graph: 前缀 chunk_id（避免 RRF 覆盖真实分块）
+ 分解路径子问题接入 graph 通道。全离线 mock。
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.ingestion.graph_extractor import KnowledgeGraphBuilder, EXTRACTION_PROMPT
from app.retrieval.graph_retriever import GraphRetriever, ENTITY_EXTRACT_PROMPT


@pytest.fixture
def builder(tmp_path, monkeypatch):
    """隔离持久化路径的图谱构建器（注入 mock LLM，不碰生产图）"""
    monkeypatch.setattr(
        "app.ingestion.graph_extractor.get_settings",
        lambda: SimpleNamespace(graph_persist_path=str(tmp_path / "graph.json")),
    )
    return KnowledgeGraphBuilder(llm=MagicMock())


@pytest.fixture
def retriever(builder, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.graph_retriever.get_settings",
        lambda: SimpleNamespace(graph_max_hops=2, graph_max_entities=5),
    )
    return GraphRetriever(graph_builder=builder, llm=MagicMock())


# ============ 类型化三元组解析 ============

def test_parse_triples_json_typed(builder):
    """JSON 输出解析为带类型的三元组 dict。"""
    out = json.dumps({"triples": [
        {"head": "周润发", "head_type": "person", "relation": "出演",
         "tail": "卧虎藏龙", "tail_type": "work"},
    ]}, ensure_ascii=False)
    triples = builder._parse_triples(out)
    assert len(triples) == 1
    t = triples[0]
    assert t["head"] == "周润发" and t["head_type"] == "person"
    assert t["relation"] == "出演"
    assert t["tail"] == "卧虎藏龙" and t["tail_type"] == "work"


def test_parse_triples_json_with_code_fence(builder):
    """LLM 常包 ```json 代码块，需剥离后解析。"""
    out = '```json\n{"triples": [{"head": "李安", "head_type": "person", "relation": "导演", "tail": "卧虎藏龙", "tail_type": "work"}]}\n```'
    triples = builder._parse_triples(out)
    assert len(triples) == 1
    assert triples[0]["head"] == "李安" and triples[0]["tail_type"] == "work"


def test_parse_triples_unknown_type_normalized(builder):
    """未知类型归一为 other；缺省类型字段同。"""
    out = json.dumps({"triples": [
        {"head": "张三", "head_type": "超级英雄", "relation": "位于", "tail": "北京", "tail_type": "place"},
        {"head": "李四", "relation": "任职", "tail": "总经理"},
    ]}, ensure_ascii=False)
    triples = builder._parse_triples(out)
    assert triples[0]["head_type"] == "other"
    assert triples[0]["tail_type"] == "place"
    assert triples[1]["head_type"] == "other" and triples[1]["tail_type"] == "other"


def test_parse_triples_invalid_json_falls_back_to_lines(builder):
    """非 JSON 输出回退旧版逐行解析（类型置 other），不丢既有能力。"""
    out = "周润发, 出演, 卧虎藏龙\n李安, 导演, 卧虎藏龙"
    triples = builder._parse_triples(out)
    assert len(triples) == 2
    assert triples[0] == {
        "head": "周润发", "head_type": "other",
        "relation": "出演", "tail": "卧虎藏龙", "tail_type": "other",
    }


def test_parse_triples_filters_invalid(builder):
    """头/尾过短或字段缺失的三元组被过滤。"""
    out = json.dumps({"triples": [
        {"head": "X", "head_type": "person", "relation": "出演", "tail": "卧虎藏龙", "tail_type": "work"},
        {"head": "周润发", "head_type": "person", "relation": "", "tail": "卧虎藏龙", "tail_type": "work"},
    ]}, ensure_ascii=False)
    assert builder._parse_triples(out) == []


def test_extraction_prompt_is_json_typed():
    """抽取 Prompt：要求 JSON 输出 + 覆盖全部类型集 + 传记体示例。"""
    text = EXTRACTION_PROMPT.format(text="示例文本")
    assert "triples" in text and "head_type" in text
    for t in ("person", "work", "place", "org", "position", "event", "other"):
        assert t in text, f"类型集缺失 {t}"


# ============ 边写入：chunk 溯源 + 类型 ============

def test_add_triples_edges_carry_chunk_and_types(builder):
    """边写入助手（异步构建共用）：边带 chunk_id + head_type/tail_type，节点带类型。"""
    builder._add_triples(
        [{"head": "周润发", "head_type": "person", "relation": "出演",
          "tail": "卧虎藏龙", "tail_type": "work"}],
        source="a.md", chunk_id="a_0",
    )

    edge = builder.graph.get_edge_data("周润发", "卧虎藏龙")
    assert edge is not None
    assert edge["relation"] == "出演"
    assert edge["chunk_id"] == "a_0"
    assert edge["head_type"] == "person" and edge["tail_type"] == "work"
    assert builder.graph.nodes["周润发"]["type"] == "person"
    assert builder.graph.nodes["卧虎藏龙"]["type"] == "work"


def test_build_async_edges_carry_first_chunk_id(builder, monkeypatch):
    """异步构建（生产路径）：合并单元边溯源到首个 chunk_id。"""
    async def fake_extract(text):
        return [{"head": "李安", "head_type": "person", "relation": "导演",
                 "tail": "卧虎藏龙", "tail_type": "work"}]

    monkeypatch.setattr(builder, "_extract_triples_async", fake_extract)
    docs = [
        Document(page_content="块1", metadata={"doc_id": "d1", "source": "a.md", "chunk_id": "d1_0", "position": 0}),
        Document(page_content="块2", metadata={"doc_id": "d1", "source": "a.md", "chunk_id": "d1_1", "position": 1}),
    ]
    stats = asyncio.run(builder.build_from_documents_async(docs, incremental=False))
    assert stats["total_triples"] == 1

    edge = builder.graph.get_edge_data("李安", "卧虎藏龙")
    assert edge is not None
    assert edge["chunk_id"] == "d1_0"  # 合并单元（d1_0+d1_1）溯源首个块
    assert edge["head_type"] == "person"


def test_entity_relations_include_chunk_id(builder):
    """get_entity_relations 返回 chunk_id 与类型（供图检索文档溯源）。"""
    builder.graph.add_node("周润发", type="person")
    builder.graph.add_node("卧虎藏龙", type="work")
    builder.graph.add_edge(
        "周润发", "卧虎藏龙", relation="出演", source="a.md",
        chunk_id="a_0", head_type="person", tail_type="work",
    )
    rels = builder.get_entity_relations("周润发")
    assert len(rels) == 1
    assert rels[0]["chunk_id"] == "a_0"
    assert rels[0]["head_type"] == "person" and rels[0]["tail_type"] == "work"


# ============ 查询侧 ============

def test_entity_extract_prompt_generic():
    """实体抽取 Prompt 去技术域措辞，改通用（适配传记/影视语料）。"""
    text = ENTITY_EXTRACT_PROMPT.format(question="周润发演了哪部电影？")
    assert "技术名词" not in text
    assert "人物" in text


def test_retrieve_documents_carry_graph_chunk_id(retriever, builder):
    """图检索文档带 graph: 前缀 chunk_id（溯源但不与真实分块在 RRF 中互覆盖）。"""
    builder.graph.add_node("周润发", type="person")
    builder.graph.add_node("卧虎藏龙", type="work")
    builder.graph.add_edge(
        "周润发", "卧虎藏龙", relation="出演", source="a.md",
        chunk_id="a_0", head_type="person", tail_type="work",
    )
    docs = retriever.retrieve("周润发演了哪部电影？")
    assert docs, "图检索应返回文档"
    rel_doc = next(d for d in docs if d.metadata.get("type") == "graph_relations")
    assert rel_doc.metadata["chunk_id"].startswith("graph:")
    assert "a_0" in rel_doc.metadata["chunk_id"]
    assert rel_doc.metadata["source"] == "knowledge_graph"


# ============ 分解路径接入 graph 通道 ============

def test_decompose_path_uses_graph_channel():
    """multi_hop 分解后，各子问题的轻量检索包含 graph 通道（用子问题做实体匹配）。"""
    from app.retrieval.router import QueryRouter
    from test_pipeline import _make_pipeline

    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_query_router = True
    s.use_decomposition = True
    s.use_crag_gate = False
    s.latency_budget_ms = 0
    pipe.query_router = QueryRouter(settings=s)
    mocks["query_transformer"].decompose.return_value = SimpleNamespace(
        sub_questions=["范廷颂担任什么职务？", "该教区在哪里？"], chain=False
    )

    result = pipe.run("范廷颂担任总主教的那个教区在哪里？")
    assert result.query_type == "multi_hop"

    calls = mocks["graph_retriever"].retrieve.call_args_list
    assert calls, "分解路径应调用 graph 通道"
    asked = {c.args[0] for c in calls}
    # graph 实体匹配应针对子问题而非原问题
    assert asked <= {"范廷颂担任什么职务？", "该教区在哪里？"}
