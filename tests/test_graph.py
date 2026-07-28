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


def test_extract_entities_fast_only_skips_llm(retriever, monkeypatch):
    """fast_only=True：快速匹配未命中即返回空，不回退 LLM；默认路径保留回退。"""
    monkeypatch.setattr(
        retriever, "_llm_extract_entities", lambda q: ["不应被调用的实体"]
    )
    no_match = "xyz 一个图里没有的句子"
    assert retriever.extract_entities(no_match, fast_only=True) == []
    assert retriever.extract_entities(no_match) == ["不应被调用的实体"]


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
    # 零 LLM 均衡化：子问题图检索强制 fast_only（不回退 LLM 实体抽取）
    s.decompose_graph_fast_only = True
    assert all(c.kwargs.get("fast_only") for c in calls)


def test_decompose_graph_fast_only_switch_off():
    """decompose_graph_fast_only=False → 子问题图检索保留 LLM 回退。"""
    from app.retrieval.router import QueryRouter
    from test_pipeline import _make_pipeline

    pipe, mocks = _make_pipeline()
    s = mocks["settings"]
    s.use_query_router = True
    s.use_decomposition = True
    s.use_crag_gate = False
    s.latency_budget_ms = 0
    s.decompose_graph_fast_only = False
    pipe.query_router = QueryRouter(settings=s)
    mocks["query_transformer"].decompose.return_value = SimpleNamespace(
        sub_questions=["范廷颂担任什么职务？", "该教区在哪里？"], chain=False
    )
    pipe.run("范廷颂担任总主教的那个教区在哪里？")
    calls = mocks["graph_retriever"].retrieve.call_args_list
    assert calls and all(not c.kwargs.get("fast_only") for c in calls)


def test_main_path_graph_recall_keeps_llm_fallback():
    """主路径 recall 默认 fast_graph=False（LLM 实体回退保留，仅分解路径零 LLM）。"""
    from test_pipeline import _make_pipeline

    pipe, mocks = _make_pipeline()
    pipe.recall("问题", ["问题"], top_n=5, channels=("graph",))
    calls = mocks["graph_retriever"].retrieve.call_args_list
    assert calls and calls[0].kwargs.get("fast_only") is False


# ============ 路径语义治理：强/弱边区分 + 诚实标注 ============

def test_find_path_prefers_strong_over_weak_bridge(retriever, builder):
    """并存强路径与弱桥接时，优先返回纯实义路径，weak_only=False。"""
    g = builder.graph
    g.add_edge("A", "X", relation="使用")
    g.add_edge("X", "B", relation="实现")
    g.add_edge("A", "C", relation="属于")
    g.add_edge("C", "B", relation="属于")

    res = retriever.find_path("A", "B")
    assert res["found"] is True
    assert res["weak_only"] is False
    rels = {h["relation"] for h in res["path"]}
    assert rels == {"使用", "实现"}  # 选了强路径，没走 属于 桥


def test_find_path_weak_only_when_no_strong_path(retriever, builder):
    """只能靠上位概念边连通时，found=True 但 weak_only=True，每跳 weak。"""
    g = builder.graph
    g.add_edge("A", "C", relation="属于")
    g.add_edge("C", "B", relation="属于")

    res = retriever.find_path("A", "B")
    assert res["found"] is True
    assert res["weak_only"] is True
    assert res["path_length"] == 2
    assert all(h["weak"] for h in res["path"])


def test_find_path_hop_weak_flag_mixed(retriever, builder):
    """混合路径逐跳 weak 标记正确（实义跳 False / 分类跳 True）。"""
    g = builder.graph
    g.add_edge("A", "X", relation="使用")
    g.add_edge("X", "B", relation="属于")

    res = retriever.find_path("A", "B")
    assert res["found"] is True
    assert res["path"][0]["weak"] is False
    assert res["path"][1]["weak"] is True
    assert res["weak_only"] is True  # 多跳且含弱边


def test_find_path_direct_weak_edge_not_flagged_weak_only(retriever, builder):
    """直连分类边是抽取器的事实陈述：单跳 weak=True 但整条 weak_only=False。"""
    g = builder.graph
    g.add_edge("A", "B", relation="属于")

    res = retriever.find_path("A", "B")
    assert res["found"] is True
    assert res["path_length"] == 1
    assert res["path"][0]["weak"] is True
    assert res["weak_only"] is False  # 关键不变量：直连不贬低


def test_find_path_not_connected(retriever, builder):
    """不连通实体对：found=False，path 为空。"""
    g = builder.graph
    g.add_edge("A", "X", relation="使用")
    g.add_edge("B", "Y", relation="使用")

    res = retriever.find_path("A", "B")
    assert res["found"] is False
    assert res["path"] == []
    assert res["weak_only"] is False


def test_find_path_fuzzy_match_ignores_space_case(retriever, builder):
    """归一化匹配回归：节点名带空格/大小写，查询无空格小写仍命中，返回真实节点名。"""
    g = builder.graph
    g.add_edge("MySQL", "B+ 树", relation="使用")

    res = retriever.find_path("mysql", "b+树")
    assert res["found"] is True
    assert res["weak_only"] is False  # 直连强边
    assert res["path"][0]["from"] == "MySQL"
    assert res["path"][0]["to"] == "B+ 树"


# ============ 节点匹配原语 match_nodes（两历史实现合并，统一 norm_key 口径）============

def test_match_nodes_empty_graph(builder):
    """空图返回空列表（快速路径与子图检索共用的前置守卫）。"""
    assert builder.match_nodes("任意查询") == []


def test_match_nodes_default_bidirectional(builder):
    """默认模糊匹配是双向包含：query↔node 任一方向命中即可。"""
    builder.graph.add_edge("卧虎藏龙", "李安", relation="导演")
    # node 包含 query
    assert "卧虎藏龙" in builder.match_nodes("卧虎")
    # query 包含 node
    assert "李安" in builder.match_nodes("李安导演了哪部片")


def test_match_nodes_normalizes_space_and_case(builder):
    """统一归一化（去空白+小写）：'b+树' 命中节点 'B+ 树'，返回原始节点名。"""
    builder.graph.add_edge("MySQL", "B+ 树", relation="使用")
    assert "B+ 树" in builder.match_nodes("b+树")


def test_match_nodes_exact_match_first(builder):
    """非度数排序时，完全匹配优先于包含匹配。"""
    builder.graph.add_edge("Redis", "RedisCluster", relation="包含")
    matched = builder.match_nodes("redis", top_k=5)
    assert matched[0] == "Redis"


def test_match_nodes_rank_by_degree(builder):
    """快速路径配置（rank_by_degree）：度数高者排前。"""
    g = builder.graph
    g.add_edge("中心", "x", relation="r")
    g.add_edge("中心", "y", relation="r")
    g.add_edge("中心", "z", relation="r")   # 中心 degree 3
    g.add_edge("边缘", "x", relation="r")   # 边缘 degree 1
    matched = builder.match_nodes(
        "中心和边缘", bidirectional=False, rank_by_degree=True, min_len=2
    )
    assert matched[0] == "中心"


def test_match_nodes_dedup_substrings(builder):
    """快速路径配置（dedup_substrings）：被更长已选节点包含的子串节点被去除。"""
    g = builder.graph
    g.add_edge("Redis Cluster", "n1", relation="r")
    g.add_edge("Redis Cluster", "n2", relation="r")  # Redis Cluster degree 2
    g.add_edge("Redis", "n1", relation="r")           # Redis degree 1
    matched = builder.match_nodes(
        "Redis Cluster 和 Redis", bidirectional=False,
        rank_by_degree=True, dedup_substrings=True, min_len=2,
    )
    assert matched == ["Redis Cluster"]  # "Redis" 是其子串，被去重


def test_match_nodes_min_len_filters_short_nodes(builder):
    """快速路径配置（min_len=2）：过滤单字节点，即便出现在查询中。"""
    g = builder.graph
    g.add_edge("A", "B", relation="r")
    g.add_edge("数据库", "A", relation="r")
    matched = builder.match_nodes("A 和 数据库", bidirectional=False, min_len=2)
    assert "数据库" in matched
    assert "A" not in matched


def test_fast_entity_match_unified_normalization(retriever, builder):
    """快速实体匹配修复回归：带空格节点 'B+ 树' 对无空格查询 'b+树' 也能命中。

    历史 _fast_entity_match 仅 .lower() 漏去空白，此场景失配回退 LLM；
    合并到 match_nodes（统一 norm_key）后快速路径直接命中。LLM 被 mock，
    回退路径只会得到空列表，故断言命中即证明走了快速路径。
    """
    builder.graph.add_edge("MySQL", "B+ 树", relation="使用")
    entities = retriever.extract_entities("b+树是什么数据结构")
    assert "B+ 树" in entities
