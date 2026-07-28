"""图检索器 - 基于知识图谱的关系检索与子图扩展

核心能力：
    1. 实体识别：从用户查询中识别关键实体
    2. 子图检索：以识别的实体为起点，N 跳扩展获取相关子图
    3. 路径检索：查找两个实体之间的关系路径
    4. 上下文格式化：将子图三元组转为 LLM 可理解的文本

设计决策：
    - 图检索作为向量检索的补充（而非替代），提供关系推理能力
    - 实体抽取：零 LLM 快速匹配（图节点关键词）优先，未命中回退 LLM 抽取；
      分解路径子问题走 fast_only（不回退），避免子问题数量放大 LLM 调用
    - 格式化输出为自然语言三元组列表，方便 LLM 理解
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings, build_chat_llm
from app.ingestion.graph_extractor import get_graph_builder, KnowledgeGraphBuilder
from app.utils import norm_key

logger = logging.getLogger(__name__)

# ── 路径语义治理：上位概念/分类学边词表（读侧实时判定，不写边属性）──
# 这些 relation 表达的是 taxonomy 归属（A 属于/是/示例 B），单条是事实陈述，
# 但串联成多跳路径时会把本无实质关联的实体"桥接"起来（图连通 ≠ 语义强相关）。
# 词表存归一化（去空白+小写）形式；另对 "属于*" 变体（属于级别/属于策略…）做前缀兜底。
_WEAK_RELATIONS = frozenset({
    "属于", "是一种", "是", "归类为", "类型为", "分类为",
    "示例", "例如", "包括于", "包含于", "隶属于", "是一种类型",
    "分为", "按级别划分",
})


def is_weak_relation(rel: str) -> bool:
    """判定一条 relation 是否为上位概念/分类学边（弱语义边）。

    读侧实时判定，词表可调、零迁移；用于路径搜索区分强/弱边与前端诚实标注。
    归一化口径与图节点匹配共用 :func:`app.utils.norm_key`。
    """
    n = norm_key(rel)
    return n in _WEAK_RELATIONS or n.startswith("属于")

# 实体抽取 Prompt（通用措辞，适配传记/影视/百科等语料，2026-07-26 去技术域）
ENTITY_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """从以下问题中提取关键实体（人物、作品、地点、机构、职位、事件或其他关键名词）。
只输出实体名，每行一个，不要编号，不要解释。最多 5 个。

问题：{question}

实体列表："""
)


class GraphRetriever:
    """
    图检索器 - 基于知识图谱检索关系上下文。

    工作流程：
    1. 从用户查询中识别关键实体
    2. 在知识图谱中查找这些实体的关系
    3. 扩展 N 跳获取相关子图
    4. 将子图格式化为文本上下文
    """

    def __init__(self, graph_builder: Optional[KnowledgeGraphBuilder] = None, llm=None):
        settings = get_settings()
        self.graph_builder = graph_builder or get_graph_builder()
        # #17: LLM 添加超时和重试
        self.llm = llm or build_chat_llm(timeout=30, retries=2)
        self.max_hops = settings.graph_max_hops
        self.max_entities = settings.graph_max_entities

    def extract_entities(self, question: str, fast_only: bool = False) -> List[str]:
        """
        从用户查询中抽取关键实体。

        快速路径：先在图节点上做关键词匹配（零 LLM 调用），
        匹配失败时才回退到 LLM 抽取。

        Args:
            question: 用户问题
            fast_only: True 时仅走快速匹配，未命中即返回空（不回退 LLM）。
                分解路径的子问题检索用此模式——子问题数量多、LLM 回退是
                分解延迟的主因，且零命中说明图通道对该子问题无增益，
                交给 dense/sparse 即可（2026-07-28 零 LLM 均衡化）。

        Returns:
            实体名列表
        """
        # 快速路径：基于图节点的关键词匹配
        fast_entities = self._fast_entity_match(question)
        if fast_entities:
            logger.debug(f"快速实体匹配: {fast_entities}")
            return fast_entities

        if fast_only:
            return []

        # 回退：LLM 抽取
        return self._llm_extract_entities(question)

    def _fast_entity_match(self, question: str) -> List[str]:
        """基于图节点的快速实体匹配（无 LLM 调用）。

        复用 :meth:`KnowledgeGraphBuilder.match_nodes` 原语，配置为快速路径：
        单向包含（节点名出现在查询中）+ 度数降序 + 去子串 + 过滤单字节点。
        归一化口径（去空白+小写）与子图/路径检索一致，修复历史"仅 lower 漏去
        空白"导致带空格节点（如 "B+ 树"）失配的问题。
        """
        return self.graph_builder.match_nodes(
            question,
            top_k=self.max_entities,
            bidirectional=False,
            rank_by_degree=True,
            dedup_substrings=True,
            min_len=2,
        )

    def _llm_extract_entities(self, question: str) -> List[str]:
        """使用 LLM 从查询中抽取实体（回退路径）"""
        chain = ENTITY_EXTRACT_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"question": question})
            entities = [
                line.strip().strip("-•·")
                for line in result.strip().split("\n")
                if line.strip() and len(line.strip()) > 1
            ]
            entities = entities[:self.max_entities]
            logger.debug(f"查询实体抽取: {question[:30]}... → {entities}")
            return entities
        except Exception as e:
            logger.warning(f"实体抽取失败: {e}")
            return []

    def _collect_relations(self, entities: List[str]) -> List[dict]:
        """汇集所有实体的关系并按 (head, relation, tail) 去重（保序）。"""
        all_relations = []
        for entity in entities:
            all_relations.extend(self.graph_builder.get_entity_relations(entity))
        seen = set()
        unique = []
        for r in all_relations:
            key = (r["head"], r["relation"], r["tail"])
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def retrieve(
        self, question: str, top_k: int = 5, fast_only: bool = False
    ) -> List[Document]:
        """
        图检索主入口：抽取实体 → 子图扩展 → 格式化为 Document。

        Args:
            question: 用户问题
            top_k: 最多返回的关系文档数
            fast_only: 实体抽取仅走零 LLM 快速匹配（见 extract_entities）

        Returns:
            包含图上下文信息的 Document 列表
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            return []

        # Step 1: 抽取查询实体
        entities = self.extract_entities(question, fast_only=fast_only)
        if not entities:
            return []

        # Step 2: 获取每个实体的关系（含去重）
        unique_relations = self._collect_relations(entities)

        # Step 3: 获取子图（用于补充上下文）
        subgraph = self.graph_builder.get_subgraph(entities, max_hops=self.max_hops)

        # Step 4: 格式化为 Document
        documents = self._format_as_documents(unique_relations, subgraph, entities, top_k)

        logger.info(
            f"图检索: entities={entities}, relations={len(unique_relations)}, "
            f"subgraph_nodes={subgraph.number_of_nodes()}, docs={len(documents)}"
        )
        return documents

    def retrieve_with_context(self, question: str) -> dict:
        """
        带完整上下文的图检索（用于 API 返回详情）。

        Returns:
            {
                "entities": [...],
                "relations": [...],
                "subgraph_stats": {...},
                "context_text": "...",
            }
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            return {"entities": [], "relations": [], "subgraph_stats": {}, "context_text": ""}

        entities = self.extract_entities(question)
        if not entities:
            return {"entities": [], "relations": [], "subgraph_stats": {}, "context_text": ""}

        # 获取关系（含去重）
        unique_relations = self._collect_relations(entities)

        # 子图
        subgraph = self.graph_builder.get_subgraph(entities, max_hops=self.max_hops)

        # 格式化文本
        context_text = self._format_context_text(unique_relations, subgraph, entities)

        return {
            "entities": entities,
            "relations": unique_relations[:20],
            "subgraph_stats": {
                "num_nodes": subgraph.number_of_nodes(),
                "num_edges": subgraph.number_of_edges(),
            },
            "context_text": context_text,
        }

    def _format_as_documents(
        self,
        relations: List[dict],
        subgraph,
        entities: List[str],
        top_k: int,
    ) -> List[Document]:
        """将图检索结果格式化为 Document 列表"""
        documents = []

        # 1. 关系三元组文档（每个实体的关系汇总为一个 Document）
        for entity in entities:
            entity_rels = [
                r for r in relations
                if r["head"] == entity or r["tail"] == entity
            ]
            if not entity_rels:
                continue

            # 格式化为文本
            lines = [f"关于「{entity}」的知识图谱关系："]
            for r in entity_rels[:8]:  # 每个实体最多 8 条关系
                lines.append(f"  - {r['head']} → [{r['relation']}] → {r['tail']}")

            # chunk 溯源：graph: 前缀避免与真实分块在 RRF 中同 key 互覆盖，
            # source_chunk_ids 保留完整来源列表（接通 F7 引用溯源）
            chunk_ids = sorted({r["chunk_id"] for r in entity_rels if r.get("chunk_id")})
            metadata = {
                "source": "knowledge_graph",
                "type": "graph_relations",
                "entity": entity,
                "num_relations": len(entity_rels),
                "source_chunk_ids": chunk_ids,
            }
            if chunk_ids:
                metadata["chunk_id"] = "graph:" + chunk_ids[0]

            doc = Document(
                page_content="\n".join(lines),
                metadata=metadata,
            )
            documents.append(doc)

        # 2. 子图路径文档（如果子图足够丰富）
        if subgraph.number_of_edges() > len(relations):
            context_text = self._format_context_text(relations, subgraph, entities)
            if context_text:
                doc = Document(
                    page_content=f"知识图谱子图上下文：\n{context_text}",
                    metadata={
                        "source": "knowledge_graph",
                        "type": "graph_subgraph",
                        "num_nodes": subgraph.number_of_nodes(),
                        "num_edges": subgraph.number_of_edges(),
                    },
                )
                documents.append(doc)

        return documents[:top_k]

    def _format_context_text(
        self,
        relations: List[dict],
        subgraph,
        entities: List[str],
    ) -> str:
        """将子图格式化为自然语言上下文"""
        lines = []

        # 直接关系
        if relations:
            lines.append("【直接关系】")
            for r in relations[:15]:
                lines.append(f"  {r['head']} —[{r['relation']}]→ {r['tail']}")

        # 子图中的额外边（不在直接关系中的）
        direct_keys = {(r["head"], r["tail"]) for r in relations}
        extra_edges = []
        for head, tail, data in subgraph.edges(data=True):
            if (head, tail) not in direct_keys:
                extra_edges.append((head, data.get("relation", "相关"), tail))

        if extra_edges:
            lines.append("\n【扩展关系】")
            for head, rel, tail in extra_edges[:10]:
                lines.append(f"  {head} —[{rel}]→ {tail}")

        return "\n".join(lines)

    def find_path(self, entity_a: str, entity_b: str) -> dict:
        """查找两个实体之间的关系路径，并区分语义强/弱。

        返回 dict::

            {
              "found": bool,
              "path": [{"from", "relation", "to", "weak"}, ...],
              "path_length": int,
              "weak_only": bool,  # 多跳且含上位概念边 → 仅 taxonomy 桥接，语义弱
            }

        路径选择：优先返回"剔除弱边后的纯实义最短路径"；若两实体只能靠含弱边
        的路径连通，则退回全图最短路径。weak_only 仅在 path_length>1 且路径含弱边
        时为真——直连边（哪怕分类边）视为抽取器的事实陈述，不贬低。
        """
        import networkx as nx

        empty = {"found": False, "path": [], "path_length": 0, "weak_only": False}

        # 模糊匹配（归一化口径，top_k=1 取最佳匹配节点）
        nodes_a = self.graph_builder.match_nodes(entity_a, top_k=1)
        nodes_b = self.graph_builder.match_nodes(entity_b, top_k=1)
        if not nodes_a or not nodes_b:
            return empty

        source, target = nodes_a[0], nodes_b[0]
        graph = self.graph_builder.graph

        # 全图无向 + 仅强边无向子图
        undirected = graph.to_undirected()
        strong = nx.Graph()
        strong.add_nodes_from(graph.nodes())
        for u, v, data in graph.edges(data=True):
            if not is_weak_relation(data.get("relation", "")):
                strong.add_edge(u, v)

        def _try_path(g):
            try:
                return nx.shortest_path(g, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        strong_path = _try_path(strong)
        any_path = _try_path(undirected)
        chosen = strong_path or any_path
        if not chosen:
            return empty

        # 逐跳提取关系（原图为 DiGraph，需正向/反向兜底取边属性）
        hops = []
        for i in range(len(chosen) - 1):
            edge_data = graph.get_edge_data(chosen[i], chosen[i + 1])
            if edge_data is None:
                edge_data = graph.get_edge_data(chosen[i + 1], chosen[i])
            relation = edge_data.get("relation", "相关") if edge_data else "相关"
            hops.append({
                "from": chosen[i],
                "relation": relation,
                "to": chosen[i + 1],
                "weak": is_weak_relation(relation),
            })

        # 多跳（跳数>1）且含弱边 → 仅靠 taxonomy 桥接，语义关联较弱。
        # 注意 chosen 是节点序列，跳数 = len(hops)；直连（1 跳）即便分类边也不贬低。
        weak_only = len(hops) > 1 and any(h["weak"] for h in hops)

        return {
            "found": True,
            "path": hops,
            "path_length": len(hops),
            "weak_only": weak_only,
        }
