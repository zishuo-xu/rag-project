"""图检索器 - 基于知识图谱的关系检索与子图扩展

核心能力：
    1. 实体识别：从用户查询中识别关键实体
    2. 子图检索：以识别的实体为起点，N 跳扩展获取相关子图
    3. 路径检索：查找两个实体之间的关系路径
    4. 上下文格式化：将子图三元组转为 LLM 可理解的文本

设计决策：
    - 图检索作为向量检索的补充（而非替代），提供关系推理能力
    - 使用 LLM 从查询中抽取实体（比 NER 更灵活）
    - 格式化输出为自然语言三元组列表，方便 LLM 理解
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from app.ingestion.graph_extractor import get_graph_builder, KnowledgeGraphBuilder

logger = logging.getLogger(__name__)

# 实体抽取 Prompt
ENTITY_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    """从以下问题中提取关键技术实体（技术名词、工具、算法、协议、概念等）。
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
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            request_timeout=30,
            max_retries=2,
        )
        self.max_hops = settings.graph_max_hops
        self.max_entities = settings.graph_max_entities

    def extract_entities(self, question: str) -> List[str]:
        """
        从用户查询中抽取关键实体。

        快速路径：先在图节点上做关键词匹配（零 LLM 调用），
        匹配失败时才回退到 LLM 抽取。

        Args:
            question: 用户问题

        Returns:
            实体名列表
        """
        # 快速路径：基于图节点的关键词匹配
        fast_entities = self._fast_entity_match(question)
        if fast_entities:
            logger.debug(f"快速实体匹配: {fast_entities}")
            return fast_entities

        # 回退：LLM 抽取
        return self._llm_extract_entities(question)

    def _fast_entity_match(self, question: str) -> List[str]:
        """
        基于图节点的快速实体匹配（无 LLM 调用）。

        策略：将查询与图中所有节点名做包含匹配，
        按节点度数排序返回 top-K。
        """
        graph = self.graph_builder.graph
        if graph.number_of_nodes() == 0:
            return []

        question_lower = question.lower()
        degrees = dict(graph.degree())
        matched = []

        for node in graph.nodes():
            node_lower = node.lower()
            # 节点名必须 >= 2 字符且出现在查询中
            if len(node_lower) >= 2 and node_lower in question_lower:
                matched.append((node, degrees.get(node, 0)))

        # 按度数降序，取 top-K
        matched.sort(key=lambda x: x[1], reverse=True)

        # 去除被更长实体包含的子串（如 "Redis" 和 "Redis Cluster"）
        result = []
        for node, deg in matched:
            if not any(node.lower() != m.lower() and node.lower() in m.lower() for m in result):
                result.append(node)
            if len(result) >= self.max_entities:
                break

        return result[:self.max_entities]

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

    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        """
        图检索主入口：抽取实体 → 子图扩展 → 格式化为 Document。

        Args:
            question: 用户问题
            top_k: 最多返回的关系文档数

        Returns:
            包含图上下文信息的 Document 列表
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            return []

        # Step 1: 抽取查询实体
        entities = self.extract_entities(question)
        if not entities:
            return []

        # Step 2: 获取每个实体的关系
        all_relations = []
        for entity in entities:
            relations = self.graph_builder.get_entity_relations(entity)
            all_relations.extend(relations)

        # 去重
        seen = set()
        unique_relations = []
        for r in all_relations:
            key = (r["head"], r["relation"], r["tail"])
            if key not in seen:
                seen.add(key)
                unique_relations.append(r)

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

        # 获取关系
        all_relations = []
        for entity in entities:
            relations = self.graph_builder.get_entity_relations(entity)
            all_relations.extend(relations)

        # 去重
        seen = set()
        unique_relations = []
        for r in all_relations:
            key = (r["head"], r["relation"], r["tail"])
            if key not in seen:
                seen.add(key)
                unique_relations.append(r)

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

            doc = Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "knowledge_graph",
                    "type": "graph_relations",
                    "entity": entity,
                    "num_relations": len(entity_rels),
                },
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

    def find_path(self, entity_a: str, entity_b: str) -> List[dict]:
        """
        查找两个实体之间的关系路径。

        Args:
            entity_a: 起始实体
            entity_b: 目标实体

        Returns:
            路径上的关系列表
        """
        import networkx as nx

        # 模糊匹配
        nodes_a = self.graph_builder._fuzzy_match_nodes(entity_a, top_k=1)
        nodes_b = self.graph_builder._fuzzy_match_nodes(entity_b, top_k=1)

        if not nodes_a or not nodes_b:
            return []

        source = nodes_a[0]
        target = nodes_b[0]

        try:
            # 在无向版本中找最短路径
            undirected = self.graph_builder.graph.to_undirected()
            path = nx.shortest_path(undirected, source, target)

            # 提取路径上的关系
            path_relations = []
            for i in range(len(path) - 1):
                edge_data = self.graph_builder.graph.get_edge_data(path[i], path[i + 1])
                if edge_data is None:
                    # 可能是反向边
                    edge_data = self.graph_builder.graph.get_edge_data(path[i + 1], path[i])
                relation = edge_data.get("relation", "相关") if edge_data else "相关"
                path_relations.append({
                    "from": path[i],
                    "relation": relation,
                    "to": path[i + 1],
                })

            return path_relations
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []
