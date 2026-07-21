"""知识图谱构建 - 使用 LLM 从文本中抽取实体和关系，构建 NetworkX 知识图谱

核心流程：
    文档分块 → LLM 抽取三元组 (头实体, 关系, 尾实体) → 写入 NetworkX 图 → 持久化

设计决策：
    - 使用 LLM-as-Extractor 而非 NER 模型，因为 LLM 能理解上下文语义关系
    - 使用 NetworkX 而非 Neo4j，因为当前数据规模（10篇文档）无需分布式图数据库
    - 图存储抽象为接口，生产环境可无缝切换 Neo4j
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings

logger = logging.getLogger(__name__)

# 实体/关系抽取 Prompt
EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """你是一个知识图谱构建专家。请从以下文本中抽取所有有意义的实体和关系。

## 要求
1. 实体：提取关键概念、技术名词、工具、算法、协议等（不要提取代词和通用词）
2. 关系：用简短动词/短语描述实体间的关系（如"使用"、"包含"、"优于"、"解决"）
3. 每个三元组格式：(头实体, 关系, 尾实体)
4. 每行一个三元组，不要编号
5. 实体名保持原文，不要翻译或缩写
6. 最多提取 15 个最重要的三元组

## 文本
{text}

## 输出（每行一个三元组，格式：头实体, 关系, 尾实体）"""
)


class KnowledgeGraphBuilder:
    """
    知识图谱构建器。

    使用 LLM 从文档分块中抽取实体和关系，
    构建 NetworkX 有向图，支持持久化到 JSON。
    """

    def __init__(self, llm=None):
        settings = get_settings()
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.graph = nx.DiGraph()
        self.persist_path = Path(settings.graph_persist_path)

        # 尝试加载已有图谱
        if self.persist_path.exists():
            self._load()

    def extract_triples(self, text: str) -> List[tuple]:
        """
        使用 LLM 从文本中抽取三元组。

        Args:
            text: 输入文本

        Returns:
            [(head, relation, tail), ...] 三元组列表
        """
        chain = EXTRACTION_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"text": text[:2000]})  # 限制长度
            triples = self._parse_triples(result)
            logger.debug(f"抽取到 {len(triples)} 个三元组")
            return triples
        except Exception as e:
            logger.warning(f"三元组抽取失败: {e}")
            return []

    def _parse_triples(self, text: str) -> List[tuple]:
        """
        解析 LLM 输出的三元组文本。

        支持格式：
        - (A, 关系, B)
        - A, 关系, B
        - (A, 关系, B)
        """
        triples = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # 去除可能的编号前缀 (1. / 1) / -)
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)

            # 去除外层括号
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1]
            if line.startswith("（") and line.endswith("）"):
                line = line[1:-1]

            # 按逗号分割（支持中英文逗号）
            parts = re.split(r'[,，、]', line)
            if len(parts) >= 3:
                head = parts[0].strip().strip('"\'')
                relation = parts[1].strip().strip('"\'')
                tail = parts[2].strip().strip('"\'')

                # 过滤无效三元组
                if head and relation and tail and len(head) > 1 and len(tail) > 1:
                    triples.append((head, relation, tail))

        return triples

    def build_from_documents(self, documents: List[Document]) -> dict:
        """
        从文档分块批量构建知识图谱。

        Args:
            documents: 文档分块列表

        Returns:
            构建统计信息
        """
        total_triples = 0
        processed = 0

        for i, doc in enumerate(documents):
            text = doc.page_content
            source = doc.metadata.get("source", "unknown")

            triples = self.extract_triples(text)
            for head, relation, tail in triples:
                # 添加节点（带来源信息）
                self.graph.add_node(head, type="entity", source=source)
                self.graph.add_node(tail, type="entity", source=source)
                # 添加边（关系）
                self.graph.add_edge(head, tail, relation=relation, source=source)
                total_triples += 1

            processed += 1
            if (i + 1) % 5 == 0:
                logger.info(f"图谱构建进度: {i+1}/{len(documents)} 文档, 累计 {total_triples} 三元组")

        # 持久化
        self._save()

        stats = {
            "processed_docs": processed,
            "total_triples": total_triples,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }
        logger.info(f"知识图谱构建完成: {stats}")
        return stats

    def get_subgraph(self, entities: List[str], max_hops: int = 2) -> nx.DiGraph:
        """
        获取指定实体的 N 跳子图。

        Args:
            entities: 起始实体列表
            max_hops: 最大扩展跳数

        Returns:
            子图
        """
        nodes = set()
        for entity in entities:
            # 模糊匹配：找到图中包含该关键词的节点
            matched = self._fuzzy_match_nodes(entity)
            nodes.update(matched)

        if not nodes:
            return nx.DiGraph()

        # BFS 扩展 N 跳
        expanded = set(nodes)
        frontier = set(nodes)
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                # 出边邻居
                next_frontier.update(self.graph.successors(node))
                # 入边邻居
                next_frontier.update(self.graph.predecessors(node))
            expanded.update(next_frontier)
            frontier = next_frontier - expanded

        return self.graph.subgraph(expanded).copy()

    def get_entity_relations(self, entity: str) -> List[dict]:
        """
        获取实体的所有关系（入边 + 出边）。

        Returns:
            [{"head": ..., "relation": ..., "tail": ...}, ...]
        """
        matched = self._fuzzy_match_nodes(entity)
        relations = []

        for node in matched:
            # 出边
            for _, target, data in self.graph.out_edges(node, data=True):
                relations.append({
                    "head": node,
                    "relation": data.get("relation", ""),
                    "tail": target,
                    "source": data.get("source", ""),
                })
            # 入边
            for source, _, data in self.graph.in_edges(node, data=True):
                relations.append({
                    "head": source,
                    "relation": data.get("relation", ""),
                    "tail": node,
                    "source": data.get("source", ""),
                })

        return relations

    def get_all_triples(self, limit: int = 200) -> List[dict]:
        """获取所有三元组（用于可视化）"""
        triples = []
        for head, tail, data in self.graph.edges(data=True):
            triples.append({
                "head": head,
                "relation": data.get("relation", ""),
                "tail": tail,
                "source": data.get("source", ""),
            })
            if len(triples) >= limit:
                break
        return triples

    def get_stats(self) -> dict:
        """获取图谱统计信息"""
        if self.graph.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0, "is_empty": True}

        # 度最高的实体
        degree_sorted = sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True
        )
        top_entities = [
            {"entity": node, "degree": deg}
            for node, deg in degree_sorted[:10]
        ]

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_empty": False,
            "top_entities": top_entities,
            "density": round(nx.density(self.graph), 6),
        }

    def _fuzzy_match_nodes(self, query: str, top_k: int = 3) -> List[str]:
        """
        模糊匹配图节点（支持部分匹配）。

        Args:
            query: 查询关键词
            top_k: 最多返回数量

        Returns:
            匹配到的节点名列表
        """
        query_lower = query.lower()
        matched = []

        for node in self.graph.nodes():
            node_lower = node.lower()
            # 完全匹配
            if node_lower == query_lower:
                matched.insert(0, node)
            # 包含匹配
            elif query_lower in node_lower or node_lower in query_lower:
                matched.append(node)

        return matched[:top_k]

    def _save(self):
        """持久化图谱到 JSON"""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"知识图谱已保存: {self.persist_path}")

    def _load(self):
        """从 JSON 加载图谱"""
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, directed=True)
            logger.info(
                f"知识图谱已加载: {self.graph.number_of_nodes()} 节点, "
                f"{self.graph.number_of_edges()} 边"
            )
        except Exception as e:
            logger.warning(f"知识图谱加载失败: {e}, 使用空图")
            self.graph = nx.DiGraph()

    def clear(self):
        """清空图谱"""
        self.graph = nx.DiGraph()
        if self.persist_path.exists():
            self.persist_path.unlink()
        logger.info("知识图谱已清空")


# 全局单例
_graph_builder: Optional[KnowledgeGraphBuilder] = None


def get_graph_builder() -> KnowledgeGraphBuilder:
    """获取全局知识图谱构建器实例"""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = KnowledgeGraphBuilder()
    return _graph_builder
