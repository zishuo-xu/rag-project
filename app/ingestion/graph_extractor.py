"""知识图谱构建 - 使用 LLM 从文本中抽取实体和关系，构建 NetworkX 知识图谱

核心流程：
    文档分块 → LLM 抽取三元组 (头实体, 关系, 尾实体) → 写入 NetworkX 图 → 持久化

设计决策：
    - 使用 LLM-as-Extractor 而非 NER 模型，因为 LLM 能理解上下文语义关系
    - 使用 NetworkX 而非 Neo4j，因为当前数据规模（10篇文档）无需分布式图数据库
    - 图存储抽象为接口，生产环境可无缝切换 Neo4j
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings, build_chat_llm

logger = logging.getLogger(__name__)

# 实体类型集（head_type/tail_type 必须从中选择，未知类型归一为 other）
ENTITY_TYPES = ("person", "work", "place", "org", "position", "event", "other")

# 实体/关系抽取 Prompt（JSON 类型化三元组 + 传记/影视体 few-shot，2026-07-26 升级）
EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """你是一个知识图谱构建专家。请从以下文本中抽取实体和关系，以 JSON 输出。

## 实体类型（head_type / tail_type 必须从以下选择）
- person（人物）、work（作品/影视/书籍）、place（地点）、org（组织/机构）、position（职位/头衔）、event（事件/奖项/活动）、other（其他关键概念）

## 要求
1. 关系 relation 用简短动词/短语（如"饰演"、"导演"、"位于"、"获得"、"担任"）
2. 实体名保持原文，不翻译不缩写；不提取代词和通用词
3. 最多提取 15 个最重要的三元组
4. 仅输出 JSON，不要任何解释或代码块标记

## 示例
文本：周润发在《卧虎藏龙》中饰演李慕白，该片由李安导演，获得奥斯卡最佳外语片奖。
输出：{{"triples": [{{"head": "周润发", "head_type": "person", "relation": "饰演", "tail": "李慕白", "tail_type": "person"}}, {{"head": "周润发", "head_type": "person", "relation": "出演", "tail": "卧虎藏龙", "tail_type": "work"}}, {{"head": "李安", "head_type": "person", "relation": "导演", "tail": "卧虎藏龙", "tail_type": "work"}}, {{"head": "卧虎藏龙", "head_type": "work", "relation": "获得", "tail": "奥斯卡最佳外语片奖", "tail_type": "event"}}]}}

## 文本
{text}

## 输出（仅 JSON，格式：{{"triples": [{{"head": "...", "head_type": "...", "relation": "...", "tail": "...", "tail_type": "..."}}]}}）"""
)


class KnowledgeGraphBuilder:
    """
    知识图谱构建器。

    使用 LLM 从文档分块中抽取实体和关系，
    构建 NetworkX 有向图，支持持久化到 JSON。

    优化特性：
    - 并发 LLM 调用（asyncio + Semaphore 限流）
    - 增量构建（跳过已处理的分块）
    - 实时进度追踪
    - 相邻分块合并（减少 LLM 调用次数）
    """

    # 并发 LLM 调用数上限
    MAX_CONCURRENCY = 5
    # 合并相邻分块的最大数量
    MERGE_CHUNK_SIZE = 2

    def __init__(self, llm=None):
        settings = get_settings()
        self.llm = llm or build_chat_llm()
        self.graph = nx.DiGraph()
        self.persist_path = Path(settings.graph_persist_path)
        self.meta_path = self.persist_path.with_suffix(".meta.json")

        # 构建进度状态
        self.build_state: dict = {
            "status": "idle",  # idle | building | completed | failed
            "processed": 0,
            "total": 0,
            "triples_extracted": 0,
            "skipped": 0,
            "started_at": None,
            "elapsed_seconds": 0,
            "error": None,
        }

        # 尝试加载已有图谱
        if self.persist_path.exists():
            self._load()

    def extract_triples(self, text: str) -> List[dict]:
        """
        使用 LLM 从文本中抽取类型化三元组。

        Args:
            text: 输入文本

        Returns:
            [{"head", "head_type", "relation", "tail", "tail_type"}, ...]
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

    def _parse_triples(self, text: str) -> List[dict]:
        """
        解析 LLM 输出的三元组（2026-07-26 升级：JSON 类型化优先，逐行解析兜底）。

        优先按 JSON 解析 {"triples": [{head, head_type, relation, tail, tail_type}]}
        （容忍 ```json 代码块包裹）；解析失败或无有效三元组时回退旧版逐行格式
        （头实体, 关系, 尾实体），类型统一置 other。

        Returns:
            [{"head", "head_type", "relation", "tail", "tail_type"}, ...]
        """
        # ---- JSON 路径 ----
        body = text.strip()
        body = re.sub(r"^```[a-zA-Z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(body[start:end + 1])
                raw = data.get("triples", []) if isinstance(data, dict) else []
                triples = [self._normalize_triple(t) for t in raw if isinstance(t, dict)]
                # JSON 解析成功即以 JSON 为准（即使全无效也不回退逐行，
                # 否则会把 JSON 文本按逗号拆成垃圾三元组）
                return [t for t in triples if t]
            except (json.JSONDecodeError, ValueError):
                pass

        # ---- 兜底：旧版逐行解析（类型置 other） ----
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
                    triples.append({
                        "head": head, "head_type": "other",
                        "relation": relation,
                        "tail": tail, "tail_type": "other",
                    })

        return triples

    @staticmethod
    def _normalize_triple(raw: dict) -> Optional[dict]:
        """校验并归一化单个 JSON 三元组；无效返回 None。"""
        head = str(raw.get("head", "")).strip().strip('"\'')
        relation = str(raw.get("relation", "")).strip().strip('"\'')
        tail = str(raw.get("tail", "")).strip().strip('"\'')
        if not (head and relation and tail and len(head) > 1 and len(tail) > 1):
            return None

        def _norm_type(value) -> str:
            t = str(value or "").strip().lower()
            return t if t in ENTITY_TYPES else "other"

        return {
            "head": head,
            "head_type": _norm_type(raw.get("head_type")),
            "relation": relation,
            "tail": tail,
            "tail_type": _norm_type(raw.get("tail_type")),
        }

    def _add_triples(self, triples: List[dict], source: str, chunk_id: str) -> None:
        """将类型化三元组写入图（节点带类型与来源，边带关系 + chunk 溯源 + 端点类型）。"""
        for t in triples:
            head, tail = t["head"], t["tail"]
            self.graph.add_node(head, type=t["head_type"], source=source)
            self.graph.add_node(tail, type=t["tail_type"], source=source)
            self.graph.add_edge(
                head, tail, relation=t["relation"], source=source,
                chunk_id=chunk_id, head_type=t["head_type"], tail_type=t["tail_type"],
            )

    # ============ 异步构建（并发 + 增量 + 进度追踪） ============

    @property
    def is_building(self) -> bool:
        """是否正在构建中"""
        return self.build_state["status"] == "building"

    def get_build_progress(self) -> dict:
        """获取构建进度快照"""
        state = self.build_state.copy()
        if state["status"] == "building" and state["started_at"]:
            state["elapsed_seconds"] = round(time.time() - state["started_at"], 1)
        return state

    async def build_from_documents_async(
        self,
        documents: List[Document],
        incremental: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        异步并发构建知识图谱。

        优化策略：
        1. 增量构建：跳过已处理过的 chunk（基于 chunk_id）
        2. 分块合并：同文档相邻 2 块合并为一次 LLM 调用，减少 50% 请求
        3. 并发控制：Semaphore 限制最多 5 个并发 LLM 请求
        4. 进度追踪：实时更新 build_state，支持外部轮询

        Args:
            documents: 文档分块列表
            incremental: 是否增量构建（跳过已处理分块）
            progress_callback: 进度回调函数

        Returns:
            构建统计信息
        """
        if self.is_building:
            raise RuntimeError("图谱正在构建中，请等待完成")

        processed_ids = self._load_processed_ids() if incremental else set()

        # 过滤已处理的分块
        if incremental and processed_ids:
            pending_docs = [
                doc for doc in documents
                if doc.metadata.get("chunk_id", "") not in processed_ids
            ]
            skipped = len(documents) - len(pending_docs)
        else:
            pending_docs = documents
            skipped = 0

        # 合并相邻分块（同文档的相邻 chunk 两两合并）
        tasks_units = self._merge_chunks(pending_docs)

        # 初始化构建状态
        self.build_state = {
            "status": "building",
            "processed": 0,
            "total": len(tasks_units),
            "triples_extracted": 0,
            "skipped": skipped,
            "started_at": time.time(),
            "elapsed_seconds": 0,
            "error": None,
        }
        self._notify_progress(progress_callback)

        semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)
        total_triples = 0
        newly_processed_ids: List[str] = []

        async def process_unit(unit: dict):
            """处理一个合并单元（1-2 个分块）"""
            nonlocal total_triples
            async with semaphore:
                try:
                    triples = await self._extract_triples_async(unit["text"])
                except Exception as e:
                    logger.warning(f"三元组抽取失败: {e}")
                    triples = []

                # 写入图（NetworkX 非线程安全，但 asyncio 是单线程事件循环，安全）
                # 合并单元溯源到首个 chunk（相邻块合并抽取，三元组归属近似）
                chunk_id = unit["chunk_ids"][0] if unit["chunk_ids"] else ""
                self._add_triples(triples, unit["source"], chunk_id)

                total_triples += len(triples)
                newly_processed_ids.extend(unit["chunk_ids"])

                # 更新进度
                self.build_state["processed"] += 1
                self.build_state["triples_extracted"] = total_triples

                # 每 5 个单元做一次检查点保存（防止中途中断丢失全部进度）
                if self.build_state["processed"] % 5 == 0:
                    # 在线程池中执行 IO，不阻塞事件循环
                    await asyncio.to_thread(self._save)
                    await asyncio.to_thread(
                        self._save_processed_ids, processed_ids | set(newly_processed_ids)
                    )
                    logger.info(
                        f"图谱构建检查点: {self.build_state['processed']}/{self.build_state['total']} "
                        f"单元, {total_triples} 三元组"
                    )

                self._notify_progress(progress_callback)

        try:
            # 并发执行所有抽取任务
            await asyncio.gather(*[process_unit(u) for u in tasks_units])

            # 持久化图谱 + 已处理 ID
            self._save()
            self._save_processed_ids(processed_ids | set(newly_processed_ids))

            elapsed = time.time() - self.build_state["started_at"]
            self.build_state.update({
                "status": "completed",
                "elapsed_seconds": round(elapsed, 1),
            })
            self._notify_progress(progress_callback)

            stats = {
                "processed_units": len(tasks_units),
                "total_triples": total_triples,
                "skipped_chunks": skipped,
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "elapsed_seconds": round(elapsed, 1),
                "concurrency": self.MAX_CONCURRENCY,
            }
            logger.info(f"知识图谱异步构建完成: {stats}")
            return stats

        except Exception as e:
            self.build_state.update({
                "status": "failed",
                "error": str(e),
                "elapsed_seconds": round(time.time() - self.build_state["started_at"], 1),
            })
            self._notify_progress(progress_callback)
            raise

    async def _extract_triples_async(self, text: str) -> List[dict]:
        """异步 LLM 三元组抽取（类型化 dict）"""
        chain = EXTRACTION_PROMPT | self.llm | StrOutputParser()
        result = await chain.ainvoke({"text": text[:2000]})
        triples = self._parse_triples(result)
        logger.debug(f"异步抽取到 {len(triples)} 个三元组")
        return triples

    def _merge_chunks(self, documents: List[Document]) -> List[dict]:
        """
        将同文档的相邻分块两两合并，减少 LLM 调用次数。

        Returns:
            [{"text": ..., "source": ..., "chunk_ids": [...]}]
        """
        # 按 doc_id 分组，保持 position 顺序
        groups: dict = {}
        for doc in documents:
            doc_id = doc.metadata.get("doc_id", "unknown")
            if doc_id not in groups:
                groups[doc_id] = []
            groups[doc_id].append(doc)

        units = []
        for doc_id, docs in groups.items():
            docs.sort(key=lambda d: d.metadata.get("position", 0))
            # 两两合并
            for i in range(0, len(docs), self.MERGE_CHUNK_SIZE):
                batch = docs[i:i + self.MERGE_CHUNK_SIZE]
                units.append({
                    "text": "\n\n".join(d.page_content for d in batch),
                    "source": batch[0].metadata.get("source", "unknown"),
                    "chunk_ids": [
                        d.metadata.get("chunk_id", f"{doc_id}_{j}")
                        for j, d in enumerate(batch)
                    ],
                })
        return units

    def _notify_progress(self, callback: Optional[Callable] = None):
        """触发进度回调"""
        if callback:
            try:
                callback(self.get_build_progress())
            except Exception:
                pass

    def _load_processed_ids(self) -> set:
        """加载已处理的 chunk_id 集合"""
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return set(meta.get("processed_chunk_ids", []))
            except Exception:
                pass
        return set()

    def _save_processed_ids(self, ids: set):
        """持久化已处理的 chunk_id 集合"""
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"processed_chunk_ids": sorted(ids)}, f, ensure_ascii=False)

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
            [{"head", "relation", "tail", "source", "chunk_id", "head_type", "tail_type"}, ...]
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
                    "chunk_id": data.get("chunk_id", ""),
                    "head_type": data.get("head_type", "other"),
                    "tail_type": data.get("tail_type", "other"),
                })
            # 入边
            for source, _, data in self.graph.in_edges(node, data=True):
                relations.append({
                    "head": source,
                    "relation": data.get("relation", ""),
                    "tail": node,
                    "source": data.get("source", ""),
                    "chunk_id": data.get("chunk_id", ""),
                    "head_type": data.get("head_type", "other"),
                    "tail_type": data.get("tail_type", "other"),
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
                "chunk_id": data.get("chunk_id", ""),
                "head_type": data.get("head_type", "other"),
                "tail_type": data.get("tail_type", "other"),
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
        # 归一化：去空白 + 小写，避免 "b+树" 与图节点 "B+ 树" 因空格/大小写失配
        query_norm = re.sub(r"\s+", "", query_lower)
        matched = []

        for node in self.graph.nodes():
            node_lower = node.lower()
            node_norm = re.sub(r"\s+", "", node_lower)
            # 完全匹配（归一化后）
            if node_norm == query_norm:
                matched.insert(0, node)
            # 包含匹配（归一化后）
            elif query_norm in node_norm or node_norm in query_norm:
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
        if self.meta_path.exists():
            self.meta_path.unlink()
        self.build_state["status"] = "idle"
        logger.info("知识图谱已清空")


# 全局单例
_graph_builder: Optional[KnowledgeGraphBuilder] = None


def get_graph_builder() -> KnowledgeGraphBuilder:
    """获取全局知识图谱构建器实例"""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = KnowledgeGraphBuilder()
    return _graph_builder
