"""Parent-Child 检索 - 小块精准匹配 + 大块完整上下文"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import get_settings
from app.ingestion.indexer import get_embeddings

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """
    Parent-Child 检索器。

    原理：
    - Parent（大块, 1024字符）：提供完整上下文，送入 LLM 生成回答
    - Child（小块, 200字符）：精准匹配查询语义，提高命中率

    流程：query -> 匹配 child chunks -> 去重 parent_id -> 返回 parent 内容
    """

    def __init__(self, embeddings=None):
        settings = get_settings()
        self.embeddings = embeddings or get_embeddings()
        self.parent_chunk_size = settings.parent_chunk_size
        self.child_chunk_size = settings.child_chunk_size

        self._child_store: Optional[Chroma] = None
        self._parent_store: Optional[Chroma] = None

    @property
    def child_store(self) -> Chroma:
        """获取/创建 child 索引"""
        if self._child_store is None:
            settings = get_settings()
            self._child_store = Chroma(
                collection_name=settings.chroma_child_collection,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
        return self._child_store

    @property
    def parent_store(self) -> Chroma:
        """获取/创建 parent 索引"""
        if self._parent_store is None:
            settings = get_settings()
            self._parent_store = Chroma(
                collection_name=settings.chroma_parent_collection,
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
        return self._parent_store

    def index_documents(self, documents: List[Document]):
        """
        构建 Parent-Child 双层索引。

        Args:
            documents: 原始文档（已加载，未分块）
        """
        settings = get_settings()

        # 生成 Parent 大块
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=128,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        parents = parent_splitter.split_documents(documents)

        # 为每个 parent 分配 ID
        parent_docs = []
        for i, p in enumerate(parents):
            parent_id = f"{p.metadata.get('doc_id', 'unknown')}_p{i}"
            p.metadata["parent_id"] = parent_id
            p.metadata["chunk_level"] = "parent"
            parent_docs.append(p)

        # 生成 Child 小块（从 parent 中再切分）
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )

        child_docs = []
        child_idx = 0
        for parent in parent_docs:
            # 将 parent 作为单个 Document 切分为 children
            temp_doc = Document(
                page_content=parent.page_content,
                metadata=parent.metadata.copy(),
            )
            children = child_splitter.split_documents([temp_doc])
            for child in children:
                child.metadata["parent_id"] = parent.metadata["parent_id"]
                child.metadata["child_id"] = f"{parent.metadata['parent_id']}_c{child_idx}"
                child.metadata["chunk_level"] = "child"
                child_docs.append(child)
                child_idx += 1

        # 写入 ChromaDB
        if parent_docs:
            self.parent_store.add_documents(parent_docs)
        if child_docs:
            self.child_store.add_documents(child_docs)

        logger.info(
            f"Parent-Child 索引构建: {len(parent_docs)} parents, "
            f"{len(child_docs)} children"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Parent-Child 检索：匹配 child -> 返回 parent。

        Args:
            query: 查询文本
            top_k: 返回的 parent 数量

        Returns:
            去重后的 parent 文档列表
        """
        # 匹配更多 child 以确保覆盖
        child_hits = self.child_store.similarity_search(query, k=top_k * 3)

        if not child_hits:
            logger.debug("Parent-Child: 无 child 命中")
            return []

        # 去重 parent_id，保持顺序
        seen_parents = set()
        parent_ids = []
        for child in child_hits:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_parents:
                seen_parents.add(pid)
                parent_ids.append(pid)
            if len(parent_ids) >= top_k:
                break

        # #10: 批量查询 parent（用 $in 一次查询代替逐个查询）
        results = []
        if parent_ids:
            try:
                batch_result = self.parent_store.get(
                    where={"parent_id": {"$in": parent_ids}},
                    include=["documents", "metadatas"],
                )
                # 建立 parent_id -> (document, metadata) 的映射
                pid_map = {}
                if batch_result["documents"] and batch_result["metadatas"]:
                    for doc_text, meta in zip(batch_result["documents"], batch_result["metadatas"]):
                        pid = meta.get("parent_id", "")
                        if pid:
                            pid_map[pid] = (doc_text, meta)

                # 按原始 parent_ids 顺序构建结果
                for pid in parent_ids:
                    if pid in pid_map:
                        doc_text, meta = pid_map[pid]
                        doc = Document(
                            page_content=doc_text,
                            metadata=meta.copy(),
                        )
                        doc.metadata["retrieval_method"] = "parent_child"
                        results.append(doc)
            except Exception as e:
                logger.warning(f"Parent-Child 批量查询失败，回退到逐个查询: {e}")
                # 回退：逐个查询
                for pid in parent_ids:
                    try:
                        parent_docs = self.parent_store.get(
                            where={"parent_id": pid},
                            include=["documents", "metadatas"],
                        )
                        if parent_docs["documents"]:
                            doc = Document(
                                page_content=parent_docs["documents"][0],
                                metadata=parent_docs["metadatas"][0] if parent_docs["metadatas"] else {},
                            )
                            doc.metadata["retrieval_method"] = "parent_child"
                            results.append(doc)
                    except Exception:
                        continue

        logger.info(
            f"Parent-Child 检索: {len(child_hits)} child hits -> "
            f"{len(results)} unique parents"
        )
        return results

    def has_index(self) -> bool:
        """检查是否已有 parent-child 索引"""
        try:
            count = self.child_store._collection.count()
            return count > 0
        except Exception:
            return False
