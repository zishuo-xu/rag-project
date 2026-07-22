"""层级索引构建 - 文档摘要索引(L1) + 段落明细索引(L2)"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from config import get_settings

logger = logging.getLogger(__name__)


def get_embeddings():
    """获取 Embedding 模型实例（支持本地模型和 OpenAI）"""
    settings = get_settings()

    if settings.embedding_provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url,
        )


def get_llm() -> ChatOpenAI:
    """获取 OpenAI LLM 实例"""
    settings = get_settings()
    # #17: LLM 添加超时和重试
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
        request_timeout=60,
        max_retries=2,
    )


def generate_document_summary(documents: List[Document], llm=None) -> str:
    """
    使用 LLM 生成文档级摘要。

    Args:
        documents: 同一文档的所有分块
        llm: LLM 实例

    Returns:
        文档摘要文本
    """
    if llm is None:
        llm = get_llm()

    # 拼接文档内容（截断避免超长）
    content = "\n".join([doc.page_content for doc in documents])
    content = content[:8000]  # 限制长度

    prompt = f"""请为以下文档生成一段简洁的摘要（200字以内），概括文档的核心主题和关键信息：

{content}

摘要："""

    response = llm.invoke(prompt)
    return response.content


class HierarchicalIndexer:
    """
    层级索引管理器

    L1 - 摘要索引：文档级摘要的向量索引，用于粗粒度定位
    L2 - 明细索引：段落分块的向量索引，用于细粒度检索
    """

    def __init__(self, embeddings=None, llm=None):
        self.settings = get_settings()
        self.embeddings = embeddings or get_embeddings()
        self.llm = llm or get_llm()
        self._chunk_store: Optional[Chroma] = None
        self._summary_store: Optional[Chroma] = None

    @property
    def chunk_store(self) -> Chroma:
        """获取/创建明细索引 (L2)"""
        if self._chunk_store is None:
            self._chunk_store = Chroma(
                collection_name=self.settings.chroma_chunk_collection,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
            )
        return self._chunk_store

    @property
    def summary_store(self) -> Chroma:
        """获取/创建摘要索引 (L1)"""
        if self._summary_store is None:
            self._summary_store = Chroma(
                collection_name=self.settings.chroma_summary_collection,
                embedding_function=self.embeddings,
                persist_directory=self.settings.chroma_persist_dir,
            )
        return self._summary_store

    def index_documents(self, chunks: List[Document], build_summary: bool = True):
        """
        将分块文档写入层级索引。

        Args:
            chunks: 分块后的文档列表
            build_summary: 是否构建摘要索引
        """
        # L2: 写入明细索引
        self.chunk_store.add_documents(chunks)
        logger.info(f"L2 明细索引写入: {len(chunks)} 个分块")

        # L1: 按文档分组，生成摘要并写入
        if build_summary:
            doc_groups = self._group_by_document(chunks)
            summaries: List[Document] = []

            for doc_id, doc_chunks in doc_groups.items():
                try:
                    summary_text = generate_document_summary(doc_chunks, self.llm)
                    summary_doc = Document(
                        page_content=summary_text,
                        metadata={
                            "doc_id": doc_id,
                            "source": doc_chunks[0].metadata.get("source", ""),
                            "num_chunks": len(doc_chunks),
                            "type": "summary",
                        },
                    )
                    summaries.append(summary_doc)
                except Exception as e:
                    logger.warning(f"生成摘要失败 (doc_id={doc_id}): {e}")

            if summaries:
                self.summary_store.add_documents(summaries)
                logger.info(f"L1 摘要索引写入: {len(summaries)} 个文档摘要")

    def _group_by_document(self, chunks: List[Document]) -> dict:
        """按 doc_id 分组"""
        groups = {}
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            if doc_id not in groups:
                groups[doc_id] = []
            groups[doc_id].append(chunk)
        return groups

    def search_chunks(self, query: str, top_k: int = 10) -> List[Document]:
        """L2 明细检索"""
        return self.chunk_store.similarity_search(query, k=top_k)

    def search_summaries(self, query: str, top_k: int = 3) -> List[Document]:
        """L1 摘要检索 - 定位相关文档"""
        return self.summary_store.similarity_search(query, k=top_k)

    def hierarchical_search(self, query: str, top_k: int = 5) -> List[Document]:
        """
        层级检索：先通过摘要定位文档，再在文档内精细检索。

        流程：
        1. L1 摘要检索 -> 找到最相关的文档
        2. L2 在相关文档内做明细检索

        Args:
            query: 查询文本
            top_k: 最终返回的文档数

        Returns:
            精排后的文档列表
        """
        # Step 1: 摘要检索定位文档
        summaries = self.search_summaries(query, top_k=3)
        target_doc_ids = [s.metadata.get("doc_id") for s in summaries]
        logger.info(f"层级检索 - L1 定位文档: {target_doc_ids}")

        # Step 2: 在目标文档内做明细检索
        if target_doc_ids:
            # 使用 where 过滤限定文档范围
            try:
                results = self.chunk_store.similarity_search(
                    query,
                    k=top_k,
                    filter={"doc_id": {"$in": target_doc_ids}},
                )
                return results
            except Exception:
                # 过滤失败时回退到全局检索
                pass

        # 回退：全局明细检索
        return self.search_chunks(query, top_k=top_k)

    def get_all_chunks(self) -> List[Document]:
        """获取所有已索引的分块（用于 BM25 构建）"""
        collection = self.chunk_store._collection
        data = collection.get(include=["documents", "metadatas"])
        docs = []
        for i, text in enumerate(data["documents"]):
            metadata = data["metadatas"][i] if data["metadatas"] else {}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def get_chunk_embeddings(self, doc_id: str) -> List[dict]:
        """
        获取指定文档所有分块的向量信息（用于可视化）。

        Returns:
            [{chunk_id, position, vector_dim, vector_preview, norm}]
        """
        collection = self.chunk_store._collection
        data = collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "embeddings"],
        )
        results = []
        embeddings = data["embeddings"]
        if embeddings is None:
            return results
        for i, emb in enumerate(embeddings):
            metadata = data["metadatas"][i] if data["metadatas"] else {}
            vec = list(emb)
            norm = sum(v * v for v in vec) ** 0.5
            results.append({
                "chunk_id": metadata.get("chunk_id", ""),
                "position": metadata.get("position", 0),
                "vector_dim": len(vec),
                "vector_preview": [round(v, 6) for v in vec[:8]],
                "norm": round(norm, 6),
            })
        results.sort(key=lambda x: x["position"])
        return results

    def get_document_summary(self, doc_id: str) -> Optional[str]:
        """获取指定文档的 L1 摘要"""
        collection = self.summary_store._collection
        data = collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        if data["documents"]:
            return data["documents"][0]
        return None
