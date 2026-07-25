"""摄入编排服务层 - 加载 → 分块 → 索引 → 增强索引 → BM25 增量

routes 的上传与重建索引端点共享同一编排；本层只负责摄入流程，
不处理 HTTP 关切（文件校验 / 临时文件 / 响应模型留在 routes）。
增强索引（Parent-Child / 上下文增强）失败一律降级为警告，绝不阻断主索引。

chain 参数为鸭子类型（避免 ingestion → generation 反向依赖），
需提供：indexer / parent_child_retriever / sparse_retriever /
rebuild_parent_child_index（见 RAGChain）。
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from config import get_settings
from app.ingestion.loader import load_document
from app.ingestion.chunker import smart_chunk

logger = logging.getLogger(__name__)


def _index_parent_child(chain, docs: List[Document]) -> None:
    """F6a Parent-Child 接线（小块检索大块返回）。

    注意：parent_child.index_documents 接收原始文档 docs（其内部自行切分 parent/child）。
    """
    if not chain.parent_child_retriever:
        return
    try:
        chain.parent_child_retriever.index_documents(docs)
    except Exception as e:
        logger.warning(f"F6a Parent-Child 索引构建失败: {e}")


def _index_contextual(chain, chunks: List[Document]) -> int:
    """F6a 上下文增强索引（索引时一次性 LLM，零在线增量）。返回入索引块数。"""
    if not get_settings().use_contextual_chunks:
        return 0
    try:
        from app.ingestion.contextual import build_chunk_contexts
        contexts = build_chunk_contexts(chunks)
        chain.indexer.index_documents_contextual(chunks, contexts)
        return len(chunks)
    except Exception as e:
        logger.warning(f"F6a 上下文增强索引失败: {e}")
        return 0


def ingest_file(chain, path: str, chunk_strategy: str = "recursive") -> Tuple[List[Document], List[Document]]:
    """单文件摄入全流程：加载 → 分块 → 主索引 → 增强索引 → BM25 增量。

    Args:
        chain: RAGChain 实例（鸭子类型，见模块 docstring）
        path: 文件路径（routes 负责临时文件与格式校验）
        chunk_strategy: recursive（递归字符分块）| semantic（语义分块）

    Returns:
        (docs, chunks)：原始文档与分块，供端点组装响应。
    """
    docs = load_document(path)
    chunks = smart_chunk(
        docs,
        embeddings=chain.indexer.embeddings if chunk_strategy == "semantic" else None,
        use_semantic=(chunk_strategy == "semantic"),
    )
    chain.indexer.index_documents(chunks)
    _index_parent_child(chain, docs)
    _index_contextual(chain, chunks)
    # #11: 增量更新 BM25 索引（避免全量重建）
    chain.sparse_retriever.add_documents(chunks)
    return docs, chunks


def rebuild_enhanced_indexes(chain) -> Tuple[int, int]:
    """F6a：从已索引分块重建 Parent-Child + 上下文增强索引（补齐历史文档，一次性）。

    上下文增强需对每块调用一次 LLM，文档多时较慢，属一次性管理操作。

    Returns:
        (num_documents, num_contextual_chunks)
    """
    num_docs = chain.rebuild_parent_child_index()
    if not get_settings().use_contextual_chunks:
        return num_docs, 0
    return num_docs, _index_contextual(chain, chain.indexer.get_all_chunks())
