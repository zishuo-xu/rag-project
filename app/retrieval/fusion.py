"""RRF 融合策略 - Reciprocal Rank Fusion 合并多路召回结果 + 文档去重原语"""

import logging
from typing import List, Dict

from langchain_core.documents import Document

from config import get_settings

logger = logging.getLogger(__name__)


def chunk_key(doc: Document):
    """文档唯一标识：chunk_id 优先，缺失时退化为对象 id（仅与自身去重）。

    全管道共用的文档身份定义（召回去重 / 迭代收敛判断 / agent 证据累积）。
    注：RRF 融合内部用内容哈希兜底（跨通道同内容不同对象也要合并计分），
    是融合算法的一部分，不使用本函数。
    """
    return doc.metadata.get("chunk_id", id(doc))


def dedup_by_chunk_id(documents: List[Document]) -> List[Document]:
    """按 chunk_key 去重，保留首次出现（保序）。"""
    seen = set()
    unique = []
    for doc in documents:
        key = chunk_key(doc)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int | None = None,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) 融合多路召回结果。

    公式: RRF_score(d) = Σ 1 / (k + rank_i(d))
    其中 k 为平滑常数（默认60），rank_i 为文档 d 在第 i 路结果中的排名。

    优势：
    - 不依赖各路检索的原始分数（避免分数不可比问题）
    - 仅利用排名信息，简单且鲁棒
    - 多路都命中的文档会获得更高分数

    Args:
        result_lists: 多路检索结果列表，每路按相关性排序
        k: RRF 平滑常数，默认从配置读取

    Returns:
        融合后按 RRF 分数降序排列的文档列表
    """
    settings = get_settings()
    k = k or settings.retrieval_rrf_k

    # 计算每个文档的 RRF 分数
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # 用 chunk_id 或内容哈希作为唯一标识
            doc_key = doc.metadata.get("chunk_id", hash(doc.page_content))
            doc_key = str(doc_key)

            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + 1.0 / (k + rank)
            doc_map[doc_key] = doc  # 保留最新的文档引用

    # 按 RRF 分数降序排序
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = rrf_scores[key]
        results.append(doc)

    logger.info(
        f"RRF 融合: {len(result_lists)} 路结果 -> {len(results)} 个唯一文档 "
        f"(k={k})"
    )
    return results
