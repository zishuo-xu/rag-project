"""RRF 融合策略 - Reciprocal Rank Fusion 合并多路召回结果"""

import logging
from typing import List, Dict

from langchain_core.documents import Document

from config import get_settings

logger = logging.getLogger(__name__)


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


def weighted_fusion(
    result_lists: List[List[Document]],
    weights: List[float] | None = None,
) -> List[Document]:
    """
    加权融合 - 对不同检索路的结果赋予不同权重。

    适用于某些检索路更可靠的场景（如稠密检索通常比 BM25 更准）。

    Args:
        result_lists: 多路检索结果
        weights: 各路权重，默认均匀

    Returns:
        融合后的文档列表
    """
    if weights is None:
        weights = [1.0 / len(result_lists)] * len(result_lists)

    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for weight, result_list in zip(weights, result_lists):
        for rank, doc in enumerate(result_list, start=1):
            doc_key = str(doc.metadata.get("chunk_id", hash(doc.page_content)))
            # 使用倒数排名作为基础分数，乘以权重
            score = weight * (1.0 / rank)
            scores[doc_key] = scores.get(doc_key, 0) + score
            doc_map[doc_key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["fusion_score"] = scores[key]
        results.append(doc)

    return results
