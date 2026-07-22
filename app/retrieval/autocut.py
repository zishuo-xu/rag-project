"""Autocut 自适应截断 - Kneedle 膝点检测动态截断重排结果

问题背景：
    固定 TopK 截断（reranker 的 `ranked_docs[:top_k]`）不管第 K+1 名之后分数是否
    断崖式下跌，都塞满 K 篇给 LLM，造成"噪声注入"——无关文档稀释上下文、干扰生成、
    浪费 token。

算法（Kneedle 膝点检测）：
    1. 候选已按 rerank_score 降序排列（reranker 保证）。
    2. 分数 min-max 归一化到 [0,1]（CrossEncoder 原始 logit 可为负/无界，归一化后
       相对阈值才有意义）。
    3. 把每个候选看作曲线上的点 (rank_i, score_i)，找它到"首尾连线"垂直距离最大的点
       ——曲线弯曲最剧烈处即"膝点"，是高相关区与噪声尾的自然分界。
    4. 在膝点处截断，再施加 [min_docs, top_k] 上下界：
       - 下界 min_docs：永不返回过少（避免单点突变切太狠）
       - 上界 top_k：永不超过原 TopK（纯降噪，不扩容）
    5. 曲线平坦（点全在连线上，无膝点）→ 回退固定 top_k。

相比备选方案：
    - 相对阈值：是"固定相对 cutoff"而非检测结构性断点，自适应味淡；
    - 最大间隙：对单点离群突变脆弱；
    - Kneedle：看整体曲率，抗噪、真正自适应、可解释（面试一句话讲清）。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 浮点判平阈值：分数极差或垂直距离小于此值视为"平坦/无膝点"
_EPS = 1e-9
# 并列容差：对称 S 曲线会在拐点两侧产生理论相等的距离，浮点噪声会扰乱取舍。
# 仅当某点距离"显著更大"（超过容差）才更新膝点，使平局稳定取靠前者
# （靠前者=高相关平台末尾，切在噪声尾之前，降噪更彻底）。
_TIE_TOL = 1e-6


def find_knee(scores: List[float]) -> Optional[int]:
    """在降序分数曲线中找膝点（Kneedle，垂直距离法）。

    Args:
        scores: 按降序排列的相关性分数列表。

    Returns:
        膝点索引（0-based）；曲线平坦/无内点/无显著膝点时返回 None。
    """
    n = len(scores)
    if n < 3:  # 少于 3 点无内点，谈不上膝点
        return None

    s_min, s_max = min(scores), max(scores)
    if s_max - s_min < _EPS:  # 全等分（平坦）
        return None

    # min-max 归一化
    rng = s_max - s_min
    ys = [(s - s_min) / rng for s in scores]
    xs = [i / (n - 1) for i in range(n)]

    # 首尾连线的一般式 A*x + B*y + C = 0
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    a = y1 - y0
    b = x0 - x1
    c = x1 * y0 - x0 * y1

    # 内点到连线的垂直距离（分子，分母恒定可省）；严格 > 使并列时取靠前者（更安全）
    best_idx: Optional[int] = None
    best_dist = 0.0
    for i in range(1, n - 1):
        dist = abs(a * xs[i] + b * ys[i] + c)
        if dist > best_dist + _TIE_TOL:  # 容差使浮点平局稳定取靠前者
            best_dist = dist
            best_idx = i

    if best_idx is None or best_dist < _EPS:
        return None  # 点全在连线上（线性），无膝点
    return best_idx


def autocut_truncate(
    documents: List[Document],
    top_k: int,
    min_docs: int = 2,
    score_key: str = "rerank_score",
) -> List[Document]:
    """对已按分数降序的候选做自适应截断。

    Args:
        documents: 已按 score_key 降序排列的候选文档（reranker 输出）。
        top_k: 截断上界（永不超过此数，等价于原固定 TopK）。
        min_docs: 截断下界（至少保留此数，避免切太狠）。
        score_key: 文档 metadata 中的分数字段名。

    Returns:
        截断后的文档列表（新列表，不修改入参）。
    """
    n = len(documents)
    if n == 0:
        return []
    if n <= min_docs:
        return list(documents)

    scores = [float(doc.metadata.get(score_key, 0.0)) for doc in documents]
    knee = find_knee(scores)

    if knee is None:
        keep = min(top_k, n)  # 无膝点 → 回退固定 top_k
    else:
        keep = knee + 1  # 保留到膝点（含）

    # 施加上下界：clamp(keep, min_docs, min(top_k, n))
    keep = max(min_docs, min(keep, top_k, n))

    logger.info(
        f"Autocut: {n} 候选 -> 保留 {keep} "
        f"(膝点={knee}, min_docs={min_docs}, top_k={top_k})"
    )
    return list(documents[:keep])
