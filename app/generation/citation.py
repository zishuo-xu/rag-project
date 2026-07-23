"""F7 引用溯源与答案定位 - 把答案句级 claim 关联到最相关源文档块（零在线 LLM）

生产级 RAG 必须让答案可验证。本模块在生成后把答案切成句子级 claim，
用 embedding 余弦相似度把每个 claim 关联到最相关的源文档块，输出结构化引用
（含来源、块 id、置信度、证据片段），供前端高亮与人工核验。

时延：零 LLM，仅一次批量 embedding（claim 数受 citation_max_claims 上限约束）。
异常：任何失败返回空列表，不阻断主链路（与 F1-F6 一致的优雅降级）。
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document

from config import get_settings

logger = logging.getLogger(__name__)

# 来源标注（如 [来源: xxx]、[文档 1]）与纯数字编号，切 claim 时剔除
_SOURCE_TAG = re.compile(r"\[(?:来源|文档|source|doc)[^\]]*\]", re.IGNORECASE)
# 句子边界
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?\n])")


@dataclass
class Citation:
    """单条引用：答案中的一个 claim 关联到的源文档块。"""
    claim: str
    source: str
    chunk_id: str
    doc_index: int          # 在送入 LLM 的文档列表中的序号（1-based）
    confidence: float       # claim-块余弦相似度 [0,1]
    snippet: str = ""       # 源块中与 claim 最相关的证据片段（截断）


def split_claims(answer: str, max_claims: int | None = None) -> List[str]:
    """把答案切成句子级 claim：先剔除来源标注，再按句末标点切分，过滤过短/空句。"""
    settings = get_settings()
    max_claims = max_claims or settings.citation_max_claims
    if not answer or not answer.strip():
        return []
    # 先整体剔除来源标注（避免标注内的 "." 被误当句子边界切碎）
    answer = _SOURCE_TAG.sub(" ", answer)
    claims: List[str] = []
    for raw in _SENT_SPLIT.split(answer):
        # 去首尾空白/项目符号/编号与句末标点
        cleaned = raw.strip()
        cleaned = cleaned.lstrip("-*•0123456789.、) ").strip()
        cleaned = cleaned.rstrip("。！？.!?,，;；:： ").strip()
        if len(cleaned) >= 4:  # 过短（如"是的"）无溯源价值
            claims.append(cleaned)
        if len(claims) >= max_claims:
            break
    return claims


class CitationBuilder:
    """引用构建器：claim 向量 vs 源块向量余弦关联，零在线 LLM。"""

    def __init__(
        self,
        embeddings=None,
        threshold: Optional[float] = None,
        max_claims: Optional[int] = None,
    ):
        settings = get_settings()
        self.embeddings = embeddings
        self.threshold = (
            threshold if threshold is not None else settings.citation_threshold
        )
        self.max_claims = max_claims or settings.citation_max_claims

    def build(
        self, question: str, answer: str, documents: List[Document]
    ) -> List[Citation]:
        """为答案中的每个 claim 关联最相关源块。任何异常返回 []（降级）。"""
        if not self.embeddings or not documents:
            return []
        claims = split_claims(answer, self.max_claims)
        if not claims:
            return []
        try:
            claim_vecs = np.array(self.embeddings.embed_documents(claims))
            doc_texts = [d.page_content[:400] for d in documents]
            doc_vecs = np.array(self.embeddings.embed_documents(doc_texts))
            claim_vecs = _l2_normalize(claim_vecs)
            doc_vecs = _l2_normalize(doc_vecs)
            sim = claim_vecs @ doc_vecs.T  # [num_claims, num_docs]

            citations: List[Citation] = []
            for i, claim in enumerate(claims):
                j = int(np.argmax(sim[i]))
                conf = float(sim[i][j])
                doc = documents[j]
                citations.append(Citation(
                    claim=claim,
                    source=doc.metadata.get("source", ""),
                    chunk_id=str(doc.metadata.get("chunk_id", "")),
                    doc_index=j + 1,
                    confidence=round(conf, 4),
                    snippet=_best_snippet(doc.page_content, claim),
                ))
            logger.info(
                f"F7 引用溯源: {len(claims)} claims -> {len(citations)} 引用 "
                f"(avg conf={np.mean([c.confidence for c in citations]):.3f})"
            )
            return citations
        except Exception as e:
            logger.warning(f"F7 引用溯源失败，降级为空: {e}")
            return []


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行 L2 归一化；零向量保持为零。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _tokens(text: str) -> set:
    """细粒度词元：中文按字二元组（bigram）切分 + 连续英数作为一个词。

    中文整句作为一个 token 粒度过粗（会导致句间无重叠），用 bigram 提升重叠分辨。
    """
    text = text.lower()
    toks = set(re.findall(r"[a-zA-Z0-9]+", text))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            toks.add(run)
        else:
            toks.update(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def _best_snippet(content: str, claim: str, window: int = 80) -> str:
    """在源块中找与 claim 词重叠最多的句子作为证据片段（零 LLM）。"""
    sentences = [s.strip() for s in _SENT_SPLIT.split(content) if s.strip()]
    if not sentences:
        return content[:window]
    claim_tokens = _tokens(claim)
    best, best_score = sentences[0], -1
    for s in sentences:
        score = len(claim_tokens & _tokens(s))
        if score > best_score:
            best, best_score = s, score
    return best[:window]
