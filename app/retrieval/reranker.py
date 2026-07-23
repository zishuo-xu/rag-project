"""Cross-Encoder 重排序 - 对融合后的候选进行精排"""

import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import get_settings
from app.retrieval.caches import get_rerank_cache

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder 重排序器。

    原理：将 (query, document) 对输入 Cross-Encoder 模型，
    输出相关性分数。相比 Bi-Encoder（双塔模型），
    Cross-Encoder 能捕获 query 和 document 之间的细粒度交互，
    精度更高但速度较慢，适合对少量候选做精排。

    使用模型: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        model_name = model_name or settings.rerank_model
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        # F9 L2 Rerank 缓存：相同 (query, 文档集合) 跳过 cross-encoder
        self.cache = get_rerank_cache() if settings.use_rerank_cache else None
        logger.info(f"Reranker 初始化完成: {model_name}")

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """文档唯一键：优先 chunk_id，缺失时用内容哈希兜底。"""
        cid = doc.metadata.get("chunk_id")
        if cid:
            return str(cid)
        return f"h{hash(doc.page_content)}"

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[Document]:
        """
        对候选文档进行重排序。

        Args:
            query: 查询文本
            documents: 候选文档列表（来自融合阶段）
            top_k: 返回 top-K 结果，默认从配置读取

        Returns:
            重排后的文档列表（按相关性降序）
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k

        if not documents:
            return []

        # F9 L2 缓存命中：跳过 cross-encoder，直接按缓存分排序
        doc_keys = [self._doc_key(d) for d in documents]
        if self.cache is not None:
            cached = self.cache.get_scores(query, doc_keys)
            if cached is not None:
                for doc, key in zip(documents, doc_keys):
                    doc.metadata["rerank_score"] = float(cached.get(key, 0.0))
                ranked = sorted(
                    documents, key=lambda d: d.metadata["rerank_score"], reverse=True
                )
                logger.info(
                    f"重排序缓存命中: {len(documents)} 候选 -> top-{top_k}"
                )
                return ranked[:top_k]

        # 构建 (query, document) 对
        pairs = [(query, doc.page_content) for doc in documents]

        # Cross-Encoder 打分
        scores = self.model.predict(pairs)

        # 将分数附加到文档元数据
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)

        # F9 L2 缓存写入
        if self.cache is not None:
            self.cache.put_scores(
                query, doc_keys,
                {k: float(s) for k, s in zip(doc_keys, scores)},
            )

        # 按分数降序排序
        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [doc for doc, _ in ranked_docs[:top_k]]

        logger.info(
            f"重排序完成: {len(documents)} 候选 -> top-{top_k} "
            f"(最高分: {scores.max():.4f}, 最低分: {scores.min():.4f})"
        )
        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int | None = None,
    ) -> List[dict]:
        """
        带分数的重排序。

        Returns:
            [{"document": Document, "score": float}, ...]
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"document": doc, "score": float(score)}
            for doc, score in ranked[:top_k]
        ]
