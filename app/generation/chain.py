"""RAG Chain - 编排层：缓存 + 检索管道委托 + 生成"""

import logging
import time
from dataclasses import dataclass
from typing import List, Generator

import numpy as np
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config import get_settings, get_llm_extra_body
from app.ingestion.indexer import HierarchicalIndexer
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from app.retrieval.reranker import Reranker
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.parent_child import ParentChildRetriever
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.router import QueryRouter
from app.retrieval.cache import get_semantic_cache
from app.generation.faithfulness import FaithfulnessChecker
from app.generation.prompts import (
    RAG_SIMPLE_PROMPT, RAG_CHAT_PROMPT, DIRECT_ANSWER_PROMPT, FALLBACK_RESPONSE,
    STRICT_RAG_PROMPT, STRICT_RAG_CHAT_PROMPT,
)
from app.observability.tracing import get_tracer

logger = logging.getLogger(__name__)

__all__ = ["RAGChain", "RAGResponse", "RetrievalResult"]


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    sources: List[Document]
    retrieval_result: RetrievalResult
    total_time_ms: float = 0
    cache_hit: bool = False
    # F3 生成忠实度自检观测字段
    faithful: bool | None = None      # True=忠实 / False=含幻觉 / None=未检查或未知
    faithfulness_score: float = 0.0   # 被支撑论断占比 [0,1]
    regenerated: bool = False         # 是否因不忠实触发了严格重生成


class RAGChain:
    """
    RAG 编排层：
    [Cache Check] -> RetrievalPipeline.run() -> Generate -> [Cache Put]

    检索全链路（门控/改写/召回/融合/重排/评估/补救）由 RetrievalPipeline 承担。
    """

    def __init__(
        self,
        indexer: HierarchicalIndexer | None = None,
        use_query_transform: bool = True,
        use_rerank: bool = True,
        use_graph: bool = True,
        query_strategy: str = "multi_query",
    ):
        self._settings = get_settings()
        settings = self._settings

        self.indexer = indexer or HierarchicalIndexer()
        self.dense_retriever = DenseRetriever(self.indexer)
        self.sparse_retriever = SparseRetriever(self.indexer)
        self.reranker = Reranker() if use_rerank else None
        self.query_transformer = QueryTransformer() if use_query_transform else None
        self.graph_retriever = (
            GraphRetriever() if (use_graph and settings.graph_enabled) else None
        )

        self.parent_child_retriever: ParentChildRetriever | None = None
        if settings.use_parent_child:
            try:
                self.parent_child_retriever = ParentChildRetriever(
                    embeddings=self.indexer.embeddings
                )
            except Exception as e:
                logger.warning(f"Parent-Child 初始化失败: {e}")

        self.crag_evaluator: CRAGEvaluator | None = None
        if settings.use_crag:
            self.crag_evaluator = CRAGEvaluator()

        self.query_router: QueryRouter | None = (
            QueryRouter() if settings.use_query_router else None
        )

        # F3 生成忠实度自检（幻觉检测）
        self.faithfulness_checker: FaithfulnessChecker | None = (
            FaithfulnessChecker() if settings.use_faithfulness_check else None
        )

        self.semantic_cache = get_semantic_cache() if settings.cache_enabled else None

        self.use_query_transform = use_query_transform
        self.use_rerank = use_rerank
        self.use_graph = use_graph and settings.graph_enabled
        self.query_strategy = query_strategy

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            streaming=True,
            request_timeout=60,
            max_retries=2,
            extra_body=get_llm_extra_body(),
        )

        # 检索管道（七阶段，可独立测试）
        self.pipeline = RetrievalPipeline(
            self.indexer,
            self.dense_retriever,
            self.sparse_retriever,
            reranker=self.reranker,
            query_transformer=self.query_transformer,
            graph_retriever=self.graph_retriever,
            parent_child_retriever=self.parent_child_retriever,
            crag_evaluator=self.crag_evaluator,
            query_router=self.query_router,
        )

        logger.info(
            f"RAGChain 初始化: query_transform={use_query_transform}, "
            f"rerank={use_rerank}, strategy={query_strategy}, "
            f"parent_child={self.parent_child_retriever is not None}, "
            f"crag={self.crag_evaluator is not None}, "
            f"cache={self.semantic_cache is not None}"
        )

    def rebuild_parent_child_index(self) -> int:
        """F6a：从已索引分块重建 Parent-Child 索引（一次性，用于补齐历史文档）。

        按 doc_id 把分块拼回文档级文本，再交给 ParentChildRetriever 重新切分 parent/child。
        无 parent_child_retriever 或任何异常都返回 0（优雅降级）。
        """
        if not self.parent_child_retriever:
            return 0
        try:
            all_chunks = self.indexer.get_all_chunks()
            grouped: dict[str, Document] = {}
            for ch in all_chunks:
                doc_id = ch.metadata.get("doc_id", "unknown")
                if doc_id not in grouped:
                    grouped[doc_id] = Document(
                        page_content="",
                        metadata={"doc_id": doc_id, "source": ch.metadata.get("source", "")},
                    )
                grouped[doc_id].page_content += "\n" + ch.page_content
            docs = list(grouped.values())
            if docs:
                self.parent_child_retriever.index_documents(docs)
            logger.info(f"F6a Parent-Child 重建: {len(docs)} 个文档")
            return len(docs)
        except Exception as e:
            logger.warning(f"F6a Parent-Child 重建失败: {e}")
            return 0

    def retrieve(
        self, question: str, top_k: int | None = None, trace_id: str | None = None
    ) -> RetrievalResult:
        """执行完整检索流程（委托给 RetrievalPipeline）"""
        return self.pipeline.run(
            question,
            top_k=top_k,
            query_strategy=self.query_strategy,
            use_query_transform=self.use_query_transform,
            use_rerank=self.use_rerank,
            trace_id=trace_id,
        )

    def compress_context(
        self,
        question: str,
        documents: List[Document],
        max_sentences_per_doc: int = 3,
    ) -> List[Document]:
        """上下文压缩：关键词重叠抽句（零LLM调用）"""
        import re as _re
        q_tokens = set(_re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]+', question.lower()))
        if not q_tokens:
            return documents

        compressed = []
        for doc in documents:
            sentences = _re.split(r'(?<=[。！？.!?\n])', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= max_sentences_per_doc:
                compressed.append(doc)
                continue

            scored = []
            for sent in sentences:
                s_tokens = set(_re.findall(r'[一-鿿]{2,}|[a-zA-Z0-9]+', sent.lower()))
                scored.append((len(q_tokens & s_tokens), sent))

            scored.sort(key=lambda x: -x[0])
            top_sentences = set(s for _, s in scored[:max_sentences_per_doc])
            kept = [s for s in sentences if s in top_sentences]

            if kept:
                compressed.append(Document(
                    page_content="".join(kept), metadata=doc.metadata.copy(),
                ))
            else:
                compressed.append(doc)

        return compressed

    def generate(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
        strict: bool = False,
    ) -> str:
        """基于检索结果生成回答。strict=True 时用更严格的 prompt（F3 重生成）。"""
        if not documents:
            return FALLBACK_RESPONSE

        documents = self.compress_context(question, documents)
        context = self._format_context(documents)

        if chat_history:
            template = STRICT_RAG_CHAT_PROMPT if strict else RAG_CHAT_PROMPT
            chain = template | self.llm | StrOutputParser()
            return chain.invoke({
                "context": context, "question": question,
                "chat_history": chat_history,
            })
        template = STRICT_RAG_PROMPT if strict else RAG_SIMPLE_PROMPT
        chain = template | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    def generate_stream(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
    ) -> Generator[str, None, None]:
        """流式生成回答"""
        if not documents:
            yield FALLBACK_RESPONSE
            return

        context = self._format_context(documents)
        template = RAG_CHAT_PROMPT if chat_history else RAG_SIMPLE_PROMPT
        payload = {"context": context, "question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        chain = template | self.llm | StrOutputParser()
        for chunk in chain.stream(payload):
            yield chunk

    def generate_direct(self, question: str, chat_history: List | None = None) -> str:
        """门控判定无需检索时的直接回答（无上下文）"""
        chain = DIRECT_ANSWER_PROMPT | self.llm | StrOutputParser()
        payload = {"question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        return chain.invoke(payload)

    def generate_direct_stream(
        self, question: str, chat_history: List | None = None
    ) -> Generator[str, None, None]:
        """门控跳过时的流式直接回答"""
        chain = DIRECT_ANSWER_PROMPT | self.llm | StrOutputParser()
        payload = {"question": question}
        if chat_history:
            payload["chat_history"] = chat_history
        for chunk in chain.stream(payload):
            yield chunk

    def _generate_faithful(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
    ) -> tuple[str, bool | None, float, bool]:
        """
        F3 生成 + 忠实度自检 + 有界严格重生成（invoke / invoke_stream 共用）。

        流程：生成答案 → 忠实度检查 → 若不忠实则用 strict prompt 重生成，
        最多 faithfulness_max_regen 次（成本兜底）。检查器关闭或异常时放行。

        Returns:
            (answer, faithful, faithfulness_score, regenerated)
        """
        answer = self.generate(question, documents, chat_history)
        if not self.faithfulness_checker:
            return answer, None, 0.0, False

        fb = self.faithfulness_checker.check(question, documents, answer)
        regenerated = False
        regen_left = self._settings.faithfulness_max_regen
        while fb.faithful is False and regen_left > 0:
            logger.info(f"忠实度不足(score={fb.score:.2f})，触发严格重生成")
            answer = self.generate(question, documents, chat_history, strict=True)
            regenerated = True
            regen_left -= 1
            fb = self.faithfulness_checker.check(question, documents, answer)

        return answer, fb.faithful, fb.score, regenerated

    def invoke(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """完整 RAG 调用：[缓存检查] + 检索 + 生成 + [缓存写入]（自动记录 Trace）"""
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            return cached

        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)

        tracer.start_span(trace_id, "generation")
        if retrieval_result.gate_skipped:
            answer = self.generate_direct(question, chat_history)
            faithful, fb_score, regenerated = None, 0.0, False
        elif self.faithfulness_checker and retrieval_result.documents:
            answer, faithful, fb_score, regenerated = self._generate_faithful(
                question, retrieval_result.documents, chat_history
            )
        else:
            # 无文档（召回为空）或自检关闭：直接生成，不做忠实度校验
            answer = self.generate(question, retrieval_result.documents, chat_history)
            faithful, fb_score, regenerated = None, 0.0, False
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
            "faithful": faithful,
            "faithfulness_score": fb_score,
            "regenerated": regenerated,
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=answer)
        self._write_cache(
            question, chat_history, q_embedding, answer, retrieval_result.documents
        )

        return RAGResponse(
            answer=answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
            faithful=faithful,
            faithfulness_score=fb_score,
            regenerated=regenerated,
        )

    def invoke_stream(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        流式 RAG 调用（含缓存检查 + 自动记录 Trace）。

        Yields:
            {"type": "cache_hit", "data": RAGResponse}
            {"type": "retrieving", "data": str}
            {"type": "retrieval", "data": RetrievalResult}
            {"type": "token", "data": str}
            {"type": "done", "data": RAGResponse}
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        cached, q_embedding = self._check_cache(
            question, chat_history, trace_id, start_time
        )
        if cached:
            yield {"type": "cache_hit", "data": cached}
            yield {"type": "done", "data": cached}
            return

        yield {"type": "retrieving", "data": "正在检索相关文档..."}

        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)
        yield {"type": "retrieval", "data": retrieval_result}

        tracer.start_span(trace_id, "generation")
        full_answer = ""
        faithful, fb_score, regenerated = None, 0.0, False
        if retrieval_result.gate_skipped:
            for token in self.generate_direct_stream(question, chat_history):
                full_answer += token
                yield {"type": "token", "data": token}
        elif self.faithfulness_checker and retrieval_result.documents:
            # F3: 先非流式生成+自检，校验通过后再输出（保证用户看到已校验答案）
            full_answer, faithful, fb_score, regenerated = self._generate_faithful(
                question, retrieval_result.documents, chat_history
            )
            yield {"type": "token", "data": full_answer}
        else:
            for token in self.generate_stream(
                question, retrieval_result.documents, chat_history
            ):
                full_answer += token
                yield {"type": "token", "data": token}
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(full_answer),
            "num_sources": len(retrieval_result.documents),
            "gate_skipped": retrieval_result.gate_skipped,
            "faithful": faithful,
            "faithfulness_score": fb_score,
            "regenerated": regenerated,
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=full_answer)
        self._write_cache(
            question, chat_history, q_embedding, full_answer,
            retrieval_result.documents,
        )

        yield {"type": "done", "data": RAGResponse(
            answer=full_answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
            faithful=faithful,
            faithfulness_score=fb_score,
            regenerated=regenerated,
        )}

    # ---- 缓存公共逻辑（invoke / invoke_stream 共用） ----

    def _check_cache(
        self,
        question: str,
        chat_history: List | None,
        trace_id: str,
        start_time: float,
    ) -> tuple[RAGResponse | None, np.ndarray | None]:
        """
        语义缓存检查。

        Returns:
            (命中时的 RAGResponse 或 None, 查询 embedding 或 None)
        """
        tracer = get_tracer()
        q_embedding = None
        if not (self.semantic_cache and not chat_history):
            return None, None
        try:
            q_embedding = np.array(self.indexer.embeddings.embed_query(question))
            cached = self.semantic_cache.get(q_embedding)
            if cached:
                tracer.start_span(trace_id, "cache_hit")
                tracer.end_span(trace_id, "cache_hit", {
                    "question": cached.question[:50],
                })
                total_time = (time.time() - start_time) * 1000
                tracer.end_trace(trace_id, answer_preview=cached.answer)
                source_docs = [
                    Document(
                        page_content=s.get("content", ""),
                        metadata={"source": s.get("source", "")},
                    )
                    for s in cached.sources
                ]
                return RAGResponse(
                    answer=cached.answer,
                    sources=source_docs,
                    retrieval_result=RetrievalResult(documents=source_docs),
                    total_time_ms=total_time,
                    cache_hit=True,
                ), q_embedding
        except Exception as e:
            logger.warning(f"缓存检查失败: {e}")
        return None, q_embedding

    def _write_cache(
        self,
        question: str,
        chat_history: List | None,
        q_embedding: np.ndarray | None,
        answer: str,
        documents: List[Document],
    ) -> None:
        """语义缓存写入（复用已计算的 embedding；带对话历史时不写）"""
        if not self.semantic_cache or chat_history:
            return
        try:
            embedding = q_embedding
            if embedding is None:
                embedding = np.array(self.indexer.embeddings.embed_query(question))
            sources_data = [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", ""),
                }
                for doc in documents
            ]
            self.semantic_cache.put(question, embedding, answer, sources_data)
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")

    def _format_context(self, documents: List[Document]) -> str:
        """格式化检索文档为上下文字符串"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"[文档 {i}] (来源: {source})\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)
