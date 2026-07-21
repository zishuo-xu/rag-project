"""RAG Chain - 组装完整的检索增强生成管道"""

import logging
import time
from typing import List, Optional, Generator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config import get_settings
from app.ingestion.indexer import HierarchicalIndexer
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.graph_retriever import GraphRetriever
from app.generation.prompts import RAG_SIMPLE_PROMPT, RAG_CHAT_PROMPT, FALLBACK_RESPONSE
from app.observability.tracing import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果封装"""
    documents: List[Document]
    dense_results: List[Document] = field(default_factory=list)
    sparse_results: List[Document] = field(default_factory=list)
    graph_results: List[Document] = field(default_factory=list)
    fused_results: List[Document] = field(default_factory=list)
    reranked_results: List[Document] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    retrieval_time_ms: float = 0


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    sources: List[Document]
    retrieval_result: RetrievalResult
    total_time_ms: float = 0


class RAGChain:
    """
    RAG 完整管道：
    Query Transform -> Multi-Recall -> RRF Fusion -> Rerank -> Generate

    支持：
    - 多路召回（Dense + Sparse）
    - 查询改写（Multi-Query / HyDE）
    - RRF 融合
    - Cross-Encoder 重排序
    - 流式输出
    - 对话历史
    """

    def __init__(
        self,
        indexer: HierarchicalIndexer | None = None,
        use_query_transform: bool = True,
        use_rerank: bool = True,
        use_graph: bool = True,
        query_strategy: str = "multi_query",
    ):
        settings = get_settings()

        # 初始化各组件
        self.indexer = indexer or HierarchicalIndexer()
        self.dense_retriever = DenseRetriever(self.indexer)
        self.sparse_retriever = SparseRetriever(self.indexer)
        self.reranker = Reranker() if use_rerank else None
        self.query_transformer = QueryTransformer() if use_query_transform else None
        self.graph_retriever = GraphRetriever() if (use_graph and settings.graph_enabled) else None

        self.use_query_transform = use_query_transform
        self.use_rerank = use_rerank
        self.use_graph = use_graph and settings.graph_enabled
        self.query_strategy = query_strategy

        # LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            streaming=True,
        )

        logger.info(
            f"RAGChain 初始化: query_transform={use_query_transform}, "
            f"rerank={use_rerank}, strategy={query_strategy}"
        )

    def retrieve(self, question: str, top_k: int | None = None, trace_id: str | None = None) -> RetrievalResult:
        """
        执行完整的检索流程。

        Args:
            question: 用户问题
            top_k: 最终返回文档数
            trace_id: 可选的追踪 ID，传入时记录各阶段耗时

        Returns:
            RetrievalResult 包含各阶段结果
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        start_time = time.time()
        tracer = get_tracer()

        result = RetrievalResult(documents=[])

        # Step 1: 查询改写
        if trace_id:
            tracer.start_span(trace_id, "query_transform")
        if self.use_query_transform and self.query_transformer:
            queries = self.query_transformer.transform(question, self.query_strategy)
        else:
            queries = [question]
        result.queries_used = queries
        if trace_id:
            tracer.end_span(trace_id, "query_transform", {
                "strategy": self.query_strategy if self.use_query_transform else "none",
                "num_queries": len(queries),
                "queries": queries[:4],
            })

        # Step 2: 多路召回（对每个查询变体）
        all_dense_results: List[Document] = []
        all_sparse_results: List[Document] = []

        if trace_id:
            tracer.start_span(trace_id, "dense_retrieval")
        for q in queries:
            dense_docs = self.dense_retriever.retrieve(q, top_k=settings.rerank_top_n)
            all_dense_results.extend(dense_docs)
        if trace_id:
            tracer.end_span(trace_id, "dense_retrieval", {"hits": len(all_dense_results)})

        if trace_id:
            tracer.start_span(trace_id, "sparse_retrieval")
        for q in queries:
            sparse_docs = self.sparse_retriever.retrieve(q, top_k=settings.rerank_top_n)
            all_sparse_results.extend(sparse_docs)
        if trace_id:
            tracer.end_span(trace_id, "sparse_retrieval", {"hits": len(all_sparse_results)})

        # 去重
        result.dense_results = self._deduplicate(all_dense_results)
        result.sparse_results = self._deduplicate(all_sparse_results)

        # Step 2.5: 图检索（知识图谱关系补充）
        if trace_id:
            tracer.start_span(trace_id, "graph_retrieval")
        if self.use_graph and self.graph_retriever:
            result.graph_results = self.graph_retriever.retrieve(question, top_k=3)
        if trace_id:
            tracer.end_span(trace_id, "graph_retrieval", {
                "enabled": self.use_graph,
                "hits": len(result.graph_results),
            })

        # Step 3: RRF 融合
        if trace_id:
            tracer.start_span(trace_id, "rrf_fusion")
        fusion_inputs = [result.dense_results, result.sparse_results]
        if result.graph_results:
            fusion_inputs.append(result.graph_results)
        result.fused_results = reciprocal_rank_fusion(fusion_inputs)
        if trace_id:
            tracer.end_span(trace_id, "rrf_fusion", {"fused": len(result.fused_results)})

        # Step 4: Rerank 重排序
        if trace_id:
            tracer.start_span(trace_id, "rerank")
        if self.use_rerank and self.reranker:
            result.reranked_results = self.reranker.rerank(
                question, result.fused_results, top_k=top_k
            )
            result.documents = result.reranked_results
        else:
            result.documents = result.fused_results[:top_k]
        if trace_id:
            tracer.end_span(trace_id, "rerank", {
                "enabled": self.use_rerank,
                "final": len(result.documents),
            })

        result.retrieval_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"检索完成: {result.retrieval_time_ms:.0f}ms, "
            f"dense={len(result.dense_results)}, sparse={len(result.sparse_results)}, "
            f"fused={len(result.fused_results)}, final={len(result.documents)}"
        )
        return result

    def generate(
        self,
        question: str,
        documents: List[Document],
        chat_history: List | None = None,
    ) -> str:
        """
        基于检索结果生成回答。

        Args:
            question: 用户问题
            documents: 检索到的参考文档
            chat_history: 对话历史

        Returns:
            生成的回答文本
        """
        if not documents:
            return FALLBACK_RESPONSE

        # 构建上下文
        context = self._format_context(documents)

        # 选择 prompt 模板
        if chat_history:
            chain = RAG_CHAT_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": question,
                "chat_history": chat_history,
            })
        else:
            chain = RAG_SIMPLE_PROMPT | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "question": question,
            })

        return response

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

        if chat_history:
            chain = RAG_CHAT_PROMPT | self.llm | StrOutputParser()
            for chunk in chain.stream({
                "context": context,
                "question": question,
                "chat_history": chat_history,
            }):
                yield chunk
        else:
            chain = RAG_SIMPLE_PROMPT | self.llm | StrOutputParser()
            for chunk in chain.stream({
                "context": context,
                "question": question,
            }):
                yield chunk

    def invoke(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """
        完整 RAG 调用：检索 + 生成（自动记录 Trace）。

        Args:
            question: 用户问题
            chat_history: 对话历史
            top_k: 检索文档数

        Returns:
            RAGResponse 包含回答、来源和检索详情
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        # 检索
        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)

        # 生成
        tracer.start_span(trace_id, "generation")
        answer = self.generate(question, retrieval_result.documents, chat_history)
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(answer),
            "num_sources": len(retrieval_result.documents),
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=answer)

        return RAGResponse(
            answer=answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
        )

    def invoke_stream(
        self,
        question: str,
        chat_history: List | None = None,
        top_k: int | None = None,
    ) -> Generator[dict, None, None]:
        """
        流式 RAG 调用（自动记录 Trace）。

        Yields:
            {"type": "retrieval", "data": RetrievalResult}
            {"type": "token", "data": str}
            {"type": "done", "data": RAGResponse}
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        # 检索阶段
        retrieval_result = self.retrieve(question, top_k=top_k, trace_id=trace_id)
        yield {"type": "retrieval", "data": retrieval_result}

        # 流式生成
        tracer.start_span(trace_id, "generation")
        full_answer = ""
        for token in self.generate_stream(question, retrieval_result.documents, chat_history):
            full_answer += token
            yield {"type": "token", "data": token}
        tracer.end_span(trace_id, "generation", {
            "answer_chars": len(full_answer),
            "num_sources": len(retrieval_result.documents),
        })

        total_time = (time.time() - start_time) * 1000
        tracer.end_trace(trace_id, answer_preview=full_answer)
        response = RAGResponse(
            answer=full_answer,
            sources=retrieval_result.documents,
            retrieval_result=retrieval_result,
            total_time_ms=total_time,
        )
        yield {"type": "done", "data": response}

    def _format_context(self, documents: List[Document]) -> str:
        """格式化检索文档为上下文字符串"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(
                f"[文档 {i}] (来源: {source})\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """基于 chunk_id 去重"""
        seen = set()
        unique = []
        for doc in documents:
            key = doc.metadata.get("chunk_id", id(doc))
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique
