"""RAG Chain - 组装完整的检索增强生成管道"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Generator
from dataclasses import dataclass, field

import numpy as np
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
from app.retrieval.parent_child import ParentChildRetriever
from app.retrieval.crag import CRAGEvaluator
from app.retrieval.cache import get_semantic_cache
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
    crag_grade: str = ""  # correct / ambiguous / incorrect
    crag_action: str = ""  # 采取的动作描述


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    sources: List[Document]
    retrieval_result: RetrievalResult
    total_time_ms: float = 0
    cache_hit: bool = False


class RAGChain:
    """
    RAG 完整管道：
    [Cache Check] -> [CRAG: Should Retrieve?] -> Query Transform -> Multi-Recall
    -> RRF Fusion -> Rerank -> [CRAG: Evaluate] -> (if incorrect: HyDE Re-retrieve)
    -> Generate -> [Cache Put]

    支持：
    - 多路召回（Dense + Sparse + Graph + Parent-Child）
    - 查询改写（Multi-Query / HyDE）
    - RRF 融合
    - Cross-Encoder 重排序
    - CRAG 自纠正检索
    - 语义缓存
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
        # #5: 缓存 settings 到实例变量，避免高频调用 get_settings()
        self._settings = get_settings()
        settings = self._settings

        # 初始化各组件
        self.indexer = indexer or HierarchicalIndexer()
        self.dense_retriever = DenseRetriever(self.indexer)
        self.sparse_retriever = SparseRetriever(self.indexer)
        self.reranker = Reranker() if use_rerank else None
        self.query_transformer = QueryTransformer() if use_query_transform else None
        self.graph_retriever = GraphRetriever() if (use_graph and settings.graph_enabled) else None

        # Parent-Child 检索器
        self.parent_child_retriever: ParentChildRetriever | None = None
        if settings.use_parent_child:
            try:
                self.parent_child_retriever = ParentChildRetriever(embeddings=self.indexer.embeddings)
            except Exception as e:
                logger.warning(f"Parent-Child 初始化失败: {e}")

        # CRAG 评估器
        self.crag_evaluator: CRAGEvaluator | None = None
        if settings.use_crag:
            self.crag_evaluator = CRAGEvaluator()

        # 语义缓存
        self.semantic_cache = get_semantic_cache() if settings.cache_enabled else None

        self.use_query_transform = use_query_transform
        self.use_rerank = use_rerank
        self.use_graph = use_graph and settings.graph_enabled
        self.query_strategy = query_strategy

        # #17: LLM 添加超时和重试
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            streaming=True,
            request_timeout=60,
            max_retries=2,
        )

        logger.info(
            f"RAGChain 初始化: query_transform={use_query_transform}, "
            f"rerank={use_rerank}, strategy={query_strategy}, "
            f"parent_child={self.parent_child_retriever is not None}, "
            f"crag={self.crag_evaluator is not None}, "
            f"cache={self.semantic_cache is not None}"
        )

    def retrieve(self, question: str, top_k: int | None = None, trace_id: str | None = None) -> RetrievalResult:
        """
        执行完整的检索流程（含 Parent-Child + CRAG）。

        Args:
            question: 用户问题
            top_k: 最终返回文档数
            trace_id: 可选的追踪 ID，传入时记录各阶段耗时

        Returns:
            RetrievalResult 包含各阶段结果
        """
        settings = self._settings
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

        # Step 2: 多路召回（并行执行 Dense + Sparse + Graph + Parent-Child）
        all_dense_results: List[Document] = []
        all_sparse_results: List[Document] = []

        if trace_id:
            tracer.start_span(trace_id, "multi_recall")

        # #19: 批量 Embedding - 一次计算所有查询的 embedding
        query_embeddings = None
        if len(queries) > 1:
            try:
                query_embeddings = self.indexer.embeddings.embed_documents(queries)
            except Exception as e:
                logger.debug(f"批量 embedding 失败，回退到逐个计算: {e}")

        def _dense_search(q: str, idx: int) -> List[Document]:
            return self.dense_retriever.retrieve(q, top_k=settings.rerank_top_n)

        def _sparse_search(q: str) -> List[Document]:
            return self.sparse_retriever.retrieve(q, top_k=settings.rerank_top_n)

        def _graph_search() -> List[Document]:
            if self.use_graph and self.graph_retriever:
                return self.graph_retriever.retrieve(question, top_k=3)
            return []

        def _pc_search() -> List[Document]:
            if self.parent_child_retriever and self.parent_child_retriever.has_index():
                try:
                    return self.parent_child_retriever.retrieve(question, top_k=3)
                except Exception as e:
                    logger.warning(f"Parent-Child 检索失败: {e}")
            return []

        # #1: 并行执行所有召回路径
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交 Dense 和 Sparse 的每个查询变体
            dense_futures = {executor.submit(_dense_search, q, i): i for i, q in enumerate(queries)}
            sparse_futures = {executor.submit(_sparse_search, q): q for q in queries}
            graph_future = executor.submit(_graph_search)
            pc_future = executor.submit(_pc_search)

            for future in as_completed(dense_futures):
                try:
                    all_dense_results.extend(future.result())
                except Exception as e:
                    logger.warning(f"Dense 检索失败: {e}")

            for future in as_completed(sparse_futures):
                try:
                    all_sparse_results.extend(future.result())
                except Exception as e:
                    logger.warning(f"Sparse 检索失败: {e}")

            try:
                result.graph_results = graph_future.result()
            except Exception as e:
                logger.warning(f"Graph 检索失败: {e}")
                result.graph_results = []

            try:
                pc_results = pc_future.result()
            except Exception as e:
                logger.warning(f"Parent-Child 检索失败: {e}")
                pc_results = []

        if trace_id:
            tracer.end_span(trace_id, "multi_recall", {
                "dense_hits": len(all_dense_results),
                "sparse_hits": len(all_sparse_results),
                "graph_hits": len(result.graph_results),
                "pc_hits": len(pc_results),
            })

        # 去重
        result.dense_results = self._deduplicate(all_dense_results)
        result.sparse_results = self._deduplicate(all_sparse_results)

        # Step 3: RRF 融合
        if trace_id:
            tracer.start_span(trace_id, "rrf_fusion")
        fusion_inputs = [result.dense_results, result.sparse_results]
        if result.graph_results:
            fusion_inputs.append(result.graph_results)
        if pc_results:
            fusion_inputs.append(pc_results)
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

        # Step 5: CRAG 自纠正评估
        if trace_id:
            tracer.start_span(trace_id, "crag_evaluation")
        if self.crag_evaluator and result.documents:
            grade, relevant_indices, reason = self.crag_evaluator.evaluate_relevance(
                question, result.documents
            )
            result.crag_grade = grade

            if grade == "incorrect":
                # 补救：用 HyDE 策略重新检索
                result.crag_action = "HyDE 重检索"
                logger.info("CRAG: 检索结果不相关，触发 HyDE 重检索")
                if self.query_transformer:
                    hyde_queries = self.query_transformer.transform(question, "hyde")
                    retry_dense = []
                    # #1: HyDE 重检索也并行化
                    with ThreadPoolExecutor(max_workers=len(hyde_queries)) as executor:
                        retry_futures = {
                            executor.submit(self.dense_retriever.retrieve, q, top_k): q
                            for q in hyde_queries
                        }
                        for future in as_completed(retry_futures):
                            try:
                                retry_dense.extend(future.result())
                            except Exception as e:
                                logger.warning(f"HyDE 重检索失败: {e}")
                    if retry_dense:
                        result.documents = self._deduplicate(retry_dense)[:top_k]
                        result.crag_grade = "recovered"
                else:
                    result.crag_action = "无可用补救策略"
            elif grade == "ambiguous":
                # 保留相关文档
                result.crag_action = "过滤不相关文档"
                result.documents = self.crag_evaluator.filter_relevant_docs(
                    result.documents, relevant_indices
                )
            else:
                result.crag_action = "直接使用"

        if trace_id:
            tracer.end_span(trace_id, "crag_evaluation", {
                "enabled": self.crag_evaluator is not None,
                "grade": result.crag_grade,
                "action": result.crag_action,
            })

        result.retrieval_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"检索完成: {result.retrieval_time_ms:.0f}ms, "
            f"dense={len(result.dense_results)}, sparse={len(result.sparse_results)}, "
            f"fused={len(result.fused_results)}, final={len(result.documents)}, "
            f"crag={result.crag_grade or 'off'}"
        )
        return result

    def compress_context(
        self,
        question: str,
        documents: List[Document],
        max_sentences_per_doc: int = 3,
    ) -> List[Document]:
        """
        上下文压缩：抽取每个文档中与问题最相关的句子，减少噪声。

        策略：基于关键词重叠度排序句子，保留 top-N（无LLM调用，纯算法）。

        Args:
            question: 用户问题
            documents: 检索到的文档
            max_sentences_per_doc: 每个文档最多保留的句子数

        Returns:
            压缩后的文档列表
        """
        import re as _re
        # 提取问题关键词
        q_tokens = set(_re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+', question.lower()))
        if not q_tokens:
            return documents

        compressed = []
        for doc in documents:
            # 按句子切分
            sentences = _re.split(r'(?<=[。！？.!?\n])', doc.page_content)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= max_sentences_per_doc:
                compressed.append(doc)
                continue

            # 计算每个句子与问题的关键词重叠度
            scored = []
            for sent in sentences:
                s_tokens = set(_re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+', sent.lower()))
                overlap = len(q_tokens & s_tokens)
                scored.append((overlap, sent))

            # 保留重叠度最高的句子（保持原始顺序）
            scored.sort(key=lambda x: -x[0])
            top_sentences = set(s for _, s in scored[:max_sentences_per_doc])
            kept = [s for s in sentences if s in top_sentences]

            if kept:
                new_doc = Document(
                    page_content="".join(kept),
                    metadata=doc.metadata.copy(),
                )
                compressed.append(new_doc)
            else:
                compressed.append(doc)

        return compressed

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

        # 上下文压缩：抽取相关句子减少噪声
        documents = self.compress_context(question, documents)

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
        完整 RAG 调用：[缓存检查] + 检索 + 生成 + [缓存写入]（自动记录 Trace）。
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        # #2: 只计算一次 Embedding，缓存检查和写入复用
        q_embedding = None

        # 语义缓存检查
        if self.semantic_cache and not chat_history:
            try:
                q_embedding = np.array(self.indexer.embeddings.embed_query(question))
                cached = self.semantic_cache.get(q_embedding)
                if cached:
                    tracer.start_span(trace_id, "cache_hit")
                    tracer.end_span(trace_id, "cache_hit", {"question": cached.question[:50]})
                    total_time = (time.time() - start_time) * 1000
                    tracer.end_trace(trace_id, answer_preview=cached.answer)
                    # 重建 sources 为 Document
                    source_docs = [
                        Document(page_content=s.get("content", ""), metadata={"source": s.get("source", "")})
                        for s in cached.sources
                    ]
                    return RAGResponse(
                        answer=cached.answer,
                        sources=source_docs,
                        retrieval_result=RetrievalResult(documents=source_docs),
                        total_time_ms=total_time,
                        cache_hit=True,
                    )
            except Exception as e:
                logger.warning(f"缓存检查失败: {e}")

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

        # 写入缓存（复用已计算的 embedding）
        if self.semantic_cache and not chat_history:
            try:
                if q_embedding is None:
                    q_embedding = np.array(self.indexer.embeddings.embed_query(question))
                sources_data = [
                    {"content": doc.page_content[:200], "source": doc.metadata.get("source", "")}
                    for doc in retrieval_result.documents
                ]
                self.semantic_cache.put(question, q_embedding, answer, sources_data)
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")

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
        流式 RAG 调用（含缓存检查 + 自动记录 Trace）。

        Yields:
            {"type": "cache_hit", "data": RAGResponse}  -- 缓存命中时
            {"type": "retrieving", "data": str}         -- 检索进度提示
            {"type": "retrieval", "data": RetrievalResult}
            {"type": "token", "data": str}
            {"type": "done", "data": RAGResponse}
        """
        tracer = get_tracer()
        trace_id = tracer.start_trace(question)
        start_time = time.time()

        # #2: 只计算一次 Embedding
        q_embedding = None

        # 语义缓存检查
        if self.semantic_cache and not chat_history:
            try:
                q_embedding = np.array(self.indexer.embeddings.embed_query(question))
                cached = self.semantic_cache.get(q_embedding)
                if cached:
                    tracer.start_span(trace_id, "cache_hit")
                    tracer.end_span(trace_id, "cache_hit", {"question": cached.question[:50]})
                    total_time = (time.time() - start_time) * 1000
                    tracer.end_trace(trace_id, answer_preview=cached.answer)
                    source_docs = [
                        Document(page_content=s.get("content", ""), metadata={"source": s.get("source", "")})
                        for s in cached.sources
                    ]
                    response = RAGResponse(
                        answer=cached.answer,
                        sources=source_docs,
                        retrieval_result=RetrievalResult(documents=source_docs),
                        total_time_ms=total_time,
                        cache_hit=True,
                    )
                    yield {"type": "cache_hit", "data": response}
                    yield {"type": "done", "data": response}
                    return
            except Exception as e:
                logger.warning(f"缓存检查失败: {e}")

        # #14: 发送检索开始事件
        yield {"type": "retrieving", "data": "正在检索相关文档..."}

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

        # 写入缓存（复用已计算的 embedding）
        if self.semantic_cache and not chat_history:
            try:
                if q_embedding is None:
                    q_embedding = np.array(self.indexer.embeddings.embed_query(question))
                sources_data = [
                    {"content": doc.page_content[:200], "source": doc.metadata.get("source", "")}
                    for doc in retrieval_result.documents
                ]
                self.semantic_cache.put(question, q_embedding, full_answer, sources_data)
            except Exception as e:
                logger.warning(f"缓存写入失败: {e}")

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
