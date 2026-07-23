"""FastAPI 路由定义 - Chat / Documents / Evaluation / Health"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    ChatRequest, ChatResponse, SourceDocument, RetrievalDetail, Citation,
    UploadResponse, DocumentListResponse, DocumentInfo,
    EvalRequest, EvalReport, HealthResponse,
)
from config import get_settings
from app.ingestion.loader import load_document
from app.ingestion.chunker import smart_chunk
from app.generation.chain import RAGChain, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# 全局 RAG Chain 实例（在 main.py 中初始化）
_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """获取全局 RAG Chain 实例"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def set_rag_chain(chain: RAGChain):
    """设置全局 RAG Chain 实例"""
    global _rag_chain
    _rag_chain = chain


# 并发闸门（在 main.py lifespan 中初始化）
_concurrency_gate: asyncio.Semaphore | None = None


def get_concurrency_gate() -> asyncio.Semaphore | None:
    """获取全局并发闸门"""
    return _concurrency_gate


def set_concurrency_gate(gate: asyncio.Semaphore | None):
    """设置全局并发闸门"""
    global _concurrency_gate
    _concurrency_gate = gate


async def _acquire_gate(gate: asyncio.Semaphore) -> bool:
    """获取闸门许可，超时返回 False"""
    settings = get_settings()
    try:
        await asyncio.wait_for(
            gate.acquire(), timeout=settings.request_queue_timeout
        )
        return True
    except asyncio.TimeoutError:
        return False


# ============ Chat 端点 ============

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口 - 支持普通和流式两种模式。

    完整流程：查询改写 -> 多路召回 -> RRF融合 -> Rerank -> LLM生成
    并发闸门限制同时处理的请求数，超出排队，排队超时返回 503。
    同步阻塞的 chain 调用通过 asyncio.to_thread 移出事件循环。
    """
    chain = get_rag_chain()

    # 构建对话历史
    chat_history = _build_chat_history(request.chat_history)

    if request.stream:
        return EventSourceResponse(
            _stream_response(chain, request, chat_history)
        )

    # 非流式：闸门内执行完整调用（同步阻塞调用移出事件循环）
    gate = get_concurrency_gate()
    if gate is not None:
        if not await _acquire_gate(gate):
            raise HTTPException(
                status_code=503, detail="服务繁忙，请求排队超时，请稍后重试"
            )
        try:
            response = await asyncio.to_thread(
                chain.invoke,
                question=request.question,
                chat_history=chat_history,
                top_k=request.top_k,
            )
        finally:
            gate.release()
    else:
        response = await asyncio.to_thread(
            chain.invoke,
            question=request.question,
            chat_history=chat_history,
            top_k=request.top_k,
        )

    return _build_chat_response(response)


async def _stream_response(
    chain: RAGChain,
    request: ChatRequest,
    chat_history: list,
) -> AsyncGenerator[dict, None]:
    """SSE 流式响应生成器（闸门在流结束后释放）"""
    gate = get_concurrency_gate()
    if gate is not None:
        if not await _acquire_gate(gate):
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "服务繁忙，请求排队超时，请稍后重试"},
                    ensure_ascii=False,
                ),
            }
            return

    try:
        event_iter = chain.invoke_stream(
            question=request.question,
            chat_history=chat_history,
            top_k=request.top_k,
        )
        while True:
            # 同步生成器逐事件移出事件循环，None 为耗尽哨兵
            event = await asyncio.to_thread(next, event_iter, None)
            if event is None:
                break
            if event["type"] == "cache_hit":
                response: RAGResponse = event["data"]
                sources = [
                    {
                        "content": doc.page_content[:200],
                        "source": doc.metadata.get("source", ""),
                        "score": doc.metadata.get("rerank_score"),
                    }
                    for doc in response.sources
                ]
                yield {
                    "event": "cache_hit",
                    "data": json.dumps({
                        "answer": response.answer,
                        "sources": sources,
                        "total_time_ms": response.total_time_ms,
                    }, ensure_ascii=False),
                }
            elif event["type"] == "retrieval":
                retrieval = event["data"]
                yield {
                    "event": "retrieval",
                    "data": json.dumps({
                        "queries_used": retrieval.queries_used,
                        "dense_count": len(retrieval.dense_results),
                        "sparse_count": len(retrieval.sparse_results),
                        "fused_count": len(retrieval.fused_results),
                        "final_count": len(retrieval.documents),
                        "retrieval_time_ms": retrieval.retrieval_time_ms,
                        "crag_grade": retrieval.crag_grade,
                        "summary_count": len(retrieval.summary_results),
                        "crag_action": retrieval.crag_action,
                    }, ensure_ascii=False),
                }
            elif event["type"] == "token":
                yield {"event": "token", "data": event["data"]}
            elif event["type"] == "correction":
                # F8 投机流式：忠实度不足时的严格重生成答案（前端应替换已显示内容）
                yield {"event": "correction", "data": event["data"]}
            elif event["type"] == "done":
                response: RAGResponse = event["data"]
                sources = [
                    {
                        "content": doc.page_content[:200],
                        "source": doc.metadata.get("source", ""),
                        "score": doc.metadata.get("rerank_score"),
                    }
                    for doc in response.sources
                ]
                citations = [
                    {
                        "claim": c.claim, "source": c.source, "chunk_id": c.chunk_id,
                        "doc_index": c.doc_index, "confidence": c.confidence,
                        "snippet": c.snippet,
                    }
                    for c in (response.citations or [])
                ]
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "sources": sources,
                        "total_time_ms": response.total_time_ms,
                        "cache_hit": response.cache_hit,
                        "citations": citations,
                        "short_answer": response.short_answer,
                        "self_consistency_used": response.self_consistency_used,
                        "rewritten_query": response.rewritten_query,
                        "faithful": response.faithful,
                        "regenerated": response.regenerated,
                    }, ensure_ascii=False),
                }
    finally:
        if gate is not None:
            gate.release()


# ============ Documents 端点 ============

@router.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
):
    """
    上传文档并建立索引。

    支持格式：PDF, TXT, Markdown
    chunk_strategy: recursive（默认，递归字符分块）| semantic（语义分块）
    """
    chain = get_rag_chain()

    if chunk_strategy not in ("recursive", "semantic"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的 chunk_strategy: {chunk_strategy}，支持: recursive, semantic",
        )

    # 保存上传文件到临时目录
    suffix = Path(file.filename).suffix
    if suffix.lower() not in (".pdf", ".txt", ".md", ".markdown"):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持: .pdf, .txt, .md"
        )

    try:
        # 写入临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 加载 -> 分块 -> 索引
        docs = load_document(tmp_path)
        chunks = smart_chunk(
            docs,
            embeddings=chain.indexer.embeddings if chunk_strategy == "semantic" else None,
            use_semantic=(chunk_strategy == "semantic"),
        )
        chain.indexer.index_documents(chunks)

        # F6a: 接线 Parent-Child（小块检索大块返回，修复"答案不在 top 块"）
        # 注意：parent_child.index_documents 接收原始文档 docs（其内部自行切分 parent/child）
        if chain.parent_child_retriever:
            try:
                chain.parent_child_retriever.index_documents(docs)
            except Exception as e:
                logger.warning(f"F6a Parent-Child 索引构建失败: {e}")

        # F6a: 上下文增强索引（索引时一次性 LLM，零在线增量）
        if get_settings().use_contextual_chunks:
            try:
                from app.ingestion.contextual import build_chunk_contexts
                contexts = build_chunk_contexts(chunks)
                chain.indexer.index_documents_contextual(chunks, contexts)
            except Exception as e:
                logger.warning(f"F6a 上下文增强索引失败: {e}")

        # #11: 增量更新 BM25 索引（避免全量重建）
        chain.sparse_retriever.add_documents(chunks)

        return UploadResponse(
            message=f"文档 '{file.filename}' 上传并索引成功",
            filename=file.filename,
            num_chunks=len(chunks),
            doc_id=docs[0].metadata.get("doc_id", "") if docs else "",
        )

    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.post("/api/documents/reindex")
async def reindex_f6a():
    """F6a：从已索引分块重建 Parent-Child + 上下文增强索引（补齐历史文档，一次性）。

    上下文增强需对每块调用一次 LLM，文档多时较慢，属一次性管理操作。
    """
    chain = get_rag_chain()
    num_docs = chain.rebuild_parent_child_index()
    num_ctx = 0
    if get_settings().use_contextual_chunks:
        try:
            from app.ingestion.contextual import build_chunk_contexts
            chunks = chain.indexer.get_all_chunks()
            contexts = build_chunk_contexts(chunks)
            chain.indexer.index_documents_contextual(chunks, contexts)
            num_ctx = len(chunks)
        except Exception as e:
            logger.warning(f"F6a 上下文增强重建失败: {e}")
    return {
        "message": "F6a 索引重建完成",
        "num_documents": num_docs,
        "num_contextual_chunks": num_ctx,
    }


@router.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """获取已索引的文档列表"""
    chain = get_rag_chain()

    try:
        all_chunks = chain.indexer.get_all_chunks()

        # 按 doc_id 分组统计
        doc_stats = {}
        for chunk in all_chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            source = chunk.metadata.get("source", "")
            if doc_id not in doc_stats:
                doc_stats[doc_id] = {"source": source, "count": 0}
            doc_stats[doc_id]["count"] += 1

        documents = [
            DocumentInfo(doc_id=doc_id, source=info["source"], num_chunks=info["count"])
            for doc_id, info in doc_stats.items()
        ]

        return DocumentListResponse(documents=documents, total=len(documents))

    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return DocumentListResponse(documents=[], total=0)


@router.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """
    获取指定文档的全链路索引详情（用于可视化）。

    返回：分块内容 + Embedding 向量 + L1 摘要 + BM25 词项
    """
    chain = get_rag_chain()

    try:
        all_chunks = chain.indexer.get_all_chunks()
        doc_chunks = [
            c for c in all_chunks if c.metadata.get("doc_id") == doc_id
        ]
        doc_chunks.sort(key=lambda x: x.metadata.get("position", 0))

        # 1. 分块内容 + BM25 分词
        chunks_info = []
        for c in doc_chunks:
            tokens = chain.sparse_retriever._tokenize(c.page_content)
            # 词频统计 top-10
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            top_terms = sorted(freq.items(), key=lambda x: -x[1])[:10]
            chunks_info.append({
                "chunk_id": c.metadata.get("chunk_id", ""),
                "position": c.metadata.get("position", 0),
                "content": c.page_content,
                "char_count": len(c.page_content),
                "token_count": len(tokens),
                "top_terms": [{"term": t, "count": n} for t, n in top_terms],
                "source": c.metadata.get("source", ""),
            })

        # 2. Embedding 向量信息
        embeddings_info = chain.indexer.get_chunk_embeddings(doc_id)

        # 3. L1 文档摘要
        summary = chain.indexer.get_document_summary(doc_id)

        # 4. 汇总统计
        total_chars = sum(c["char_count"] for c in chunks_info)
        total_tokens = sum(c["token_count"] for c in chunks_info)

        return {
            "doc_id": doc_id,
            "total": len(chunks_info),
            "stats": {
                "total_chars": total_chars,
                "total_tokens": total_tokens,
                "avg_chunk_chars": round(total_chars / len(chunks_info)) if chunks_info else 0,
                "vector_dim": embeddings_info[0]["vector_dim"] if embeddings_info else 0,
                "embedding_model": get_settings().embedding_model,
                "has_summary": summary is not None,
            },
            "summary": summary,
            "embeddings": embeddings_info,
            "chunks": chunks_info,
        }

    except Exception as e:
        logger.error(f"获取索引详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Retrieval Compare 端点 ============

@router.post("/api/retrieval/compare")
async def retrieval_compare(request: ChatRequest):
    """
    检索策略对比实验。

    同一问题分别用 Dense / Sparse / Hybrid(RRF) / Hybrid+Rerank 四种策略检索，
    返回各策略的命中结果、分数和耗时，用于对比分析。
    """
    chain = get_rag_chain()
    settings = get_settings()
    import time as _time

    question = request.question
    top_k = request.top_k or settings.retrieval_top_k

    # 1. Dense Only
    t0 = _time.time()
    dense_docs = chain.dense_retriever.retrieve(question, top_k=top_k)
    dense_ms = (_time.time() - t0) * 1000

    # 2. Sparse Only (BM25)
    t0 = _time.time()
    sparse_docs = chain.sparse_retriever.retrieve(question, top_k=top_k)
    sparse_ms = (_time.time() - t0) * 1000

    # 3. Hybrid RRF (no rerank)
    t0 = _time.time()
    from app.retrieval.fusion import reciprocal_rank_fusion
    fused_docs = reciprocal_rank_fusion([dense_docs, sparse_docs])[:top_k]
    rrf_ms = (_time.time() - t0) * 1000

    # 4. Hybrid + Rerank
    t0 = _time.time()
    if chain.reranker:
        reranked_docs = chain.reranker.rerank(question, fused_docs, top_k=top_k)
    else:
        reranked_docs = fused_docs
    rerank_ms = (_time.time() - t0) * 1000

    def _fmt(docs):
        return [
            {
                "content": d.page_content[:150],
                "chunk_id": d.metadata.get("chunk_id", ""),
                "source": d.metadata.get("source", ""),
                "score": d.metadata.get("rerank_score") or d.metadata.get("bm25_score"),
            }
            for d in docs
        ]

    # 计算重叠度：各策略与 reranked 的 chunk_id 交集
    rerank_ids = {d.metadata.get("chunk_id") for d in reranked_docs}
    def _overlap(docs):
        ids = {d.metadata.get("chunk_id") for d in docs}
        return len(ids & rerank_ids)

    return {
        "question": question,
        "top_k": top_k,
        "strategies": {
            "dense": {
                "name": "Dense（向量检索）",
                "time_ms": round(dense_ms, 1),
                "count": len(dense_docs),
                "overlap_with_rerank": _overlap(dense_docs),
                "results": _fmt(dense_docs),
            },
            "sparse": {
                "name": "Sparse（BM25 关键词）",
                "time_ms": round(sparse_ms, 1),
                "count": len(sparse_docs),
                "overlap_with_rerank": _overlap(sparse_docs),
                "results": _fmt(sparse_docs),
            },
            "rrf": {
                "name": "Hybrid RRF（融合）",
                "time_ms": round(rrf_ms, 1),
                "count": len(fused_docs),
                "overlap_with_rerank": _overlap(fused_docs),
                "results": _fmt(fused_docs),
            },
            "rerank": {
                "name": "Hybrid + Rerank（最终）",
                "time_ms": round(rerank_ms, 1),
                "count": len(reranked_docs),
                "overlap_with_rerank": len(rerank_ids),
                "results": _fmt(reranked_docs),
            },
        },
    }


# ============ Traces 端点 ============

@router.get("/api/traces")
async def get_traces(limit: int = 20):
    """获取最近的管道追踪记录（瀑布图数据）"""
    from app.observability.tracing import get_tracer
    tracer = get_tracer()
    return {"traces": tracer.get_traces(limit)}


@router.get("/api/traces/stats")
async def get_trace_stats():
    """获取各阶段平均耗时统计"""
    from app.observability.tracing import get_tracer
    tracer = get_tracer()
    return tracer.get_stats()


# ============ Evaluation 端点 ============

@router.post("/api/evaluate", response_model=EvalReport)
async def evaluate(request: EvalRequest):
    """
    触发 RAG 系统评估。

    使用 RAGAS 指标评估检索和生成质量。
    """
    if not request.questions:
        raise HTTPException(status_code=400, detail="请提供测试问题列表")

    try:
        from app.evaluation.metrics import evaluate_rag

        report = evaluate_rag(
            questions=request.questions,
            ground_truths=request.ground_truths or None,
        )
        return report

    except ImportError:
        raise HTTPException(status_code=501, detail="评估模块依赖未安装")
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise HTTPException(status_code=500, detail=f"评估执行失败: {str(e)}")


# ============ Graph RAG 端点 ============

# 图谱构建后台任务引用
_graph_build_task: asyncio.Task | None = None


@router.post("/api/graph/build")
async def build_knowledge_graph():
    """
    启动知识图谱构建（后台异步执行）。

    立即返回，通过 GET /api/graph/build/progress 轮询进度。
    支持增量构建：已处理过的分块自动跳过。
    """
    global _graph_build_task
    from app.ingestion.graph_extractor import get_graph_builder

    builder = get_graph_builder()

    # 如果正在构建，返回当前进度
    if builder.is_building:
        return {
            "message": "图谱正在构建中",
            "status": "building",
            "progress": builder.get_build_progress(),
        }

    chain = get_rag_chain()
    all_chunks = chain.indexer.get_all_chunks()
    if not all_chunks:
        raise HTTPException(status_code=400, detail="无已索引文档，请先上传文档")

    # 启动后台任务（不阻塞事件循环）
    async def _run_build():
        try:
            await builder.build_from_documents_async(all_chunks, incremental=True)
        except Exception as e:
            logger.error(f"知识图谱后台构建失败: {e}")

    _graph_build_task = asyncio.create_task(_run_build())

    return {
        "message": "图谱构建已启动",
        "status": "building",
        "total_chunks": len(all_chunks),
        "progress": builder.get_build_progress(),
    }


@router.get("/api/graph/build/progress")
async def get_graph_build_progress():
    """查询图谱构建进度（前端轮询用）"""
    from app.ingestion.graph_extractor import get_graph_builder
    builder = get_graph_builder()
    progress = builder.get_build_progress()

    # 构建完成时附带统计信息
    if progress["status"] == "completed":
        progress["stats"] = builder.get_stats()

    return progress


@router.get("/api/graph/stats")
async def get_graph_stats():
    """获取知识图谱统计信息"""
    from app.ingestion.graph_extractor import get_graph_builder
    builder = get_graph_builder()
    return builder.get_stats()


@router.get("/api/graph/triples")
async def get_graph_triples(limit: int = 100):
    """获取知识图谱三元组（用于可视化）"""
    from app.ingestion.graph_extractor import get_graph_builder
    builder = get_graph_builder()
    return {
        "triples": builder.get_all_triples(limit=limit),
        "stats": builder.get_stats(),
    }


@router.post("/api/graph/query")
async def query_graph(request: ChatRequest):
    """
    图检索查询 - 返回实体、关系和子图信息。
    """
    chain = get_rag_chain()
    if not chain.graph_retriever:
        raise HTTPException(status_code=400, detail="Graph RAG 未启用")

    try:
        result = chain.graph_retriever.retrieve_with_context(request.question)
        return result
    except Exception as e:
        logger.error(f"图检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/graph/path")
async def find_graph_path(source: str, target: str):
    """
    查找两个实体之间的关系路径。

    例如: /api/graph/path?source=Redis&target=B+树
    """
    from app.ingestion.graph_extractor import get_graph_builder
    from app.retrieval.graph_retriever import GraphRetriever

    builder = get_graph_builder()
    if builder.graph.number_of_nodes() == 0:
        return {"path": [], "message": "知识图谱为空，请先构建"}

    # 复用全局 RAGChain 的图检索器（避免每次新建 LLM 实例）
    chain = get_rag_chain()
    retriever = chain.graph_retriever or GraphRetriever(graph_builder=builder)
    path = retriever.find_path(source, target)

    return {
        "source": source,
        "target": target,
        "path": path,
        "path_length": len(path),
        "found": len(path) > 0,
    }


@router.get("/api/graph/visual")
async def get_graph_visual_data():
    """
    获取知识图谱可视化数据（nodes + edges）。

    返回适合前端力导向图渲染的 JSON 格式。
    """
    from app.ingestion.graph_extractor import get_graph_builder
    builder = get_graph_builder()
    graph = builder.graph

    if graph.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "is_empty": True}

    # 计算节点度数用于大小映射
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    nodes = []
    for node in graph.nodes():
        deg = degrees.get(node, 0)
        nodes.append({
            "id": node,
            "label": node,
            "degree": deg,
            "size": 10 + (deg / max_degree) * 30,  # 10~40 映射
        })

    edges = []
    for head, tail, data in graph.edges(data=True):
        edges.append({
            "from": head,
            "to": tail,
            "relation": data.get("relation", "相关"),
            "source": data.get("source", ""),
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "is_empty": False,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
    }


@router.delete("/api/graph")
async def clear_graph():
    """清空知识图谱"""
    from app.ingestion.graph_extractor import get_graph_builder
    builder = get_graph_builder()
    builder.clear()
    return {"message": "知识图谱已清空"}


# ============ Health 端点 ============

@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    chain = get_rag_chain()
    try:
        all_chunks = chain.indexer.get_all_chunks()
        num_docs = len(set(c.metadata.get("doc_id") for c in all_chunks))
    except Exception:
        num_docs = 0

    return HealthResponse(
        status="ok",
        version="0.1.0",
        indexed_documents=num_docs,
    )


# ============ F11 可观测性端点 ============

@router.get("/api/metrics")
async def get_metrics_endpoint(format: str = "json"):
    """F11 指标导出。format=json（默认）返回结构化快照；format=prometheus 返回文本格式。

    含请求计数、时延直方图（count/avg/p50/p95）、缓存命中、忠实度分布等。
    """
    from app.observability.metrics import get_metrics
    registry = get_metrics()
    if format == "prometheus":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(registry.prometheus(), media_type="text/plain")
    return registry.snapshot()


@router.get("/api/cache/stats")
async def get_cache_stats():
    """F9 多级缓存命中率统计（语义缓存 + L1 embedding + L2 rerank）。"""
    chain = get_rag_chain()
    stats: dict = {}
    try:
        if chain.semantic_cache is not None:
            sc = chain.semantic_cache
            stats["semantic_cache"] = {"size": sc.size, "threshold": sc.threshold}
        if getattr(chain, "embedding_cache", None) is not None:
            c = chain.embedding_cache.cache
            stats["embedding_cache"] = {
                "size": len(c), "hits": c.hits, "misses": c.misses,
                "hit_rate": round(c.hit_rate, 4),
            }
        if chain.reranker is not None and getattr(chain.reranker, "cache", None) is not None:
            c = chain.reranker.cache.cache
            stats["rerank_cache"] = {
                "size": len(c), "hits": c.hits, "misses": c.misses,
                "hit_rate": round(c.hit_rate, 4),
            }
    except Exception as e:
        logger.warning(f"缓存统计失败: {e}")
    return stats


# ============ 辅助函数 ============

def _build_chat_history(history: list) -> list:
    """将 API 格式的对话历史转为 LangChain Message 格式"""
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def _build_chat_response(response: RAGResponse) -> ChatResponse:
    """将 RAGResponse 转为 API 响应格式"""
    sources = [
        SourceDocument(
            content=doc.page_content[:500],
            source=doc.metadata.get("source", ""),
            score=doc.metadata.get("rerank_score"),
            metadata={k: v for k, v in doc.metadata.items()
                      if k in ("chunk_id", "doc_id", "position")},
        )
        for doc in response.sources
    ]

    retrieval_detail = RetrievalDetail(
        queries_used=response.retrieval_result.queries_used,
        dense_count=len(response.retrieval_result.dense_results),
        sparse_count=len(response.retrieval_result.sparse_results),
        graph_count=len(response.retrieval_result.graph_results),
        fused_count=len(response.retrieval_result.fused_results),
        final_count=len(response.retrieval_result.documents),
        retrieval_time_ms=response.retrieval_result.retrieval_time_ms,
        crag_grade=response.retrieval_result.crag_grade,
        crag_action=response.retrieval_result.crag_action,
    )

    return ChatResponse(
        answer=response.answer,
        sources=sources,
        retrieval_detail=retrieval_detail,
        total_time_ms=response.total_time_ms,
        cache_hit=response.cache_hit,
        citations=[
            Citation(
                claim=c.claim, source=c.source, chunk_id=c.chunk_id,
                doc_index=c.doc_index, confidence=c.confidence, snippet=c.snippet,
            )
            for c in (response.citations or [])
        ],
        short_answer=response.short_answer,
        self_consistency_used=response.self_consistency_used,
        rewritten_query=response.rewritten_query,
    )
