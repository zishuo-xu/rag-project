"""FastAPI 路由定义 - Chat / Documents / Evaluation / Health"""

import json
import logging
import tempfile
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import (
    ChatRequest, ChatResponse, SourceDocument, RetrievalDetail,
    UploadResponse, DocumentListResponse, DocumentInfo,
    EvalRequest, EvalReport, HealthResponse,
)
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


# ============ Chat 端点 ============

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口 - 支持普通和流式两种模式。

    完整流程：查询改写 -> 多路召回 -> RRF融合 -> Rerank -> LLM生成
    """
    chain = get_rag_chain()

    # 构建对话历史
    chat_history = _build_chat_history(request.chat_history)

    if request.stream:
        return EventSourceResponse(
            _stream_response(chain, request, chat_history)
        )

    # 非流式：完整调用
    response = chain.invoke(
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
    """SSE 流式响应生成器"""
    for event in chain.invoke_stream(
        question=request.question,
        chat_history=chat_history,
        top_k=request.top_k,
    ):
        if event["type"] == "retrieval":
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
                }, ensure_ascii=False),
            }
        elif event["type"] == "token":
            yield {"event": "token", "data": event["data"]}
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
            yield {
                "event": "done",
                "data": json.dumps({
                    "sources": sources,
                    "total_time_ms": response.total_time_ms,
                }, ensure_ascii=False),
            }


# ============ Documents 端点 ============

@router.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并建立索引。

    支持格式：PDF, TXT, Markdown
    """
    chain = get_rag_chain()

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
        chunks = smart_chunk(docs)
        chain.indexer.index_documents(chunks)

        # 重建 BM25 索引
        chain.sparse_retriever.build_index()

        return UploadResponse(
            message=f"文档 '{file.filename}' 上传并索引成功",
            filename=file.filename,
            num_chunks=len(chunks),
            doc_id=docs[0].metadata.get("doc_id", "") if docs else "",
        )

    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


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
        fused_count=len(response.retrieval_result.fused_results),
        final_count=len(response.retrieval_result.documents),
        retrieval_time_ms=response.retrieval_result.retrieval_time_ms,
    )

    return ChatResponse(
        answer=response.answer,
        sources=sources,
        retrieval_detail=retrieval_detail,
        total_time_ms=response.total_time_ms,
    )
