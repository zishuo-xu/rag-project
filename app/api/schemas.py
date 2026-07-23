"""Pydantic 数据模型 - 请求/响应 Schema"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============ Chat 相关 ============

class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    chat_history: List[dict] = Field(
        default_factory=list,
        description="对话历史, 格式: [{role: user, content: ...}, ...]"
    )
    top_k: int = Field(default=5, description="检索文档数量", ge=1, le=20)
    use_query_transform: bool = Field(default=True, description="是否使用查询改写")
    use_rerank: bool = Field(default=True, description="是否使用重排序")
    query_strategy: str = Field(
        default="multi_query",
        description="查询改写策略: multi_query | hyde | none"
    )
    stream: bool = Field(default=False, description="是否流式返回")


class SourceDocument(BaseModel):
    """来源文档"""
    content: str = Field(description="文档内容片段")
    source: str = Field(default="", description="文档来源")
    score: Optional[float] = Field(default=None, description="相关性分数")
    metadata: dict = Field(default_factory=dict, description="元数据")


class RetrievalDetail(BaseModel):
    """检索过程详情"""
    queries_used: List[str] = Field(default_factory=list)
    dense_count: int = 0
    sparse_count: int = 0
    graph_count: int = 0
    fused_count: int = 0
    final_count: int = 0
    retrieval_time_ms: float = 0
    crag_grade: str = Field(default="", description="CRAG 评级: correct/ambiguous/incorrect/recovered")
    crag_action: str = Field(default="", description="CRAG 采取的动作")


class Citation(BaseModel):
    """F7 引用溯源：答案 claim 关联到的源文档块"""
    claim: str = Field(description="答案中的论断句")
    source: str = Field(default="", description="来源文档")
    chunk_id: str = Field(default="", description="源块 id")
    doc_index: int = Field(default=0, description="文档序号(1-based)")
    confidence: float = Field(default=0.0, description="claim-块相似度置信度")
    snippet: str = Field(default="", description="证据片段")


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(description="回答内容")
    sources: List[SourceDocument] = Field(default_factory=list, description="引用来源")
    retrieval_detail: RetrievalDetail = Field(default_factory=RetrievalDetail)
    total_time_ms: float = 0
    cache_hit: bool = Field(default=False, description="是否命中语义缓存")
    # RAG 3.0 增强字段
    citations: List[Citation] = Field(default_factory=list, description="F7 引用溯源")
    short_answer: str = Field(default="", description="F10 抽取的核心短答案")
    self_consistency_used: bool = Field(default=False, description="F10 是否触发自一致性")
    rewritten_query: str = Field(default="", description="F12 历史感知重写后的查询")


# ============ Document 相关 ============

class UploadResponse(BaseModel):
    """文档上传响应"""
    message: str
    filename: str
    num_chunks: int = 0
    doc_id: str = ""


class DocumentInfo(BaseModel):
    """文档信息"""
    doc_id: str
    source: str
    num_chunks: int = 0


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    documents: List[DocumentInfo] = Field(default_factory=list)
    total: int = 0


# ============ Evaluation 相关 ============

class EvalRequest(BaseModel):
    """评估请求"""
    questions: List[str] = Field(default_factory=list, description="测试问题列表")
    ground_truths: List[str] = Field(default_factory=list, description="标准答案列表")


class EvalMetrics(BaseModel):
    """评估指标"""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class EvalReport(BaseModel):
    """评估报告"""
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    num_samples: int = 0
    details: List[dict] = Field(default_factory=list)


# ============ Health ============

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    version: str = "0.1.0"
    indexed_documents: int = 0
