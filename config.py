"""统一配置管理模块 - 支持环境变量 + 默认值"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """应用配置，优先从环境变量读取，否则使用默认值"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI / DeepSeek
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Embedding
    embedding_provider: str = "local"  # "local" | "openai"
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_chunk_collection: str = "chunks"
    chroma_summary_collection: str = "summaries"

    # Retrieval
    retrieval_top_k: int = 5
    retrieval_rrf_k: int = 60
    rerank_top_n: int = 20
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_chunk_threshold: float = 0.75

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Graph RAG
    graph_enabled: bool = True
    graph_max_hops: int = 2  # 图检索最大跳数
    graph_max_entities: int = 5  # 每次查询最多匹配实体数
    graph_persist_path: str = "./data/knowledge_graph.json"

    # Parent-Child 检索
    use_parent_child: bool = True
    parent_chunk_size: int = 1024
    child_chunk_size: int = 200
    chroma_child_collection: str = "child_chunks"
    chroma_parent_collection: str = "parent_chunks"

    # CRAG 自纠正检索
    use_crag: bool = True
    crag_relevance_threshold: float = 0.5

    # 检索管道接线
    use_summary_recall: bool = True  # L1 摘要索引作为第5路召回
    use_crag_gate: bool = True       # CRAG 门控：判断是否需要检索
    recall_max_workers: int = 6      # 多路召回线程池大小

    # RAG 2.0 深度增强（每项独立开关，异常均优雅降级到原行为）
    # F1 Autocut 自适应截断（Kneedle 膝点检测，替代固定 TopK 降噪）
    use_autocut: bool = True
    autocut_min_docs: int = 2        # 截断下界（上界复用 retrieval_top_k）
    # F2 Self-RAG 迭代检索（质量驱动终止：充分性/收敛性，硬上限仅兜底）
    use_iterative_retrieval: bool = True
    max_retrieval_iterations: int = 2
    # F3 生成忠实度自检（幻觉检测 + 严格重生成）
    use_faithfulness_check: bool = True
    faithfulness_threshold: float = 0.7
    faithfulness_max_regen: int = 1
    # F4 查询路由 / 类型自适应（规则驱动，零 LLM）
    use_query_router: bool = True

    # F6 答案定位增强（每项独立开关，异常均优雅降级到原行为）
    # F6a 细粒度召回 + 上下文增强（零在线 LLM 增量）
    use_contextual_chunks: bool = True            # 查询时使用上下文增强嵌入集合
    contextual_max_chars: int = 80                # 上下文片段长度上限
    chroma_contextual_collection: str = "chunks_contextual"
    # F6b 多跳查询分解（仅多跳查询触发；并行优先，依赖时链式）
    use_decomposition: bool = True
    decomposition_max_subquestions: int = 4       # 子问题数上限
    decomposition_max_hops: int = 2               # 链式分解跳数硬上限（3→2：每跳界 1 次串行 refine LLM，2026-07-26 延迟治理）

    # 延迟治理（2026-07-26）：全局时延预算，超预算熔断可选阶段（F2 迭代/F3 重生成）
    latency_budget_ms: int = 25000   # 查询级预算；<=0 关闭（典型路径 ~11s，仅熔断离群尾）
    answer_max_tokens: int = 1024    # 生成答案 token 封顶（原无界，长答案拖慢生成）

    # 并发控制
    max_concurrent_requests: int = 4   # /api/chat 并发闸门
    request_queue_timeout: float = 30.0  # 排队超时秒数，超时返回503

    # LLM 思考模式（DeepSeek reasoning）
    llm_thinking_enabled: bool = False  # False=关闭思考，直接输出，更快更省token

    # 语义缓存
    cache_enabled: bool = True
    cache_threshold: float = 0.92
    cache_ttl: int = 3600
    cache_max_size: int = 200

    # ===== RAG 3.0 生产级增强（F7-F12，每项独立开关，异常均优雅降级） =====
    # F7 引用溯源与答案定位（零在线 LLM：claim-块 embedding 余弦关联）
    use_citations: bool = True
    citation_threshold: float = 0.5      # claim-块相似度置信度下界
    citation_max_claims: int = 6         # 答案最多切分的 claim 数（控制编码量）

    # F8 低延迟流式 + 投机忠实度（先流式吐字，流末自检，不忠实追加 correction）
    use_speculative_streaming: bool = True

    # F9 多级缓存（L1 embedding / L2 rerank，L3 为既有语义响应缓存）
    use_embedding_cache: bool = True
    embedding_cache_size: int = 512
    use_rerank_cache: bool = True
    rerank_cache_size: int = 256

    # F10 答案质量增强（聚焦 prompt + 零LLM抽取 + 自适应自一致性）
    use_answer_focus: bool = True                 # 答案前置聚焦 prompt
    use_answer_extraction: bool = True            # 零LLM 抽取核心短答案 span
    use_self_consistency: bool = False            # 默认关以保时延，评测可开
    self_consistency_samples: int = 3             # 采样投票次数
    self_consistency_types: str = "numeric,factual"  # 仅这些查询类型触发自一致性

    # F11 可观测性与生产加固
    enable_metrics: bool = True            # 进程内指标 + /api/metrics 导出
    api_key: str = ""                      # 空=关闭鉴权；非空校验 X-API-Key
    rate_limit_rpm: int = 0                # 每客户端每分钟请求上限，0=关闭
    log_json: bool = False                 # 结构化 JSON 日志

    # F12 多轮对话记忆 / 历史感知查询重写
    use_history_rewrite: bool = True
    history_rewrite_use_llm: bool = False  # 默认零LLM启发式，开启后用一次小调用
    history_rewrite_max_turns: int = 4     # 重写时参考的最近历史轮数

    # F13 Agentic RAG（ReAct 状态机自主检索；默认关，异常/空证据降级回七阶段管道）
    use_agentic: bool = False
    agentic_max_steps: int = 4             # 决策步数硬上限（安全兜底）
    agentic_decision_max_tokens: int = 256  # 每步决策调用 token 预算

    # LangSmith (Optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-project"


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()


def get_llm_extra_body() -> dict | None:
    """LLM extra_body 参数：关闭思考模式时返回禁用参数，开启时返回 None"""
    if get_settings().llm_thinking_enabled:
        return None
    return {"thinking": {"type": "disabled"}}
