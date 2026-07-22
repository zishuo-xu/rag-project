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
