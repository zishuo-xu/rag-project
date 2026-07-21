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

    # LangSmith (Optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-project"


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例"""
    return Settings()
