"""FastAPI 应用入口 - RAG 系统 API 服务"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from app.api.routes import router, set_rag_chain
from app.generation.chain import RAGChain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("正在初始化 RAG 系统...")
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY 未配置！\n"
            "请执行: cp .env.example .env  然后填入你的 API Key"
        )

    # 初始化 RAG Chain
    rag_chain = RAGChain(
        use_query_transform=True,
        use_rerank=True,
        query_strategy="multi_query",
    )
    set_rag_chain(rag_chain)

    # 构建 BM25 索引（如果有已索引的文档）
    try:
        rag_chain.sparse_retriever.build_index()
    except Exception as e:
        logger.warning(f"BM25 索引构建跳过: {e}")

    logger.info("RAG 系统初始化完成")
    yield
    logger.info("RAG 系统关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG System API",
    description="生产级检索增强生成系统 - 多路召回 + Rerank + 层级索引",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
