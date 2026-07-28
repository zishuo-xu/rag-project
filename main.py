"""FastAPI 应用入口 - RAG 系统 API 服务"""

import os


def _local_models_cached() -> bool:
    """检测本地 embedding/rerank 模型是否已在 HF 缓存中。

    有缓存才开启离线模式(跳过网络检查、加速启动);干净机器上保持联网,
    让首跑自动下载模型,避免强制离线导致 LocalEntryNotFoundError 崩溃。
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        from config import get_settings

        s = get_settings()
        for model_id in (s.embedding_model, s.rerank_model):
            if try_to_load_from_cache(model_id, "config.json") in (None, "repo"):
                return False
        return True
    except Exception:
        return False


if _local_models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 跳过 HuggingFace 网络检查，加速启动
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from app.api.routes import router, set_rag_chain, set_concurrency_gate
from app.generation.chain import RAGChain


def _setup_logging():
    """配置日志：log_json=True 时输出结构化 JSON 行（F11），否则普通格式。"""
    settings = get_settings()
    if settings.log_json:
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }, ensure_ascii=False)

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


_setup_logging()
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

    # 并发闸门：限制同时处理的 chat 请求数，防止高并发打爆 LLM
    import asyncio
    set_concurrency_gate(asyncio.Semaphore(settings.max_concurrent_requests))
    logger.info(f"并发闸门: max_concurrent_requests={settings.max_concurrent_requests}")

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

# F11 生产加固：API Key 鉴权 + 限流（仅在配置开启时注册，默认关闭不影响现有行为）
_settings = get_settings()
if _settings.api_key or _settings.rate_limit_rpm > 0:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from app.api.security import is_exempt, verify_api_key, get_rate_limiter

    class SecurityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if not is_exempt(request.url.path):
                if not verify_api_key(request.headers.get("X-API-Key")):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "无效或缺失的 API Key"},
                    )
                client = request.client.host if request.client else "unknown"
                if not get_rate_limiter().allow(client):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "请求过于频繁，请稍后重试"},
                    )
            return await call_next(request)

    app.add_middleware(SecurityMiddleware)
    logger.info(
        f"F11 生产加固已启用: api_key={'on' if _settings.api_key else 'off'}, "
        f"rate_limit_rpm={_settings.rate_limit_rpm}"
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
