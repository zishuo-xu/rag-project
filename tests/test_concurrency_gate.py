"""并发闸门测试（mock RAGChain，离线运行）"""
import asyncio
import io
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes import router, set_rag_chain, set_concurrency_gate
from app.generation.chain import RAGResponse
from app.retrieval.pipeline import RetrievalResult


@pytest.fixture(autouse=True)
def _reset_global_state():
    """每个测试后复位全局状态，避免污染其他测试文件"""
    yield
    set_concurrency_gate(None)
    set_rag_chain(None)


def _make_app(concurrency: int):
    app = FastAPI()
    app.include_router(router)

    active = {"cur": 0, "max": 0}

    def slow_invoke(question, chat_history=None, top_k=None):
        active["cur"] += 1
        active["max"] = max(active["max"], active["cur"])
        time.sleep(0.2)  # 模拟阻塞的 LLM 调用
        active["cur"] -= 1
        return RAGResponse(
            answer="ok", sources=[],
            retrieval_result=RetrievalResult(documents=[]),
        )

    chain = MagicMock()
    chain.invoke.side_effect = slow_invoke
    set_rag_chain(chain)
    set_concurrency_gate(asyncio.Semaphore(concurrency))
    return app, active


@pytest.mark.asyncio
async def test_gate_limits_concurrency():
    """并发 5 个请求，闸门=2 时同时在执行的 invoke 不超过 2"""
    app, active = _make_app(concurrency=2)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        await asyncio.gather(*[
            client.post("/api/chat", json={"question": f"q{i}", "stream": False})
            for i in range(5)
        ])
    assert active["max"] <= 2


@pytest.mark.asyncio
async def test_gate_timeout_returns_503(monkeypatch):
    """排队超时返回 503"""
    app, _ = _make_app(concurrency=1)
    settings = MagicMock()
    settings.request_queue_timeout = 0.05  # 50ms 必超时
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        results = await asyncio.gather(*[
            client.post("/api/chat", json={"question": f"q{i}", "stream": False})
            for i in range(3)
        ])
    statuses = [r.status_code for r in results]
    assert 503 in statuses
