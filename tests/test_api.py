"""API 端点测试（mock RAGChain，离线运行）"""

import pytest
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router, set_rag_chain


@pytest.fixture
def client():
    """测试客户端：独立 FastAPI 实例 + mock RAGChain，不触发真实初始化"""
    app = FastAPI()
    app.include_router(router)
    set_rag_chain(MagicMock())
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查测试"""

    def test_health(self, client):
        """健康检查返回 200"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestChatEndpoint:
    """对话接口测试"""

    def test_chat_validation(self, client):
        """空问题应返回 422"""
        response = client.post("/api/chat", json={"question": ""})
        assert response.status_code == 422

    def test_chat_missing_question(self, client):
        """缺少 question 字段"""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_stream_retrieval_event_has_summary_count(self):
        """SSE retrieval 事件应包含 summary_count 字段"""
        chain = MagicMock()
        retrieval = MagicMock()
        retrieval.queries_used = ["q1"]
        retrieval.dense_results = []
        retrieval.sparse_results = []
        retrieval.fused_results = []
        retrieval.documents = []
        retrieval.retrieval_time_ms = 1.0
        retrieval.crag_grade = "correct"
        retrieval.summary_results = [MagicMock(), MagicMock()]
        retrieval.crag_action = "none"
        chain.invoke_stream.return_value = iter([
            {"type": "retrieval", "data": retrieval},
        ])

        app = FastAPI()
        app.include_router(router)
        set_rag_chain(chain)
        client = TestClient(app)

        response = client.post("/api/chat", json={"question": "hi", "stream": True})
        assert response.status_code == 200
        assert "event: retrieval" in response.text
        assert '"summary_count": 2' in response.text


class TestDocumentEndpoint:
    """文档接口测试"""

    def test_list_documents(self, client):
        """获取文档列表"""
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_unsupported_format(self, client):
        """上传不支持的格式"""
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.exe", b"binary content", "application/octet-stream")},
        )
        assert response.status_code == 400
