"""API 端点测试"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """测试客户端"""
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
