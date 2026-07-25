"""上传 API chunk_strategy 参数测试（mock RAGChain，离线运行）"""
import io
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router, set_rag_chain


def _make_client():
    app = FastAPI()
    app.include_router(router)
    chain = MagicMock()
    chain.indexer.index_documents.return_value = None
    chain.sparse_retriever.add_documents.return_value = None
    set_rag_chain(chain)
    return TestClient(app), chain


def test_upload_default_recursive():
    """默认走递归分块"""
    client, chain = _make_client()
    with patch("app.ingestion.service.smart_chunk") as mock_chunk, \
         patch("app.ingestion.service.load_document") as mock_load:
        mock_load.return_value = [MagicMock(metadata={"doc_id": "d1"})]
        mock_chunk.return_value = [MagicMock()]
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
        )
    assert resp.status_code == 200
    assert mock_chunk.call_args.kwargs.get("use_semantic", False) is False


def test_upload_semantic_strategy():
    """chunk_strategy=semantic 时启用语义分块"""
    client, chain = _make_client()
    with patch("app.ingestion.service.smart_chunk") as mock_chunk, \
         patch("app.ingestion.service.load_document") as mock_load:
        mock_load.return_value = [MagicMock(metadata={"doc_id": "d1"})]
        mock_chunk.return_value = [MagicMock()]
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
            data={"chunk_strategy": "semantic"},
        )
    assert resp.status_code == 200
    assert mock_chunk.call_args.kwargs["use_semantic"] is True


def test_upload_invalid_strategy_400():
    """非法 chunk_strategy 返回 400"""
    client, _ = _make_client()
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", io.BytesIO("测试内容".encode()), "text/plain")},
        data={"chunk_strategy": "banana"},
    )
    assert resp.status_code == 400
