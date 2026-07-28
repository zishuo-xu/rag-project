"""文档加载器测试（错误分支 + 正常路径，离线）"""
import pytest

from app.ingestion.loader import load_document, LOADER_MAPPING


def test_supported_extensions_mapping():
    assert set(LOADER_MAPPING.keys()) == {".pdf", ".txt", ".md", ".markdown"}


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/path/doc.md")


def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "data.docx"
    f.write_bytes(b"whatever")
    with pytest.raises(ValueError):
        load_document(str(f))


def test_load_txt_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("第一行内容\n第二行内容", encoding="utf-8")
    docs = load_document(str(f))
    assert docs, "应加载出至少一个文档"
    assert "第一行内容" in docs[0].page_content


def test_load_markdown_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    docs = load_document(str(f))
    assert docs
    assert "正文内容" in docs[0].page_content


def test_extension_case_insensitive(tmp_path):
    f = tmp_path / "UPPER.TXT"
    f.write_text("大小写扩展名", encoding="utf-8")
    docs = load_document(str(f))
    assert docs
