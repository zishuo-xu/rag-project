"""多格式文档加载器 - 支持 PDF / Markdown / TXT"""

import os
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)

# 文件扩展名 -> 加载器映射
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".markdown": TextLoader,
}


def load_document(file_path: str) -> List[Document]:
    """
    加载单个文档，根据扩展名自动选择加载器。

    Args:
        file_path: 文档文件路径

    Returns:
        加载后的 Document 列表（含元数据）
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()
    loader_cls = LOADER_MAPPING.get(suffix)

    if loader_cls is None:
        raise ValueError(
            f"不支持的文件格式: {suffix}，"
            f"支持: {list(LOADER_MAPPING.keys())}"
        )

    logger.info(f"加载文档: {file_path} (格式: {suffix})")
    loader = loader_cls(str(path))
    docs = loader.load()

    # 补充元数据
    for i, doc in enumerate(docs):
        doc.metadata.update({
            "source": str(path.name),
            "file_path": str(path.absolute()),
            "doc_id": path.stem,
            "chunk_index": i,
        })

    logger.info(f"文档加载完成: {path.name}, 共 {len(docs)} 页/段")
    return docs


def load_directory(dir_path: str, recursive: bool = True) -> List[Document]:
    """
    批量加载目录下所有支持格式的文档。

    Args:
        dir_path: 目录路径
        recursive: 是否递归子目录

    Returns:
        所有文档的 Document 列表
    """
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"不是有效目录: {dir_path}")

    all_docs: List[Document] = []
    pattern = "**/*" if recursive else "*"

    for file_path in sorted(path.glob(pattern)):
        if file_path.suffix.lower() in LOADER_MAPPING:
            try:
                docs = load_document(str(file_path))
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"加载失败 {file_path}: {e}")

    logger.info(f"目录加载完成: {dir_path}, 共 {len(all_docs)} 个文档片段")
    return all_docs
