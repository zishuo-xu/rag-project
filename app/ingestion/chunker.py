"""智能分块模块 - 递归字符分块 + 语义分块"""

import logging
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger = logging.getLogger(__name__)


def recursive_chunk(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    递归字符分块 - 按段落/句子/字符层级递归切分。

    Args:
        documents: 原始文档列表
        chunk_size: 分块大小（字符数），默认从配置读取
        chunk_overlap: 重叠大小，默认从配置读取

    Returns:
        分块后的 Document 列表
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # 为每个 chunk 添加唯一 ID 和位置信息
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'unknown')}_{i}"
        chunk.metadata["position"] = i

    logger.info(
        f"递归分块完成: {len(documents)} 文档 -> {len(chunks)} 块 "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def semantic_chunk(
    documents: List[Document],
    embeddings,
    threshold: float | None = None,
) -> List[Document]:
    """
    语义分块 - 基于 embedding 相似度检测语义边界。

    原理：计算相邻句子的 embedding 余弦相似度，
    当相似度低于阈值时，认为语义发生跳变，在此处切分。

    Args:
        documents: 原始文档列表
        embeddings: LangChain Embedding 模型实例
        threshold: 语义相似度阈值（低于此值则切分），默认从配置读取

    Returns:
        语义分块后的 Document 列表
    """
    settings = get_settings()
    threshold = threshold or settings.semantic_chunk_threshold

    # 先按句子拆分
    sentence_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        separators=["。", ".", "\n", " ", ""],
    )
    sentences = sentence_splitter.split_documents(documents)

    if len(sentences) <= 1:
        return sentences

    # 计算所有句子的 embedding
    texts = [s.page_content for s in sentences]
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors)

    # 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(vectors) - 1):
        sim = np.dot(vectors[i], vectors[i + 1]) / (
            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i + 1]) + 1e-8
        )
        similarities.append(sim)

    # 在相似度低于阈值处切分
    chunks: List[Document] = []
    current_sentences: List[str] = [texts[0]]
    current_metadata = sentences[0].metadata.copy()

    for i, sim in enumerate(similarities):
        if sim < threshold:
            # 语义边界：合并当前累积的句子为一个 chunk
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Document(page_content=chunk_text, metadata=current_metadata.copy())
            )
            current_sentences = [texts[i + 1]]
            current_metadata = sentences[i + 1].metadata.copy()
        else:
            current_sentences.append(texts[i + 1])

    # 最后一个 chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            Document(page_content=chunk_text, metadata=current_metadata.copy())
        )

    # 添加 chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'unknown')}_sem_{i}"
        chunk.metadata["position"] = i

    logger.info(
        f"语义分块完成: {len(sentences)} 句 -> {len(chunks)} 块 "
        f"(threshold={threshold})"
    )
    return chunks


def smart_chunk(
    documents: List[Document],
    embeddings=None,
    use_semantic: bool = False,
    short_doc_threshold: int = 1000,
) -> List[Document]:
    """
    智能分块入口 - 根据配置选择分块策略。

    优化：短文档（< short_doc_threshold 字符）整段保留不切分，
    避免小文档被切碎导致上下文丢失。

    Args:
        documents: 原始文档列表
        embeddings: Embedding 模型（语义分块时需要）
        use_semantic: 是否使用语义分块
        short_doc_threshold: 短文档阈值（字符数），低于此值不切分

    Returns:
        分块后的 Document 列表
    """
    # 分离短文档和长文档
    short_docs = [d for d in documents if len(d.page_content) <= short_doc_threshold]
    long_docs = [d for d in documents if len(d.page_content) > short_doc_threshold]

    chunks: List[Document] = []

    # 短文档：整段保留，只添加元数据
    for i, doc in enumerate(short_docs):
        doc.metadata.setdefault("chunk_id", f"{doc.metadata.get('doc_id', 'unknown')}_{i}")
        doc.metadata.setdefault("position", 0)
        chunks.append(doc)

    if short_docs:
        logger.info(f"短文档保留: {len(short_docs)} 篇 (<= {short_doc_threshold} 字符不切分)")

    # 长文档：正常分块
    if long_docs:
        if use_semantic and embeddings:
            chunks.extend(semantic_chunk(long_docs, embeddings))
        else:
            chunks.extend(recursive_chunk(long_docs))

    return chunks
