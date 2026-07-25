# app/ingestion/contextual.py
"""上下文增强分块 - 索引时为每个块生成文档级上下文（Anthropic Contextual Retrieval 思路）

动机：脱离文档上下文的"裸块"embedding 无法表达"这是讲 X 的文档里关于 Y 的段落"，
排序精度受限。索引时给每块补一句文档级定位并用"上下文+原文"做 embedding，
可让真正含答案的块排序上升。上下文生成是索引时一次性 LLM，在线检索零增量。
失败一律降级为裸块（空上下文），绝不阻断索引。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings, build_chat_llm

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """给定一篇文档和其中的一个片段，请用一句话（{max_chars}字以内）说明这个片段在文档中的定位
（文档主题 + 本片段在讲什么），用于增强该片段的检索向量。只输出这一句话，不要解释。

文档（节选）:
{doc_text}

片段:
{chunk_text}

上下文定位："""
)


def _get_llm():
    return build_chat_llm(timeout=30, retries=2)


def generate_chunk_context(
    doc_text: str,
    chunk_text: str,
    llm=None,
    max_chars: Optional[int] = None,
) -> str:
    """为单个块生成文档级上下文；任何失败返回空串（降级为裸块）。"""
    settings = get_settings()
    max_chars = max_chars or settings.contextual_max_chars
    if llm is None:
        llm = _get_llm()
    try:
        prompt_value = CONTEXT_PROMPT.invoke({
            "doc_text": (doc_text or "")[:4000],
            "chunk_text": chunk_text or "",
            "max_chars": max_chars,
        })
        result = llm.invoke(prompt_value.to_messages())
        ctx = result.content if hasattr(result, "content") else str(result)
        ctx = (ctx or "").strip().split("\n")[0].strip()
        return ctx[:max_chars] if ctx else ""
    except Exception as e:
        logger.warning(f"上下文生成失败，降级裸块: {e}")
        return ""


def build_chunk_contexts(
    chunks: List[Document],
    llm=None,
    max_chars: Optional[int] = None,
) -> List[str]:
    """为每个块生成上下文，返回与 chunks 等长、顺序一致的列表。

    同 doc_id 的块共享一份 doc_text（由该文档所有块拼接近似还原）。
    """
    if llm is None:
        llm = _get_llm()

    # 按 doc_id 还原 doc_text
    doc_texts: dict[str, str] = {}
    for ch in chunks:
        doc_id = ch.metadata.get("doc_id", "unknown")
        doc_texts[doc_id] = doc_texts.get(doc_id, "") + "\n" + ch.page_content

    contexts: List[str] = []
    for ch in chunks:
        doc_id = ch.metadata.get("doc_id", "unknown")
        contexts.append(
            generate_chunk_context(doc_texts[doc_id], ch.page_content, llm=llm, max_chars=max_chars)
        )
    return contexts
