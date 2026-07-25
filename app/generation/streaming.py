"""F8 低延迟流式 + 投机忠实度 - 先流式吐字，流末自检，不忠实追加 correction

RAG 2.0 的 F3 在流式路径会"先非流式完整生成+自检，再整体吐出"，导致首 token
时延（TTFT）≈ 完整生成时延，丢失逐 token 体验，full 模式时延 +1.8s。

本模块实现投机流式：先把答案逐 token 流给用户（快 TTFT），流结束后做忠实度
检查；若不忠实则严格重生成并追加一个 correction 事件。用户既快又能看到已校验结果。

时延：TTFT 从 ~完整生成 降到 ~首 token；自检与重生成只在流末发生，不阻塞首屏。
降级：checker 为 None 时直接放行（与 F3 关闭行为一致）。
check+regen 循环与 chain F3 阻塞路径同源（faithfulness.regen_until_faithful），
流式路径同样受延迟预算约束（deadline 透传）。
"""

import logging
from typing import Callable, Generator, List, Optional

from langchain_core.documents import Document

from app.generation.faithfulness import regen_until_faithful

logger = logging.getLogger(__name__)


def speculative_faithful_stream(
    *,
    stream_fn: Callable[[], Generator[str, None, None]],
    question: str,
    documents: List[Document],
    chat_history: Optional[List],
    checker,                       # FaithfulnessChecker 或 None
    regen_fn: Callable[[], str],   # 严格重生成（strict=True）
    max_regen: int = 1,
    deadline=None,                 # 延迟治理 Deadline（与 chain F3 同一 F3_regen 熔断）
) -> Generator[dict, None, None]:
    """投机流式生成 + 流末忠实度自检。

    Yields:
        {"type": "token", "data": str}        逐 token（首屏快）
        {"type": "correction", "data": str}   不忠实时的严格重生成完整答案
        {"type": "final", "data": {...}}      最终答案与忠实度元信息
    """
    full_answer = ""
    for token in stream_fn():
        full_answer += token
        yield {"type": "token", "data": token}

    # 无自检器：直接放行
    if checker is None:
        yield {"type": "final", "data": {
            "answer": full_answer, "faithful": None, "score": 0.0, "regenerated": False,
        }}
        return

    # 流末忠实度检查 + 有界严格重生成（与 chain F3 共用循环体）
    corrections: List[str] = []
    answer, faithful, fb_score, regenerated = regen_until_faithful(
        checker, question, documents, full_answer,
        produce_fn=regen_fn, max_regen=max_regen, deadline=deadline,
        on_regen=corrections.append,
    )
    for correction in corrections:
        yield {"type": "correction", "data": correction}

    yield {"type": "final", "data": {
        "answer": answer, "faithful": faithful,
        "score": fb_score, "regenerated": regenerated,
    }}
