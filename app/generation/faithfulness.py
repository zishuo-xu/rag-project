"""生成忠实度自检 - LLM-judge 检测答案是否被检索上下文支撑（生成层幻觉检测）

对应文章深度问题二：召回 100% 准确，LLM 仍可能生成错误答案。
检索再准也根除不了生成层幻觉，故在生成后做一次忠实度校验，
不忠实则用更严格的 prompt 重生成（见 chain._generate_faithful）。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import get_settings, get_llm_extra_body
from app.retrieval.crag import CRAGEvaluator
from app.generation.prompts import FAITHFULNESS_CHECK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    """忠实度检查结果"""
    faithful: Optional[bool]   # True=忠实 / False=含幻觉 / None=未知(放行,不阻断)
    score: float = 0.0         # 被支撑论断占比 [0,1]
    unsupported: List[str] = field(default_factory=list)  # 未被上下文支撑的论断
    reason: str = ""


class FaithfulnessChecker:
    """
    生成忠实度检查器（LLM-judge）。

    让模型从答案中抽取关键论断，逐条判断是否被检索上下文支撑，
    返回 score = 支撑论断数 / 总论断数；faithful = score >= threshold。
    异常或输出不可解析时返回 faithful=None（未知，放行，不阻断主流程）。
    """

    def __init__(self, llm=None, threshold: Optional[float] = None):
        settings = get_settings()
        self.threshold = (
            threshold if threshold is not None else settings.faithfulness_threshold
        )
        # 沿用项目模式：LLM 带 timeout/retry，关闭思考模式
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
            max_tokens=512,
            request_timeout=30,
            max_retries=2,
            extra_body=get_llm_extra_body(),
        )

    def check(
        self, question: str, context_docs: List[Document], answer: str
    ) -> FaithfulnessResult:
        """校验答案忠实度。任何异常都返回 faithful=None 放行，不抛出。"""
        if not answer or not answer.strip():
            return FaithfulnessResult(faithful=True, score=1.0, reason="空答案无需校验")

        context = self._format_context(context_docs)
        try:
            prompt = FAITHFULNESS_CHECK_PROMPT.format(
                question=question, context=context, answer=answer,
            )
            response = self.llm.invoke(prompt)
            data = CRAGEvaluator._extract_json(response.content)
            if not data:
                return FaithfulnessResult(faithful=None, reason="LLM 输出无法解析为JSON")

            score = max(0.0, min(1.0, float(data.get("score", 0.0))))
            unsupported = data.get("unsupported", []) or []
            reason = data.get("reason", "")
            faithful = score >= self.threshold
            logger.info(
                f"忠实度检查: score={score:.2f}, faithful={faithful}, "
                f"unsupported={len(unsupported)} ({reason})"
            )
            return FaithfulnessResult(
                faithful=faithful, score=score,
                unsupported=unsupported, reason=reason,
            )
        except Exception as e:
            logger.warning(f"忠实度检查失败，放行: {e}")
            return FaithfulnessResult(faithful=None, reason=f"检查异常: {e}")

    @staticmethod
    def _format_context(
        documents: List[Document], max_docs: int = 5, max_chars: int = 400
    ) -> str:
        """拼接前若干篇文档的截断内容作为事实来源"""
        parts = [
            f"[文档{i}] {doc.page_content[:max_chars]}"
            for i, doc in enumerate(documents[:max_docs], 1)
        ]
        return "\n".join(parts)
