"""生成忠实度自检 - LLM-judge 检测答案是否被检索上下文支撑（生成层幻觉检测）

对应文章深度问题二：召回 100% 准确，LLM 仍可能生成错误答案。
检索再准也根除不了生成层幻觉，故在生成后做一次忠实度校验，
不忠实则用更严格的 prompt 重生成（见 chain._generate_faithful）。
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import get_settings, build_chat_llm
from app.utils import extract_json
from app.generation.prompts import FAITHFULNESS_CHECK_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class FaithfulnessResult:
    """忠实度检查结果"""
    faithful: Optional[bool]   # True=忠实 / False=含幻觉 / None=未知(放行,不阻断)
    score: float = 0.0         # 被支撑论断占比 [0,1]
    unsupported: List[str] = field(default_factory=list)  # 未被上下文支撑的论断
    reason: str = ""


def regen_until_faithful(
    checker: "FaithfulnessChecker",
    question: str,
    context: str,
    answer: str,
    produce_fn: Callable[[], str],
    max_regen: int = 1,
    deadline=None,
    on_regen: Optional[Callable[[str], None]] = None,
) -> tuple[str, Optional[bool], float, bool]:
    """忠实度 check+regen 共享循环（chain F3 阻塞路径 / F8 投机流式 共用）。

    对初始答案做忠实度检查；不忠实则调用 produce_fn 严格重生成，最多 max_regen 次。
    时延预算熔断：deadline 超预算时跳过后续重生成（每次重生成 = 2 次串行 LLM）。

    Args:
        context: 生成器实际使用的格式化上下文（单一事实源；裁判与生成同据，
            消除旧版裁判自行截断上下文导致的假阳性重生成）
        produce_fn: 严格重生成函数（strict prompt），返回新答案
        deadline: 延迟治理 Deadline（可选）；F8 流式路径同样受预算约束
        on_regen: 每次重生成后的回调（F8 用于发 correction 事件）

    Returns:
        (最终答案, faithful, score, 是否触发过重生成)
    """
    fb = checker.check(question, context, answer)
    regenerated = False
    regen_left = max_regen
    while fb.faithful is False and regen_left > 0:
        if deadline is not None and deadline.check_skip("F3_regen"):
            logger.info("忠实度不足但时延预算耗尽，跳过严格重生成")
            break
        logger.info(f"忠实度不足(score={fb.score:.2f})，触发严格重生成")
        answer = produce_fn()
        regenerated = True
        regen_left -= 1
        if on_regen:
            on_regen(answer)
        fb = checker.check(question, context, answer)
    return answer, fb.faithful, fb.score, regenerated


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
        self.llm = llm or build_chat_llm(max_tokens=512, timeout=30, retries=2)

    def check(
        self, question: str, context: str, answer: str
    ) -> FaithfulnessResult:
        """校验答案忠实度。context 为生成器实际使用的格式化上下文（单一事实源）。

        任何异常都返回 faithful=None 放行，不抛出。
        """
        if not answer or not answer.strip():
            return FaithfulnessResult(faithful=True, score=1.0, reason="空答案无需校验")

        try:
            prompt = FAITHFULNESS_CHECK_PROMPT.format(
                question=question, context=context, answer=answer,
            )
            response = self.llm.invoke(prompt)
            data = extract_json(response.content)
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
