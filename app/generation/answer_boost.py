"""F10 答案质量增强 - 零LLM短答案抽取 + 自适应自一致性

针对 RAG 2.0 端到端 EM=0 / F1≈0.36 的痛点：完整回答冗长、把答案埋在解释里，
与短答案 span 对不齐。本模块两件套（均可独立开关）：

1. 零LLM 短答案抽取：从回答中抽出核心短答案 span（数字型优先抽数字），
   写入 RAGResponse.short_answer，供评测对齐与前端高亮。
2. 自适应自一致性：仅对 numeric/factual 短答案型查询采样 N 次抽取投票，
   其余类型跳过以保时延（默认关闭，评测可开）。

答案前置由基础 QA Prompt 的"聚焦问题"原则约束（prompts.RAG_SYSTEM_PROMPT）。
时延：抽取零额外调用；自一致性默认关。异常返回原答案/空抽取（降级）。
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import get_settings
from app.retrieval.router import is_numeric_question

logger = logging.getLogger(__name__)

_SOURCE_TAG = re.compile(r"\[(?:来源|文档|source|doc)[^\]]*\]", re.IGNORECASE)
# 答案前置填充词（抽取时剔除）
_FILLER = re.compile(r"^(?:答案是|答案为|根据文档[，,]?|根据参考文档[，,]?|根据上述文档[，,]?|答[:：]|回答[:：]|简答[:：])")


@dataclass
class BoostResult:
    """答案增强结果。"""
    short_answer: str = ""                 # 抽取的核心短答案 span
    self_consistency_used: bool = False    # 是否触发自一致性投票
    voted_answer: str = ""                 # 自一致性投票胜出的短答案（空=未触发）
    samples: int = 0                       # 实际采样次数


def extract_short_answer(question: str, answer: str) -> str:
    """零LLM 从完整回答中抽取核心短答案 span。

    - 数字型问题：优先抽取年份/数字（如 "1963年"）。
    - 其余：取首个实质句子，剔除来源标注与"答案是"等填充词。
    """
    if not answer or not answer.strip():
        return ""
    ans = _SOURCE_TAG.sub(" ", answer)

    # 数字型：优先抽数字/年份（与 router 共用唯一判定）
    if is_numeric_question(question):
        m = re.search(r"\d{2,4}\s*年", ans) or re.search(r"\d+(?:\.\d+)?\s*[%％]?", ans)
        if m:
            return m.group().strip()

    # 取首个实质句子
    for raw in re.split(r"[。！？.!?\n]", ans):
        sent = raw.strip()
        sent = _FILLER.sub("", sent).strip()
        sent = sent.lstrip("-*•0123456789.、) ").strip()
        if len(sent) >= 2:
            return sent[:60]
    return ans.strip()[:60]


def self_consistency_vote(short_answers: List[str]) -> str:
    """对多个抽取出的短答案做多数投票，返回胜出者（平票取最先出现）。"""
    valid = [s for s in short_answers if s and s.strip()]
    if not valid:
        return ""
    counts = Counter(valid)
    top = counts.most_common()
    best_count = top[0][1]
    # 在最高票候选中，取原始顺序最先出现者（稳定）
    for s in valid:
        if counts[s] == best_count:
            return s
    return valid[0]


class AnswerBooster:
    """答案增强器：抽取 + 自适应自一致性。"""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()

    def _enabled_types(self) -> set:
        raw = self._settings.self_consistency_types or ""
        return {t.strip() for t in raw.split(",") if t.strip()}

    def boost(
        self,
        question: str,
        answer: str,
        query_type: str = "",
        sample_fn: Optional[Callable[[], str]] = None,
    ) -> BoostResult:
        """增强答案：先零LLM抽取短答案；满足条件再做自一致性投票。

        Args:
            sample_fn: 采样生成函数（temperature>0），返回一次完整回答；
                       自一致性需要时调用，未提供则跳过自一致性。
        """
        result = BoostResult()
        if not self._settings.use_answer_extraction:
            result.short_answer = ""
        else:
            try:
                result.short_answer = extract_short_answer(question, answer)
            except Exception as e:
                logger.warning(f"F10 短答案抽取失败: {e}")
                result.short_answer = ""

        # 自适应自一致性：默认关；仅短答案型查询触发
        if not self._settings.use_self_consistency:
            return result
        if query_type not in self._enabled_types():
            return result
        if sample_fn is None:
            return result

        n = max(1, self._settings.self_consistency_samples)
        shorts: List[str] = []
        try:
            for _ in range(n):
                sampled = sample_fn()
                shorts.append(extract_short_answer(question, sampled))
            result.samples = n
            result.self_consistency_used = True
            voted = self_consistency_vote(shorts)
            result.voted_answer = voted
            if voted:
                result.short_answer = voted
            logger.info(f"F10 自一致性: {n} 采样 -> '{voted}' (votes={dict(Counter(shorts))})")
        except Exception as e:
            logger.warning(f"F10 自一致性失败，保留抽取结果: {e}")
        return result
