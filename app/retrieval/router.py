"""查询路由 / 类型自适应 - 规则驱动（零 LLM，确定可测）

动机：
    不同查询类型适合不同的检索策略。现状对所有查询用同一套 top_k/降噪强度。
    本模块把既有的「数字型特判」（CRAG.validate_numeric_answer 的雏形）泛化为
    一个 principled 的查询路由器。

设计取舍：
    路由只调整「检索深度(top_k)」与「降噪强度(autocut_min_docs)」，
    **不削减召回路**——避免漏召回（召回宁可多，截断阶段再降噪）。

查询类型与优先级（先匹配先返回）：
    numeric > comparative > multi_hop > conceptual > factual(默认)
"""

import re
from dataclasses import dataclass

from config import get_settings

# 数字型：询问时间/数量/年份（精确答案，宜收紧降噪）
_NUMERIC_PATTERNS = [
    r"什么时候|哪一年|哪年|何时|多少|几个|几年|第几|\d+年",
    r"when\b|what year|how many|how much",
]
# 对比型：需更多候选参与比较
_COMPARATIVE_PATTERN = re.compile(
    r"区别|差异|对比|比较|优劣|优缺点|哪个好|哪种好|有何不同|(?:^|\s)vs(?:\s|$)|VS",
    re.IGNORECASE,
)
# 概念型：语义理解为主
_CONCEPTUAL_PATTERN = re.compile(
    r"什么是|是什么|原理|为什么|如何|怎么|怎样|含义|意思|解释|定义",
    re.IGNORECASE,
)
# 多跳型疑问词
_MULTIHOP_QWORD = re.compile(r"是什么|是谁|是哪|什么|谁|哪里|何时")


@dataclass
class RoutingDecision:
    """路由决策：query_type 用于观测/叙事，top_k/autocut_min_docs 为真实行为参数。

    top_k / autocut_min_docs 为 None 时表示「沿用调用方/配置默认值」。
    """
    query_type: str
    top_k: int | None = None
    autocut_min_docs: int | None = None
    reason: str = ""


class QueryRouter:
    """规则驱动的查询路由器（零 LLM，确定可测）。"""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()

    def route(self, question: str) -> RoutingDecision:
        q = (question or "").strip()
        if not q:
            return RoutingDecision("factual", reason="空查询，默认路由")

        if self._is_numeric(q):
            return RoutingDecision(
                "numeric", autocut_min_docs=1,
                reason="数字型：精确答案，收紧截断降噪",
            )
        if self._is_comparative(q):
            return RoutingDecision(
                "comparative",
                top_k=self._settings.retrieval_top_k + 3,
                autocut_min_docs=3,
                reason="对比型：需更多候选参与对比",
            )
        if self._is_multi_hop(q):
            return RoutingDecision(
                "multi_hop",
                top_k=self._settings.retrieval_top_k + 3,
                autocut_min_docs=3,
                reason="多跳型：需聚合更多证据",
            )
        if self._is_conceptual(q):
            return RoutingDecision(
                "conceptual", reason="概念型：语义理解为主，默认策略",
            )
        return RoutingDecision("factual", reason="事实型：默认策略")

    # ---- 类型判定 ----

    @staticmethod
    def _is_numeric(q: str) -> bool:
        return any(re.search(p, q, re.IGNORECASE) for p in _NUMERIC_PATTERNS)

    @staticmethod
    def _is_comparative(q: str) -> bool:
        return bool(_COMPARATIVE_PATTERN.search(q))

    @staticmethod
    def _is_multi_hop(q: str) -> bool:
        # 关系链：两个「的」近距离出现（X的Y的Z）且带疑问词
        has_chain = re.search(r"的.{1,10}的", q)
        return bool(has_chain and _MULTIHOP_QWORD.search(q))

    @staticmethod
    def _is_conceptual(q: str) -> bool:
        return bool(_CONCEPTUAL_PATTERN.search(q))
