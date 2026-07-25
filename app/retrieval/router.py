"""查询路由 / 类型自适应 - 规则驱动（零 LLM，确定可测）

动机：
    不同查询类型适合不同的检索策略。现状对所有查询用同一套 top_k/降噪强度。
    本模块把既有的「数字型特判」（CRAG.validate_numeric_answer 的雏形）泛化为
    一个 principled 的查询路由器。

设计取舍：
    路由只调整「检索深度(top_k)」与「降噪强度(autocut_min_docs)」，
    **不削减召回路**——避免漏召回（召回宁可多，截断阶段再降噪）。

查询类型与优先级（先匹配先返回）：
    multi_hop > numeric > comparative > conceptual > factual(默认)
    （multi_hop 最优先：被 numeric 截胡会导致 F6b 分解不触发——2026-07-25 实测教训）
"""

import re
from dataclasses import dataclass

from config import get_settings

# 数字型：询问时间/数量/年份（精确答案，宜收紧降噪）
_NUMERIC_PATTERNS = [
    r"什么时候|哪一年|哪年|何时|多少|几个|几年|第几|\d+年",
    r"when\b|what year|how many|how much",
]


def is_numeric_question(q: str) -> bool:
    """数字型问题判定（全管道唯一定义：router / crag / answer_boost 共用）。"""
    return any(re.search(p, q or "", re.IGNORECASE) for p in _NUMERIC_PATTERNS)
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
# 多跳型疑问词（含裸「哪」：哪支球队/哪国人 等指代第二事实的疑问）
_MULTIHOP_QWORD = re.compile(r"是什么|是谁|是哪|什么|谁|哪里|何时|哪")
# 多跳型并行标记：两实体并列提问（A和B分别/各自…）
_MULTIHOP_PARALLEL = re.compile(r"分别|各自")
# 多跳型指代链：X 的那个/那位 Y（第一跳定位实体，第二跳问属性）
_MULTIHOP_PRONOUN_CHAIN = re.compile(r"的那个|的那位|的这位")
# 多跳型先后比较：先…还是…先
_MULTIHOP_ORDER = re.compile(r"先.{1,8}(?:还是|先)")
# 多跳型时间序列：…（哪里/何时）之后/以前…（两事实时序拼接）
_MULTIHOP_TEMPORAL = re.compile(r"(?:之后|以前|之前)")
# 多跳型序数/唯一指代：第一个/唯一/最早…哪…（先定位序数实体，再问其属性）
_MULTIHOP_ORDINAL = re.compile(r"(?:第一个|首次|首座|唯一|仅有|最早|最后|最大|最小|最高).{0,12}哪")
# 多跳型角色指代：饰演/扮演的…哪…（演员→角色→角色属性两跳）
_MULTIHOP_ROLE = re.compile(r"(?:饰演|扮演)的.{0,10}哪")
# 多跳型排他计数：除…外还有…（排除一项再问其余，自带双事实语义）
_MULTIHOP_EXCLUSION = re.compile(r"除.{1,20}外还")
# 第二事实标记：数字型关系链提问需此标记才判多跳（否则是单事实所有格链）
_SECOND_FACT_MARKER = re.compile(r"还|也|并|同时|以及")


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

        # 优先级：multi_hop > numeric > comparative > conceptual > factual
        # multi_hop 最优先：多跳问题需聚合更多证据（top_k 上调 + 触发 F6b 分解），
        # 被 numeric 截胡会导致分解永远不触发（2026-07-25 评估闭环实测 15 条仅判中 1 条）。
        if self._is_multi_hop(q):
            return RoutingDecision(
                "multi_hop",
                top_k=self._settings.retrieval_top_k + 3,
                autocut_min_docs=3,
                reason="多跳型：需聚合更多证据",
            )
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
        if self._is_conceptual(q):
            return RoutingDecision(
                "conceptual", reason="概念型：语义理解为主，默认策略",
            )
        return RoutingDecision("factual", reason="事实型：默认策略")

    # ---- 类型判定 ----

    @staticmethod
    def _is_numeric(q: str) -> bool:
        return is_numeric_question(q)

    @staticmethod
    def _is_comparative(q: str) -> bool:
        return bool(_COMPARATIVE_PATTERN.search(q))

    @staticmethod
    def _is_multi_hop(q: str) -> bool:
        """多跳判定（七类信号 + 双疑问词规则）：

        ① 关系链：两个「的」在 15 字内（X的Y的Z），距离由 10 放宽——
           「出道单曲被用作哪部动画的片头曲」为 11 字，原规则漏判
        ② 并行标记：A 和 B 分别/各自…
        ③ 指代链：X 的那个/那位 Y
        ④ 先后比较：先…还是…先（自带双事实语义，**无需疑问词**）
        ⑤ 时间序列：…之后/以前…（如「攻克哪里之后颁行」）
        ⑥ 序数/唯一指代：第一个/唯一/最早…哪…（mh10「第一个帽子戏法…哪支球队」）
        ⑦ 排他计数：除…外还有…（自带双事实语义，**无需疑问词**）
        ⑧ 双疑问词：哪…哪…/什么…什么…（一句话问两个事实）

        防误判（CMRC 回归防线）：
        - 除 ④⑦ 外均需疑问词佐证
        - 数字型 + 关系链 的单事实所有格提问（「《X的Y》是哪一年…」）
          需第二事实标记（还/也/并/同时/以及）才判多跳，否则落回 numeric——
          mh2「…那一年，他还兼任了哪个…」有「还」仍判多跳，CMRC 单事实不误伤
        """
        # ④⑦ 自带双事实语义，独立判定
        if _MULTIHOP_ORDER.search(q):
            return True
        if _MULTIHOP_EXCLUSION.search(q):
            return True
        # ⑧ 双疑问词 = 一句话问两个事实
        if len(_MULTIHOP_QWORD.findall(q)) >= 2:
            return True
        # 其余信号均需疑问词佐证，防误判
        if not _MULTIHOP_QWORD.search(q):
            return False
        if re.search(r"的.{1,15}的", q):
            # 数字型所有格链：无第二事实标记 → 单事实，落回 numeric
            if not (QueryRouter._is_numeric(q) and not _SECOND_FACT_MARKER.search(q)):
                return True
        if _MULTIHOP_PARALLEL.search(q):
            return True
        if _MULTIHOP_PRONOUN_CHAIN.search(q):
            return True
        if _MULTIHOP_TEMPORAL.search(q):
            return True
        if _MULTIHOP_ORDINAL.search(q):
            return True
        if _MULTIHOP_ROLE.search(q):
            return True
        return False

    @staticmethod
    def _is_conceptual(q: str) -> bool:
        return bool(_CONCEPTUAL_PATTERN.search(q))
