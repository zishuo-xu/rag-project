"""F12 多轮对话记忆 / 历史感知查询重写 - 解析指代与省略，构造自包含查询

多轮追问常含指代/省略（"它的原理呢？""那区别呢？"），直接检索会丢上下文。
本模块用对话历史把当前问题重写为自包含查询：

- 零 LLM 启发式（默认）：检测指代词/省略型追问，用最近一轮的主题词回填。
- 可选 LLM 重写（history_rewrite_use_llm）：一次小调用做指代消解，失败回退启发式。

时延：启发式零 LLM；LLM 路径默认关闭。异常一律返回原问题（优雅降级）。
"""

import logging
import re
from typing import List, Optional, Tuple

from config import get_settings, build_chat_llm

logger = logging.getLogger(__name__)

# 指代词（含"的"后缀优先匹配，避免只替换半截）
_PRONOUNS = ["它的", "它们的", "这个", "那个", "前者", "后者", "上面的", "刚才的", "它", "这", "那"]
# 省略型追问起始连接词
_ELLIPSIS_START = ("那", "还有", "另外", "再说", "又", "再")
# 主题抽取时需剔除的疑问/客套词
_QUESTION_WORDS = [
    "什么是", "是什么", "请问", "如何", "怎么", "怎样", "为什么", "哪些",
    "介绍", "讲讲", "说说", "解释", "告诉我", "想知道", "了解",
]


def _msg_role_content(msg) -> Tuple[str, str]:
    """统一取 (role, content)，兼容 dict 与 LangChain Message。"""
    if isinstance(msg, dict):
        return msg.get("role", ""), msg.get("content", "")
    return getattr(msg, "type", ""), getattr(msg, "content", "")


def needs_rewrite(question: str) -> bool:
    """判断当前问题是否依赖上下文（含指代词或省略型追问）。"""
    q = (question or "").strip()
    if not q:
        return False
    if any(p in q for p in _PRONOUNS):
        return True
    # 省略型：以连接词开头且较短，或以"呢"结尾的追问
    if q.endswith("呢") or q.endswith("呢?") or q.endswith("呢？"):
        return True
    if q.startswith(_ELLIPSIS_START) and len(q) <= 15:
        return True
    return False


def extract_topic(text: str) -> str:
    """从一轮用户问题中抽取主题词：剔除疑问/客套词与标点。"""
    t = text or ""
    for w in _QUESTION_WORDS:
        t = t.replace(w, "")
    t = re.sub(r"[?？。！!，,、\s的了吗呢]", "", t)
    return t.strip()


class ConversationRewriter:
    """历史感知查询重写器。"""

    def __init__(self, llm=None, settings=None):
        self._settings = settings or get_settings()
        self.llm = llm  # 仅在 history_rewrite_use_llm 时使用

    def rewrite(self, question: str, history: Optional[List]) -> str:
        """把当前问题重写为自包含查询。无需重写/异常时返回原问题。"""
        if not self._settings.use_history_rewrite:
            return question
        if not history or not needs_rewrite(question):
            return question

        topic = self._recent_topic(history)
        if not topic:
            return question

        # 可选 LLM 指代消解；失败回退启发式
        if self._settings.history_rewrite_use_llm and self.llm is not None:
            llm_out = self._llm_rewrite(question, history, topic)
            if llm_out:
                return llm_out

        return self._heuristic_rewrite(question, topic)

    def _recent_topic(self, history: List) -> str:
        """从最近 max_turns 轮中找最后一条用户消息的主题。"""
        max_turns = self._settings.history_rewrite_max_turns
        recent = history[-max_turns * 2:] if max_turns > 0 else history
        for msg in reversed(recent):
            role, content = _msg_role_content(msg)
            if role in ("human", "user") and content:
                topic = extract_topic(content)
                if topic:
                    return topic
        return ""

    @staticmethod
    def _heuristic_rewrite(question: str, topic: str) -> str:
        """指代词→主题回填；无指代词的省略型追问→前置主题。"""
        for p in _PRONOUNS:
            if p in question:
                rewritten = question.replace(p, topic, 1)
                logger.info(f"F12 查询重写(指代): '{question}' -> '{rewritten}'")
                return rewritten
        rewritten = f"{topic}的{question.lstrip('那还有另外再又')}"
        logger.info(f"F12 查询重写(省略): '{question}' -> '{rewritten}'")
        return rewritten

    def _llm_rewrite(self, question: str, history: List, topic: str) -> str:
        """LLM 指代消解；任何异常返回空串（触发启发式回退）。"""
        try:
            llm = self.llm
            if llm is True:  # 占位：允许注入 True 时按配置自建
                llm = build_chat_llm(max_tokens=64, timeout=15, retries=1)
            hist_text = "\n".join(
                f"{_msg_role_content(m)[0]}: {_msg_role_content(m)[1]}"
                for m in history[-4:]
            )
            prompt = (
                "把下面的追问改写成不依赖上下文的独立问题，只输出改写后的问题，不要解释。\n"
                f"对话历史:\n{hist_text}\n当前追问: {question}\n改写:"
            )
            out = llm.invoke(prompt)
            content = getattr(out, "content", str(out)).strip()
            if content:
                logger.info(f"F12 查询重写(LLM): '{question}' -> '{content}'")
                return content
        except Exception as e:
            logger.warning(f"F12 LLM 重写失败，回退启发式: {e}")
        return ""
