"""F12 多轮对话记忆 / 历史感知查询重写测试（离线，LLM mock）

覆盖：
1. needs_rewrite：指代词 / 省略型 / 独立完整问题
2. extract_topic：剔除疑问词与标点
3. ConversationRewriter.rewrite：指代回填 / 省略前置 / 无需重写 / 关闭开关 /
   无历史 / LLM 路径 / LLM 失败回退启发式
"""
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from app.retrieval.conversation import (
    needs_rewrite, extract_topic, ConversationRewriter,
)


def _settings(rewrite=True, use_llm=False, max_turns=4):
    s = MagicMock()
    s.use_history_rewrite = rewrite
    s.history_rewrite_use_llm = use_llm
    s.history_rewrite_max_turns = max_turns
    return s


# ============ needs_rewrite ============

def test_needs_rewrite_pronoun():
    assert needs_rewrite("它的原理是什么？") is True
    assert needs_rewrite("这个怎么解决？") is True


def test_needs_rewrite_ellipsis():
    assert needs_rewrite("那区别呢？") is True
    assert needs_rewrite("还有别的方案吗") is True


def test_needs_rewrite_complete_question():
    assert needs_rewrite("什么是缓存穿透？") is False
    assert needs_rewrite("") is False


# ============ extract_topic ============

def test_extract_topic_removes_question_words():
    assert extract_topic("什么是缓存穿透？") == "缓存穿透"


def test_extract_topic_removes_punct():
    assert extract_topic("Redis 的 ZSet 原理？") == "RedisZSet原理"


def test_extract_topic_how_to():
    assert extract_topic("如何解决缓存击穿？") == "解决缓存击穿"


# ============ rewrite（dict 历史） ============

def test_rewrite_pronoun_backfill():
    r = ConversationRewriter(settings=_settings())
    history = [{"role": "user", "content": "什么是缓存穿透？"},
               {"role": "assistant", "content": "缓存穿透是..."}]
    out = r.rewrite("它的原理呢？", history)
    assert "缓存穿透" in out
    assert "它" not in out


def test_rewrite_ellipsis_prefix_topic():
    r = ConversationRewriter(settings=_settings())
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    out = r.rewrite("那解决方案呢？", history)
    assert "缓存穿透" in out


def test_rewrite_no_history_returns_original():
    r = ConversationRewriter(settings=_settings())
    assert r.rewrite("它的原理呢？", None) == "它的原理呢？"
    assert r.rewrite("它的原理呢？", []) == "它的原理呢？"


def test_rewrite_complete_question_untouched():
    r = ConversationRewriter(settings=_settings())
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    q = "Redis支持哪些数据结构？"
    assert r.rewrite(q, history) == q


def test_rewrite_disabled_returns_original():
    r = ConversationRewriter(settings=_settings(rewrite=False))
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    assert r.rewrite("它的原理呢？", history) == "它的原理呢？"


def test_rewrite_no_user_topic_returns_original():
    r = ConversationRewriter(settings=_settings())
    history = [{"role": "assistant", "content": "只有助手消息"}]
    assert r.rewrite("它的原理呢？", history) == "它的原理呢？"


# ============ rewrite（LangChain Message 历史） ============

def test_rewrite_langchain_messages():
    r = ConversationRewriter(settings=_settings())
    history = [HumanMessage(content="什么是缓存穿透？"),
               AIMessage(content="缓存穿透是...")]
    out = r.rewrite("它怎么预防？", history)
    assert "缓存穿透" in out


def test_rewrite_respects_max_turns():
    r = ConversationRewriter(settings=_settings(max_turns=1))
    # max_turns=1 → 只看最近 2 条；较早的"什么是缓存穿透？"在窗口外不被采用，
    # 窗口内用户消息为空内容 → 抽不出主题 → 返回原问题
    history = [HumanMessage(content="什么是缓存穿透？"),
               AIMessage(content="..."),
               HumanMessage(content=""),
               AIMessage(content="...")]
    out = r.rewrite("它的原理呢？", history)
    assert out == "它的原理呢？"


# ============ LLM 路径 ============

def test_rewrite_llm_path_used():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="缓存穿透的原理是什么？")
    r = ConversationRewriter(llm=llm, settings=_settings(use_llm=True))
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    out = r.rewrite("它的原理呢？", history)
    assert out == "缓存穿透的原理是什么？"
    llm.invoke.assert_called_once()


def test_rewrite_llm_failure_falls_back_to_heuristic():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("timeout")
    r = ConversationRewriter(llm=llm, settings=_settings(use_llm=True))
    history = [{"role": "user", "content": "什么是缓存穿透？"}]
    out = r.rewrite("它的原理呢？", history)
    assert "缓存穿透" in out  # 启发式回填
    assert "它" not in out
