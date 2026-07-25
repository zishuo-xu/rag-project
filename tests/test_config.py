"""配置项测试"""
from unittest.mock import patch

from config import Settings, build_chat_llm


def test_new_pipeline_settings_defaults():
    # _env_file=None: 隔离本地 .env（如 MAX_CONCURRENT_REQUESTS=8），只验证代码默认值
    s = Settings(_env_file=None)
    assert s.use_summary_recall is True
    assert s.use_crag_gate is True
    assert s.recall_max_workers == 6
    assert s.max_concurrent_requests == 4
    assert s.request_queue_timeout == 30.0


# ============ build_chat_llm 工厂契约 ============

def test_build_chat_llm_omits_unset_budgets():
    """None 参数不下传：沿用 ChatOpenAI 默认（保 graph_extractor 等原行为）。"""
    with patch("config.ChatOpenAI") as m:
        build_chat_llm()
        kw = m.call_args.kwargs
        assert kw["temperature"] == 0
        assert kw["streaming"] is False
        assert "request_timeout" not in kw
        assert "max_retries" not in kw
        assert "max_tokens" not in kw


def test_build_chat_llm_passes_explicit_budgets():
    """显式参数逐点下传（延迟治理各点差异不被统一默认抹平）。"""
    with patch("config.ChatOpenAI") as m:
        build_chat_llm(
            temperature=0.7, timeout=20, retries=1, max_tokens=512, streaming=True
        )
        kw = m.call_args.kwargs
        assert kw["temperature"] == 0.7
        assert kw["request_timeout"] == 20
        assert kw["max_retries"] == 1
        assert kw["max_tokens"] == 512
        assert kw["streaming"] is True
