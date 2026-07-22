"""LLM 思考模式开关测试"""
from config import Settings, get_llm_extra_body


def test_thinking_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_THINKING_ENABLED", raising=False)
    get_settings_cache_clear()
    assert get_llm_extra_body() == {"thinking": {"type": "disabled"}}


def test_thinking_enabled(monkeypatch):
    monkeypatch.setenv("LLM_THINKING_ENABLED", "true")
    get_settings_cache_clear()
    assert get_llm_extra_body() is None


def get_settings_cache_clear():
    from config import get_settings
    get_settings.cache_clear()
