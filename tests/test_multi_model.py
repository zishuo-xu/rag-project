"""多模型选择测试：llm_provider 在 deepseek(原 openai_*) 与 qwen 间切换，互不删除。"""
from config import active_llm_config, get_llm_extra_body


def _clear():
    from config import get_settings
    get_settings.cache_clear()


def test_default_provider_is_deepseek(monkeypatch):
    """代码默认 provider=deepseek（忽略 .env 与 OS env，纯验默认值）。"""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from config import Settings
    assert Settings(_env_file=None).llm_provider == "deepseek"


def test_deepseek_uses_openai_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    _clear()
    model, key, base = active_llm_config()
    assert model == "deepseek-chat"
    assert base == "https://api.deepseek.com/v1"


def test_qwen_uses_qwen_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_MODEL", "qwen3.8-max-preview")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("QWEN_API_KEY", "sk-test")
    _clear()
    model, key, base = active_llm_config()
    assert model == "qwen3.8-max-preview"
    assert base == "https://example.com/v1"
    assert key == "sk-test"


def test_qwen_extra_body_always_none(monkeypatch):
    """qwen 预览端点 enable_thinking 受限为 True，无法关闭 → 恒不传 extra_body。"""
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.delenv("LLM_THINKING_ENABLED", raising=False)
    _clear()
    assert get_llm_extra_body() is None


def test_deepseek_extra_body_thinking_disabled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_THINKING_ENABLED", raising=False)
    _clear()
    assert get_llm_extra_body() == {"thinking": {"type": "disabled"}}
