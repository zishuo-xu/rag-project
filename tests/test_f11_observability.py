"""F11 可观测性与生产加固测试（离线）

覆盖：
1. MetricsRegistry：计数器累加/标签区分、直方图统计与百分位、Prometheus/JSON 导出、reset
2. security.verify_api_key：关闭/正确/错误/缺 key
3. RateLimiter：关闭/窗口内放行/超限拒绝/跨窗口重置/清理
4. check_request（生产中间件 main.py SecurityMiddleware 共用的判定实现）：
   探针豁免/文档豁免随鉴权开关联动/鉴权失败 401/限流 429/正确 key 放行
"""
from unittest.mock import MagicMock

from app.observability.metrics import MetricsRegistry
from app.api.security import (
    verify_api_key, RateLimiter, is_exempt, check_request,
)


# ============ MetricsRegistry ============

def test_counter_accumulates():
    m = MetricsRegistry()
    m.inc("requests_total", endpoint="chat")
    m.inc("requests_total", endpoint="chat")
    assert m.get_counter("requests_total", endpoint="chat") == 2.0


def test_counter_labels_distinguished():
    m = MetricsRegistry()
    m.inc("req", status="200")
    m.inc("req", status="500")
    assert m.get_counter("req", status="200") == 1.0
    assert m.get_counter("req", status="500") == 1.0


def test_counter_default_zero():
    m = MetricsRegistry()
    assert m.get_counter("missing") == 0.0


def test_histogram_stats_and_percentiles():
    m = MetricsRegistry()
    for v in range(1, 101):  # 1..100
        m.observe("latency", v)
    snap = m.snapshot()["histograms"]["latency"]
    assert snap["count"] == 100
    assert snap["min"] == 1
    assert snap["max"] == 100
    assert 50 <= snap["p50"] <= 51
    assert 95 <= snap["p95"] <= 96


def test_snapshot_json_structure():
    m = MetricsRegistry()
    m.inc("c", a="1")
    m.observe("h", 10)
    snap = m.snapshot()
    assert "counters" in snap and "histograms" in snap
    assert snap["counters"]["c{a=1}"] == 1.0


def test_prometheus_format():
    m = MetricsRegistry()
    m.inc("requests_total", endpoint="chat")
    m.observe("latency_ms", 12.5)
    text = m.prometheus()
    assert 'requests_total{endpoint="chat"} 1.0' in text
    assert "latency_ms_count 1" in text


def test_reset_clears():
    m = MetricsRegistry()
    m.inc("c")
    m.observe("h", 1)
    m.reset()
    assert m.get_counter("c") == 0.0
    assert m.snapshot()["histograms"] == {}


# ============ verify_api_key ============

def test_auth_disabled_when_empty():
    assert verify_api_key(None, expected="") is True
    assert verify_api_key("anything", expected="") is True


def test_auth_correct_key():
    assert verify_api_key("secret", expected="secret") is True


def test_auth_wrong_key():
    assert verify_api_key("wrong", expected="secret") is False
    assert verify_api_key(None, expected="secret") is False


# ============ RateLimiter ============

def test_rate_limit_disabled():
    rl = RateLimiter(rpm=0)
    for _ in range(100):
        assert rl.allow("c1") is True


def test_rate_limit_allows_within_limit():
    rl = RateLimiter(rpm=3)
    now = 1000.0
    assert rl.allow("c1", now) is True
    assert rl.allow("c1", now) is True
    assert rl.allow("c1", now) is True


def test_rate_limit_blocks_over_limit():
    rl = RateLimiter(rpm=2)
    now = 1000.0
    assert rl.allow("c1", now) is True
    assert rl.allow("c1", now) is True
    assert rl.allow("c1", now) is False  # 超限


def test_rate_limit_window_resets():
    rl = RateLimiter(rpm=1)
    assert rl.allow("c1", now=1000.0) is True
    assert rl.allow("c1", now=1000.0) is False
    # 进入下一分钟窗口 → 重置
    assert rl.allow("c1", now=1060.0) is True


def test_rate_limit_per_client_isolated():
    rl = RateLimiter(rpm=1)
    now = 1000.0
    assert rl.allow("c1", now) is True
    assert rl.allow("c2", now) is True  # 不同客户端独立计数
    assert rl.allow("c1", now) is False


# ============ check_request（生产中间件判定） ============

def _request(path, api_key=None, host="1.2.3.4"):
    req = MagicMock()
    req.url.path = path
    req.headers = {"X-API-Key": api_key} if api_key else {}
    req.client.host = host
    return req


def _settings(api_key="", rate_limit_rpm=0):
    return MagicMock(api_key=api_key, rate_limit_rpm=rate_limit_rpm)


def test_probe_path_always_exempt(monkeypatch):
    """探针/监控端点即使开启鉴权也放行（否则健康检查被 401）。"""
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key="secret"))
    assert check_request(_request("/api/health")) is None
    assert check_request(_request("/api/metrics")) is None


def test_docs_exempt_only_when_auth_off(monkeypatch):
    """API 文档：鉴权关闭时豁免，开启后即关门（不泄露 API 表面）。"""
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key=""))
    assert is_exempt("/docs") is True
    assert is_exempt("/openapi.json") is True
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key="secret"))
    assert is_exempt("/docs") is False
    assert is_exempt("/openapi.json") is False
    assert is_exempt("/api/chat") is False  # 任何情况下业务端点不豁免


def test_check_auth_failure_returns_401(monkeypatch):
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key="secret"))
    resp = check_request(_request("/api/chat", api_key="wrong"))
    assert resp is not None and resp.status_code == 401
    # 缺 key 头同样 401
    resp2 = check_request(_request("/api/chat"))
    assert resp2 is not None and resp2.status_code == 401


def test_check_correct_key_passes(monkeypatch):
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key="secret", rate_limit_rpm=0))
    assert check_request(_request("/api/chat", api_key="secret")) is None


def test_check_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr("app.api.security.get_settings",
                        lambda: _settings(api_key="", rate_limit_rpm=1))
    limiter = RateLimiter(rpm=1)  # 共享同一实例，计数才累积
    monkeypatch.setattr("app.api.security.get_rate_limiter", lambda: limiter)
    req = _request("/api/chat", host="9.9.9.9")
    assert check_request(req) is None  # 第一次放行
    resp = check_request(req)          # 第二次超限
    assert resp is not None and resp.status_code == 429


def test_verify_api_key_missing_provided():
    """配置了 key 但未提供 → 拒绝（而非 compare_digest 对 None 崩溃）。"""
    assert verify_api_key(None, expected="secret") is False
    assert verify_api_key("", expected="secret") is False
    assert verify_api_key("secret", expected="secret") is True
    assert verify_api_key("anything", expected="") is True  # 未配置 → 关闭鉴权
