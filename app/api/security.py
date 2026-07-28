"""F11 生产加固 - API Key 鉴权 + 限流（固定窗口）

- API Key 鉴权：api_key 配置为空则关闭；非空校验 X-API-Key 头（常时比对）。
  探针/监控端点恒豁免；API 文档仅在鉴权关闭时豁免（关门后不泄露 API 表面）。
- 限流：按客户端固定窗口（每分钟）计数，超 rate_limit_rpm 返回 429；
  rate_limit_rpm=0 关闭。O(1) 判定，自动清理过期窗口。
- 接线：生产中间件在 main.py（SecurityMiddleware 调用本模块函数）。

已知限制：client_id 取 request.client.host，反代部署下所有客户端共享代理 IP
一桶；按真实用户限流需部署侧传 X-Forwarded-For 并在此显式信任（未实现，登记）。

时延：均为内存 O(1) 操作，开销可忽略。
"""

import hmac
import logging
import threading
import time
from typing import Dict, Optional

from config import get_settings

logger = logging.getLogger(__name__)

# 探针/监控端点：恒豁免（健康检查与指标采集不受鉴权影响）
_PROBE_PREFIXES = ("/api/health", "/api/metrics")
# API 文档端点：仅鉴权关闭时豁免（开启鉴权即关门，不再无认证暴露 API schema）
_DOCS_PREFIXES = ("/docs", "/openapi.json", "/redoc")


def is_exempt(path: str) -> bool:
    if any(path.startswith(p) for p in _PROBE_PREFIXES):
        return True
    if any(path.startswith(p) for p in _DOCS_PREFIXES):
        return not get_settings().api_key
    return False


def verify_api_key(provided: Optional[str], expected: Optional[str] = None) -> bool:
    """校验 API Key。expected 为空表示关闭鉴权（放行）。常时比对防计时侧信道。"""
    if expected is None:
        expected = get_settings().api_key
    if not expected:  # 未配置 → 关闭鉴权
        return True
    if not provided:
        return False
    return hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    )


class RateLimiter:
    """固定窗口限流器（按客户端每分钟计数）。"""

    def __init__(self, rpm: Optional[int] = None):
        settings = get_settings()
        self.rpm = rpm if rpm is not None else settings.rate_limit_rpm
        self._buckets: Dict[str, tuple] = {}  # client -> (window_start, count)
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: Optional[float] = None) -> bool:
        """是否放行。rpm<=0 表示关闭限流。"""
        if self.rpm <= 0:
            return True
        now = now if now is not None else time.time()
        window = int(now // 60)
        with self._lock:
            entry = self._buckets.get(client_id)
            if entry is None or entry[0] != window:
                self._buckets[client_id] = (window, 1)
                self._prune(window)
                return True
            count = entry[1]
            if count >= self.rpm:
                return False
            self._buckets[client_id] = (window, count + 1)
            return True

    def _prune(self, current_window: int) -> None:
        """清理非当前窗口的条目，防内存膨胀。"""
        stale = [c for c, (w, _) in self._buckets.items() if w != current_window]
        for c in stale:
            del self._buckets[c]


# 全局限流器
_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def check_request(request) -> Optional["JSONResponse"]:
    """生产中间件检查（main.py SecurityMiddleware 与测试共用这一份实现）。

    豁免路径 / 鉴权失败 / 限流超限分别返回 None / 401 / 429 响应。
    早期这里存在两份实现（本模块的依赖注入版 + main.py 内联版）且已漂移，
    生产走内联版、测试测依赖版——合并为单一事实源。
    """
    from starlette.responses import JSONResponse

    from app.observability.metrics import get_metrics

    if is_exempt(request.url.path):
        return None
    if not verify_api_key(request.headers.get("X-API-Key")):
        get_metrics().inc("errors_total", code="401")
        return JSONResponse(
            status_code=401, content={"detail": "无效或缺失的 API Key"}
        )
    client = request.client.host if request.client else "unknown"
    if not get_rate_limiter().allow(client):
        get_metrics().inc("errors_total", code="429")
        return JSONResponse(
            status_code=429, content={"detail": "请求过于频繁，请稍后重试"}
        )
    return None
