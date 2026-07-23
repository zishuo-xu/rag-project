"""F11 生产加固 - API Key 鉴权 + 限流（令牌桶/固定窗口）

- API Key 鉴权：api_key 配置为空则关闭；非空校验 X-API-Key 头。
  健康检查/指标端点豁免（供探针与监控）。
- 限流：按客户端固定窗口（每分钟）计数，超 rate_limit_rpm 返回 429；
  rate_limit_rpm=0 关闭。O(1) 判定，自动清理过期窗口。

时延：均为内存 O(1) 操作，开销可忽略。
"""

import logging
import threading
import time
from typing import Dict, Optional

from fastapi import HTTPException, Request

from config import get_settings

logger = logging.getLogger(__name__)

# 鉴权豁免路径（探针/监控）
_EXEMPT_PREFIXES = ("/api/health", "/api/metrics", "/docs", "/openapi.json", "/redoc")


def is_exempt(path: str) -> bool:
    return any(path.startswith(p) for p in _EXEMPT_PREFIXES)


def verify_api_key(provided: Optional[str], expected: Optional[str] = None) -> bool:
    """校验 API Key。expected 为空表示关闭鉴权（放行）。"""
    if expected is None:
        expected = get_settings().api_key
    if not expected:  # 未配置 → 关闭鉴权
        return True
    return provided == expected


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


def enforce_security(request: Request) -> None:
    """FastAPI 依赖：先鉴权后限流。失败抛 HTTPException。豁免路径直接放行。"""
    if is_exempt(request.url.path):
        return
    # 鉴权
    if not verify_api_key(request.headers.get("X-API-Key")):
        raise HTTPException(status_code=401, detail="无效或缺失的 API Key")
    # 限流
    client = request.client.host if request.client else "unknown"
    if not get_rate_limiter().allow(client):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")
