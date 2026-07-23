"""F11 可观测性 - 进程内指标注册表（计数器 + 直方图，零外部依赖）

生产级 RAG 需要可观测性。本模块提供轻量进程内指标（不引 prometheus 客户端）：
- 计数器 inc(name, **labels)：如 requests_total{endpoint,status}、cache_hits_total{level}。
- 直方图 observe(name, value)：如 request_latency_ms，快照时算 count/avg/min/max/p50/p95。

开销 <1µs/请求（内存原子操作 + 有界样本）。支持 Prometheus 文本与 JSON 两种导出。
"""

import logging
import threading
from collections import deque
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_MAX_SAMPLES = 1000  # 每个直方图保留的最近样本数（有界，防内存膨胀）


def _label_key(labels: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


class MetricsRegistry:
    """线程安全的指标注册表。"""

    def __init__(self):
        self._counters: Dict[Tuple[str, Tuple], float] = {}
        self._histograms: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = (name, _label_key(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            dq = self._histograms.get(name)
            if dq is None:
                dq = deque(maxlen=_MAX_SAMPLES)
                self._histograms[name] = dq
            dq.append(float(value))

    def get_counter(self, name: str, **labels) -> float:
        return self._counters.get((name, _label_key(labels)), 0.0)

    def _percentile(self, sorted_vals: List[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    def snapshot(self) -> dict:
        """JSON 快照：counters + histograms 统计。"""
        with self._lock:
            counters = {}
            for (name, lbls), val in self._counters.items():
                label_str = ",".join(f"{k}={v}" for k, v in lbls)
                key = f"{name}{{{label_str}}}" if label_str else name
                counters[key] = round(val, 4)

            histograms = {}
            for name, dq in self._histograms.items():
                vals = sorted(dq)
                if not vals:
                    continue
                histograms[name] = {
                    "count": len(vals),
                    "avg": round(sum(vals) / len(vals), 3),
                    "min": round(vals[0], 3),
                    "max": round(vals[-1], 3),
                    "p50": round(self._percentile(vals, 50), 3),
                    "p95": round(self._percentile(vals, 95), 3),
                }
        return {"counters": counters, "histograms": histograms}

    def prometheus(self) -> str:
        """Prometheus 文本暴露格式。"""
        lines: List[str] = []
        with self._lock:
            for (name, lbls), val in sorted(self._counters.items()):
                label_str = ",".join(f'{k}="{v}"' for k, v in lbls)
                metric = f"{name}{{{label_str}}}" if label_str else name
                lines.append(f"{metric} {val}")
            for name, dq in sorted(self._histograms.items()):
                vals = sorted(dq)
                if not vals:
                    continue
                lines.append(f"{name}_count {len(vals)}")
                lines.append(f"{name}_sum {round(sum(vals), 3)}")
                lines.append(f"{name}_p50 {round(self._percentile(vals, 50), 3)}")
                lines.append(f"{name}_p95 {round(self._percentile(vals, 95), 3)}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# 全局单例
_registry: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry
