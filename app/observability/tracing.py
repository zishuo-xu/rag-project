"""轻量级分布式追踪 - 记录 RAG 管道每个阶段的耗时与上下文"""

import time
import uuid
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Span:
    """单个管道阶段的记录"""
    name: str
    start_ms: float = 0          # 相对 trace 起始的偏移 (ms)
    duration_ms: float = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 1),
            "duration_ms": round(self.duration_ms, 1),
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """一次完整 RAG 调用的追踪记录"""
    trace_id: str
    question: str
    timestamp: float
    spans: list = field(default_factory=list)
    total_ms: float = 0
    answer_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "timestamp": self.timestamp,
            "total_ms": round(self.total_ms, 1),
            "answer_preview": self.answer_preview,
            "spans": [s.to_dict() for s in self.spans],
        }


class Tracer:
    """
    内置追踪器（线程安全）。

    记录最近 N 次 RAG 调用的完整管道耗时瀑布图：
    query_transform → dense_retrieval → sparse_retrieval → rrf_fusion → rerank → generation
    """

    def __init__(self, max_traces: int = 50):
        self._traces: deque[Trace] = deque(maxlen=max_traces)
        self._active: dict[str, Trace] = {}
        self._span_starts: dict[str, float] = {}
        self._lock = threading.Lock()

    def start_trace(self, question: str) -> str:
        """开始一次追踪，返回 trace_id"""
        trace_id = uuid.uuid4().hex[:12]
        trace = Trace(
            trace_id=trace_id,
            question=question[:200],
            timestamp=time.time(),
        )
        with self._lock:
            self._active[trace_id] = trace
            self._span_starts[trace_id] = time.time()
        return trace_id

    def start_span(self, trace_id: str, name: str):
        """记录 span 开始"""
        with self._lock:
            self._span_starts[f"{trace_id}:{name}"] = time.time()

    def end_span(self, trace_id: str, name: str, metadata: Optional[dict] = None):
        """记录 span 结束并计算耗时"""
        key = f"{trace_id}:{name}"
        now = time.time()
        with self._lock:
            trace = self._active.get(trace_id)
            span_start = self._span_starts.pop(key, None)
            if trace is None or span_start is None:
                return
            trace_origin = self._span_starts.get(trace_id, span_start)
            span = Span(
                name=name,
                start_ms=(span_start - trace_origin) * 1000,
                duration_ms=(now - span_start) * 1000,
                metadata=metadata or {},
            )
            trace.spans.append(span)

    def end_trace(self, trace_id: str, answer_preview: str = ""):
        """结束追踪，归档。幂等：重复调用（如 _finalize 与兜底 finally）为 no-op。

        同时清理该 trace 遗留的 span 起始键（start_span 后未 end_span 即异常的
        阶段），否则失败请求会永久泄漏 `_span_starts` 条目。
        """
        now = time.time()
        with self._lock:
            trace = self._active.pop(trace_id, None)
            origin = self._span_starts.pop(trace_id, None)
            # 清理 "{trace_id}:{span}" 遗留键（异常路径未 end_span 的阶段）
            leftover = [k for k in self._span_starts if k.startswith(f"{trace_id}:")]
            for k in leftover:
                self._span_starts.pop(k, None)
            if trace is None:
                return
            trace.total_ms = (now - (origin or now)) * 1000
            trace.answer_preview = answer_preview[:150]
            self._traces.appendleft(trace)

    def get_traces(self, limit: int = 20) -> list[dict]:
        """获取最近的追踪记录"""
        with self._lock:
            return [t.to_dict() for t in list(self._traces)[:limit]]

    def get_stats(self) -> dict:
        """汇总统计：各阶段平均耗时"""
        with self._lock:
            traces = list(self._traces)
        if not traces:
            return {"total_traces": 0, "avg_total_ms": 0, "stage_avg": {}}

        stage_totals: dict[str, list[float]] = {}
        for t in traces:
            for s in t.spans:
                stage_totals.setdefault(s.name, []).append(s.duration_ms)

        stage_avg = {
            name: round(sum(vals) / len(vals), 1)
            for name, vals in stage_totals.items()
        }
        return {
            "total_traces": len(traces),
            "avg_total_ms": round(sum(t.total_ms for t in traces) / len(traces), 1),
            "stage_avg": stage_avg,
        }


# 全局单例
_tracer = Tracer()


def get_tracer() -> Tracer:
    """获取全局 Tracer 实例"""
    return _tracer
