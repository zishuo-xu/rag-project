"""Tracer 测试（线程安全追踪器，含失败路径的泄漏防护）"""
from app.observability.tracing import Tracer


def test_trace_lifecycle_and_span_offsets():
    t = Tracer()
    tid = t.start_trace("问题")
    t.start_span(tid, "retrieval")
    t.end_span(tid, "retrieval", {"docs": 5})
    t.end_trace(tid, answer_preview="答案")

    traces = t.get_traces()
    assert len(traces) == 1
    assert traces[0]["question"] == "问题"
    assert traces[0]["answer_preview"] == "答案"
    spans = traces[0]["spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == "retrieval"
    assert spans[0]["metadata"] == {"docs": 5}
    assert spans[0]["start_ms"] >= 0


def test_end_trace_idempotent():
    """重复 end_trace 为 no-op（_finalize 与兜底 finally 都会调用）。"""
    t = Tracer()
    tid = t.start_trace("问题")
    t.end_trace(tid)
    t.end_trace(tid)  # 第二次不应产生第二条归档
    assert len(t.get_traces()) == 1
    assert len(t._active) == 0


def test_end_trace_cleans_leaked_span_keys():
    """start_span 后未 end_span 即异常 → end_trace 清理遗留键，不泄漏内存。"""
    t = Tracer()
    tid = t.start_trace("问题")
    t.start_span(tid, "generation")  # 模拟异常路径：从未 end_span
    assert f"{tid}:generation" in t._span_starts
    t.end_trace(tid)
    assert f"{tid}:generation" not in t._span_starts
    assert tid not in t._span_starts


def test_failed_trace_still_archived():
    """失败请求也要留痕（最需要诊断的场景不能静默丢失）。"""
    t = Tracer()
    tid = t.start_trace("会失败的问题")
    t.start_span(tid, "retrieval")
    # 模拟检索异常后兜底 finally 直接 end_trace（未 end_span）
    t.end_trace(tid)
    traces = t.get_traces()
    assert len(traces) == 1
    assert traces[0]["question"] == "会失败的问题"


def test_max_traces_ring_buffer():
    t = Tracer(max_traces=3)
    for i in range(5):
        tid = t.start_trace(f"q{i}")
        t.end_trace(tid)
    traces = t.get_traces()
    assert len(traces) == 3
    assert traces[0]["question"] == "q4"  # 最新在前


def test_stats_stage_averages():
    t = Tracer()
    for _ in range(2):
        tid = t.start_trace("q")
        t.start_span(tid, "rerank")
        t.end_span(tid, "rerank")
        t.end_trace(tid)
    stats = t.get_stats()
    assert stats["total_traces"] == 2
    assert "rerank" in stats["stage_avg"]
