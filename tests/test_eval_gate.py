"""eval-as-gate 判定层单测（纯逻辑，零 LLM，构造 summary dict 验证边界）。"""
from app.evaluation.gate import evaluate_gate, _get


def _good_summary():
    """全达标的 summary（smoke 鲁棒集 + full 端到端键齐全）。"""
    return {
        "num_samples": 8, "num_ok": 8, "num_failed": 0,
        "retrieval": {"hit_rate": 1.0, "avg_coverage": 1.0},
        "end_to_end": {
            "avg_f1": 0.40, "em_rate": 0.0, "hit_rate": 0.50,
            "answer_in_top_context_rate": 0.80,
        },
        "avg_latency_ms": 5000.0,
    }


def test_get_nested_path():
    s = _good_summary()
    assert _get(s, "retrieval.hit_rate") == 1.0
    assert _get(s, "end_to_end.avg_f1") == 0.40
    assert _get(s, "missing.key") is None
    assert _get(None, "x") is None


def test_gate_passes_when_above_floors():
    res = evaluate_gate(_good_summary(), mode="smoke")
    assert res.passed is True
    assert res.violations == []


def test_gate_fails_on_absolute_floor():
    s = _good_summary()
    s["retrieval"]["hit_rate"] = 0.5  # 低于 min 0.8
    res = evaluate_gate(s, mode="smoke")
    assert res.passed is False
    metrics = {v["metric"] for v in res.violations}
    assert "retrieval.hit_rate" in metrics


def test_gate_latency_ceiling():
    s = _good_summary()
    s["avg_latency_ms"] = 60000.0  # 超 max 30000
    res = evaluate_gate(s, mode="smoke")
    assert res.passed is False
    assert any(v["metric"] == "avg_latency_ms" for v in res.violations)


def test_gate_full_detects_regression_vs_baseline():
    base = _good_summary()
    cur = _good_summary()
    cur["end_to_end"]["avg_f1"] = 0.20  # base 0.40，delta -0.20 < -tol(0.05)
    res = evaluate_gate(cur, mode="full", baseline=base)
    assert res.passed is False
    viol = {v["metric"] for v in res.violations}
    assert "end_to_end.avg_f1" in viol


def test_gate_tolerance_absorbs_jitter():
    """下降在容差内 → 不判 violation（至多 warning）。"""
    base = _good_summary()
    cur = _good_summary()
    cur["end_to_end"]["avg_f1"] = 0.33  # base 0.40，delta -0.07，tol 0.05 → 超？
    # -0.07 < -0.05 会 violation，改用 -0.03 落在容差内
    cur["end_to_end"]["avg_f1"] = 0.37  # delta -0.03，在 [-0.05, 0) → warning
    res = evaluate_gate(cur, mode="full", baseline=base)
    assert res.passed is True  # 轻降不阻断
    assert any(w["metric"] == "end_to_end.avg_f1" for w in res.warnings)


def test_gate_baseline_missing_key_skipped():
    """基线缺某端到端键 → 该指标退化比对 skip（info），不报错。"""
    base = _good_summary()
    del base["end_to_end"]["avg_f1"]  # 基线缺 avg_f1
    cur = _good_summary()
    res = evaluate_gate(cur, mode="full", baseline=base)
    # 不因基线缺键而 fail
    assert not any(v["metric"] == "end_to_end.avg_f1" for v in res.violations)
    assert any(i["metric"] == "end_to_end.avg_f1" for i in res.infos)


def test_gate_summary_missing_robust_metric_is_violation():
    """smoke 鲁棒键缺失 → violation（harness 坏信号）。"""
    s = _good_summary()
    del s["end_to_end"]["answer_in_top_context_rate"]
    res = evaluate_gate(s, mode="smoke")
    assert res.passed is False
    assert any(
        v["metric"] == "end_to_end.answer_in_top_context_rate" for v in res.violations
    )


def test_gate_chain_health():
    """num_ok < num_samples → 链路不健康 violation。"""
    s = _good_summary()
    s["num_ok"] = 6  # 8 题里 2 题抛异常
    res = evaluate_gate(s, mode="smoke")
    assert res.passed is False
    assert any(v["metric"] == "chain_health" for v in res.violations)


def test_gate_full_without_baseline_skips_e2e_compare():
    """full 模式但无基线 → 端到端退化比对 skip（info），鲁棒集仍判。"""
    res = evaluate_gate(_good_summary(), mode="full", baseline=None)
    assert res.passed is True
    assert any(i["metric"] == "end_to_end.avg_f1" for i in res.infos)
