"""eval-as-gate 质量卡点 · 判定层（纯逻辑，零 LLM，可单测）

对 run_e2e_eval 的 aggregate summary 做"退化即 fail"判定，分两套受控指标：

- **smoke 鲁棒集**（每次 push 跑）：检索命中率 / 答案入上下文 / 延迟上限 / 链路健康度。
  这些主要取决于检索+embedding，对生成措辞抖动不敏感，适合小样本卡点。
- **full 端到端集**（发版前手动跑）：在鲁棒集之上，叠加 avg_f1 / hit_rate 的
  **相对基线退化**判定（用容差 tol 吸收 LLM 非确定性抖动，避免误报工厂）。

设计原则：
- 度量函数零 LLM，但 chain.invoke 产生答案要真 LLM —— 真实质量无离线捷径，
  gate 必须跑真链路；本模块只负责"拿到 summary 后判 pass/fail"，不跑链路。
- 基线缺键 → 该指标基线比对 skip（记 info），不报错，兼容旧基线/旧 aggregate。
- summary 缺鲁棒键 → violation（harness 坏信号）；缺 full 端到端键 → warning。
- 阈值/容差为模块常量 + 参数可覆盖；初值由现有 eval_e2e_full.json 反推留余量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── 受控指标表 ────────────────────────────────────────────────
# modes: 该指标在哪些 gate-mode 下受控（"smoke" 表示 smoke+full 都查）
# min_abs / max_abs: 绝对下限/上限（不依赖基线）
# needs_baseline: 是否需要基线做相对退化判定
_METRICS: List[Dict[str, Any]] = [
    # —— smoke 鲁棒集 ——
    {"key": "retrieval.hit_rate", "name": "检索命中率",
     "min_abs": 0.8, "modes": ["smoke", "full"]},
    {"key": "end_to_end.answer_in_top_context_rate", "name": "答案入 top 上下文率",
     "min_abs": 0.5, "modes": ["smoke", "full"]},
    {"key": "avg_latency_ms", "name": "平均延迟(ms)",
     "max_abs": 30000.0, "modes": ["smoke", "full"]},
    # —— full 端到端集（相对基线退化）——
    {"key": "end_to_end.avg_f1", "name": "端到端 F1",
     "needs_baseline": True, "modes": ["full"]},
    {"key": "end_to_end.hit_rate", "name": "端到端宽松命中率",
     "needs_baseline": True, "modes": ["full"]},
]

# 相对基线退化的容差（吸收 LLM 非确定性抖动；低于 -tol 才判 violation）
DEFAULT_TOL: Dict[str, float] = {
    "end_to_end.avg_f1": 0.05,
    "end_to_end.hit_rate": 0.08,
}


@dataclass
class GateResult:
    """质量闸门判定结果。"""
    passed: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    checked: List[str] = field(default_factory=list)


def _get(summary: Optional[Dict], key_path: str) -> Optional[float]:
    """按点分路径取嵌套值（如 'retrieval.hit_rate'）；缺键返回 None。"""
    if not isinstance(summary, dict):
        return None
    cur: Any = summary
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) else None


def evaluate_gate(
    summary: Dict[str, Any],
    *,
    mode: str = "smoke",
    baseline: Optional[Dict[str, Any]] = None,
    tol: Optional[Dict[str, float]] = None,
) -> GateResult:
    """对 aggregate summary 做质量闸门判定。

    Args:
        summary: run_e2e_eval.aggregate() 的返回（含 retrieval/end_to_end/...）。
        mode: "smoke"（鲁棒集）或 "full"（鲁棒集 + 端到端基线退化）。
        baseline: full 模式用于相对退化比对的基线 summary（同结构 dict）。
        tol: 各端到端指标的容差覆盖；缺省用 DEFAULT_TOL。

    Returns:
        GateResult；passed = 无 violation。
    """
    if mode not in ("smoke", "full"):
        raise ValueError(f"mode 须为 smoke/full，得到 {mode!r}")
    tol_map = dict(DEFAULT_TOL)
    if tol:
        tol_map.update(tol)

    result = GateResult(passed=True)

    # 健康度：全链路不崩（任一题抛异常即 num_ok < num_samples）
    n_ok = summary.get("num_ok")
    n_all = summary.get("num_samples")
    result.checked.append("chain_health")
    if isinstance(n_ok, int) and isinstance(n_all, int):
        if n_ok < n_all:
            result.violations.append({
                "metric": "chain_health",
                "msg": f"链路不健康：{n_ok}/{n_all} 题成功（{n_all - n_ok} 题抛异常）",
                "cur": n_ok, "threshold": n_all,
            })
    else:
        result.violations.append({
            "metric": "chain_health",
            "msg": "summary 缺 num_ok/num_samples，无法判定链路健康度",
        })

    # 逐指标判定
    for m in _METRICS:
        if mode not in m["modes"]:
            continue
        key = m["key"]
        result.checked.append(key)
        cur = _get(summary, key)

        # 缺键处理
        if cur is None:
            is_robust = not m.get("needs_baseline")
            if mode == "smoke" and is_robust:
                result.violations.append({
                    "metric": key, "name": m["name"],
                    "msg": f"summary 缺鲁棒指标 {key}（harness 可能损坏）",
                })
            else:
                result.warnings.append({
                    "metric": key, "name": m["name"],
                    "msg": f"summary 缺指标 {key}，跳过该项判定",
                })
            continue

        # 绝对下限
        if "min_abs" in m and cur < m["min_abs"]:
            result.violations.append({
                "metric": key, "name": m["name"],
                "cur": round(cur, 4), "threshold": m["min_abs"],
                "msg": f"{m['name']}={cur:.4f} 低于下限 {m['min_abs']}",
            })
        # 绝对上限
        if "max_abs" in m and cur > m["max_abs"]:
            result.violations.append({
                "metric": key, "name": m["name"],
                "cur": round(cur, 1), "threshold": m["max_abs"],
                "msg": f"{m['name']}={cur:.1f} 超过上限 {m['max_abs']}",
            })

        # 相对基线退化（仅 needs_baseline 且提供了 baseline）
        if m.get("needs_baseline"):
            if baseline is None:
                result.infos.append({
                    "metric": key, "name": m["name"],
                    "msg": f"未提供基线，跳过 {key} 的退化比对",
                })
                continue
            base = _get(baseline, key)
            if base is None:
                result.infos.append({
                    "metric": key, "name": m["name"],
                    "msg": f"基线缺 {key}，跳过退化比对（兼容旧基线）",
                })
                continue
            t = tol_map.get(key, 0.0)
            delta = cur - base
            if delta < -t:
                result.violations.append({
                    "metric": key, "name": m["name"],
                    "cur": round(cur, 4), "base": round(base, 4),
                    "delta": round(delta, 4), "tol": t,
                    "msg": (f"{m['name']} 退化 {delta:+.4f} 超容差 -{t} "
                            f"（cur={cur:.4f} base={base:.4f}）"),
                })
            elif delta < 0:
                result.warnings.append({
                    "metric": key, "name": m["name"],
                    "cur": round(cur, 4), "base": round(base, 4),
                    "delta": round(delta, 4),
                    "msg": f"{m['name']} 轻降 {delta:+.4f}（在容差内，不阻断）",
                })

    result.passed = len(result.violations) == 0
    return result


def format_gate_report(res: GateResult) -> str:
    """把 GateResult 格式化成可读文本（供 CLI 打印）。"""
    lines = []
    if res.passed:
        lines.append("✅ 质量闸门通过")
    else:
        lines.append(f"❌ 质量闸门未通过（{len(res.violations)} 项违规）")
    for v in res.violations:
        lines.append(f"  [VIOLATION] {v.get('name', v['metric'])}: {v['msg']}")
    for w in res.warnings:
        lines.append(f"  [warning]   {w.get('name', w['metric'])}: {w['msg']}")
    for info in res.infos:
        lines.append(f"  [info]      {info.get('name', info['metric'])}: {info['msg']}")
    lines.append(f"  已检查指标: {', '.join(res.checked)}")
    return "\n".join(lines)
