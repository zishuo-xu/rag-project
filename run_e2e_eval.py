"""端到端三层评估 harness（F5）+ 模拟真实用户测试

对 CMRC 测试集跑 chain.invoke() 全链路，度量三层（对应文章「三层评估」）：
1. 检索层：命中率(keyword_coverage≥0.5 或 source_hit) + 平均检索篇数 + Autocut 降噪幅度
2. 生成层：忠实度(response.faithfulness_score, F3) + 重生成率
3. 端到端：答案正确性 vs gold —— 中文 F1 / 严格EM / 宽松命中(answer_hit)
附：平均延迟 / 路由类型分布(F4) / 迭代次数与终止原因(F2)

A/B 与特性归因（隔离 RAG2.0 四特性 F1-F4 的边际贡献）：
  --mode baseline : F1-F4 全关（RAG1.0 基线行为）
  --mode full     : F1-F4 全开
  --only F1|F2|F3|F4 : 仅开单一特性，其余关闭（单特性归因）

健壮性：每题 try/except 优雅降级并如实计入失败；增量写盘（断点不丢）；
评估强制关闭语义缓存避免跨模式污染。

用法:
  uv run python run_e2e_eval.py --mode full
  uv run python run_e2e_eval.py --mode baseline
  uv run python run_e2e_eval.py --mode full --colloquial   # 追加口语化查询定性检视
  uv run python run_e2e_eval.py --mode full --limit 5      # 冒烟：仅前5题
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

from config import get_settings
from app.evaluation.metrics import answer_f1, normalized_exact_match, answer_hit, answer_in_top_context, normalize_answer, answer_correctness
from app.evaluation.gate import evaluate_gate, format_gate_report

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# RAG2.0/3.0 特性开关（A/B 隔离对象；--only 单特性归因，每个映射单个 settings 标志）
FEATURE_FLAGS = {
    "F1": "use_autocut",
    "F2": "use_iterative_retrieval",
    "F3": "use_faithfulness_check",
    "F4": "use_query_router",
    "F6": "use_decomposition",
    "F7": "use_citations",
    "F8": "use_speculative_streaming",
    "F10": "use_answer_extraction",
    "F12": "use_history_rewrite",
    "F13": "use_agentic",
}

# F13 为重 LLM 决策路径：不随 baseline/full 批量切换，仅 --only F13 显式评估
_STANDALONE_ONLY = {"F13"}

# F6a 运行时开关：随 baseline/full 一并切换；--only 单特性归因时关闭以隔离
F6A_FLAGS = ["use_contextual_chunks", "use_parent_child"]

# F9 涉及两个标志（L1 embedding + L2 rerank），单特性归因时一并切换
F9_FLAGS = ["use_embedding_cache", "use_rerank_cache"]

# RAG 3.0 全部标志（baseline 关 / full 开）
RAG3_FLAGS = [
    "use_citations", "use_speculative_streaming",
    "use_embedding_cache", "use_rerank_cache",
    "use_answer_extraction", "use_history_rewrite",
]

# 模拟真实用户的口语化查询（覆盖边缘场景；引用 CMRC 知识库真实实体 范廷颂）
COLLOQUIAL_QUERIES = [
    "范廷颂这人到底当过啥官啊？",            # 口语化措辞
    "范廷颂是哪一年当上主教的？",            # 数字型
    "什么是天主教的主教制度？",              # 概念型
    "主教和总主教有什么区别？",              # 对比型
    "范廷颂担任总主教的那个教区在哪里？",    # 多跳
    "范廷颂的手机号码是多少？",              # 不可答（应诚实拒答而非幻觉）
    "2099年范廷颂做了什么？",                # 不可答（未来时间）
    "那他后来怎么样了？",                    # 追问式（无上下文）
    "帮我讲讲范廷颂的故事呗",                # 口语化 + 模糊
    "你好呀，今天天气怎么样？",              # 闲聊（门控应跳过检索）
]


def apply_feature_mode(mode: str, only: str | None) -> dict:
    """在构造 RAGChain 前改写 settings 单例的特性标志。返回实际开关状态。"""
    settings = get_settings()
    # 评估强制关闭语义缓存，避免 baseline 答案被 full 模式缓存命中而污染 A/B
    settings.cache_enabled = False

    if mode == "baseline":
        for flag in FEATURE_FLAGS.values():
            setattr(settings, flag, False)
        for flag in F6A_FLAGS + F9_FLAGS + RAG3_FLAGS:
            setattr(settings, flag, False)
    elif mode == "full":
        for key, flag in FEATURE_FLAGS.items():
            if key in _STANDALONE_ONLY:
                continue  # F13 仅 --only 显式开启
            setattr(settings, flag, True)
        for flag in F6A_FLAGS + F9_FLAGS + RAG3_FLAGS:
            setattr(settings, flag, True)

    if only:
        only = only.upper()
        if only not in FEATURE_FLAGS and only != "F9":
            raise ValueError(f"--only 须为 {list(FEATURE_FLAGS) + ['F9']} 之一")
        # 先全关，再单独开启目标特性
        for key, flag in FEATURE_FLAGS.items():
            setattr(settings, flag, key == only)
        for flag in F6A_FLAGS + F9_FLAGS:
            setattr(settings, flag, False)
        if only == "F9":
            for flag in F9_FLAGS:
                setattr(settings, flag, True)

    state = {key: bool(getattr(settings, flag)) for key, flag in FEATURE_FLAGS.items()}
    state["F9"] = bool(getattr(settings, "use_rerank_cache"))
    return state


def keyword_coverage(question: str, ground_truth: str, documents) -> float:
    """ground_truth 中的关键词在检索结果中的覆盖率（沿用 run_retrieval_eval 口径）"""
    keywords = set(re.findall(r"[一-鿿]{2,}|\d+", ground_truth))
    if not keywords:
        return 1.0
    context = " ".join(d.page_content for d in documents)
    hits = sum(1 for kw in keywords if kw in context)
    return hits / len(keywords)


def filter_gradable(samples: list) -> tuple[list, list]:
    """按可评分性切分样本：gold 归一化后非空才可评分。

    抽取式数据（如 CMRC）偶有 gold 为空占位——源文档实体缺失留下的《""》，
    归一化（仅留汉字/字母/数字）后为空串 → F1/EM/hit 结构性恒 0，纳入会拉低
    均值且无法反映真实质量。返回 (可评分, 不可评分)，由调用方透明上报后者。
    """
    gradable, ungradable = [], []
    for s in samples:
        (gradable if normalize_answer(s.get("ground_truth", "")) else ungradable).append(s)
    return gradable, ungradable


def eval_sample(chain, sample, judge: bool = False) -> dict:
    """对单个带 gold 的样本跑全链路并度量三层指标。异常优雅降级。

    judge=True 时追加语义正确率 `correctness`（LLM-as-judge，每样本 1 次 LLM 调用）——
    比词法 F1/hit 更真的质量信号；默认关以保住零 LLM 快路径。
    """
    question = sample["question"]
    gold = sample["ground_truth"]
    expected_source = sample.get("metadata", {}).get("source", "")
    # F12 多轮评估：样本带 history 时透传给 chain（无 history 样本行为不变）
    history = sample.get("history")

    t0 = time.time()
    try:
        resp = chain.invoke(question, chat_history=history)
    except Exception as e:
        logger.warning(f"样本失败 [{question[:20]}]: {e}")
        return {
            "id": sample.get("id"), "question": question, "gold": gold,
            "ok": False, "error": str(e),
        }
    ms = (time.time() - t0) * 1000

    docs = resp.sources
    rr = resp.retrieval_result

    # 检索层
    coverage = keyword_coverage(question, gold, docs)
    source_hit = any(
        expected_source and expected_source in d.metadata.get("source", "")
        for d in docs
    )
    retrieval_ok = coverage >= 0.5 or source_hit

    # F10 短答案：抽取的核心 span，用于度量答案对齐（改善 EM=0）
    short = getattr(resp, "short_answer", "") or ""

    result = {
        "id": sample.get("id"), "question": question, "gold": gold,
        "answer": resp.answer,
        "ok": True,
        # 检索层
        "retrieval_ok": retrieval_ok, "coverage": round(coverage, 4),
        "source_hit": source_hit, "num_docs": len(docs),
        "pre_autocut": rr.pre_autocut_count,
        # 生成层
        "faithfulness": round(resp.faithfulness_score, 4),
        "faithful": resp.faithful, "regenerated": resp.regenerated,
        # 端到端（完整答案）
        "f1": round(answer_f1(gold, resp.answer), 4),
        "em": normalized_exact_match(gold, resp.answer),
        "hit": answer_hit(gold, resp.answer),
        # 端到端（F10 短答案）
        "short_answer": short,
        "f1_short": round(answer_f1(gold, short), 4) if short else 0.0,
        "em_short": normalized_exact_match(gold, short) if short else False,
        "hit_short": answer_hit(gold, short) if short else False,
        # F7 引用溯源
        "num_citations": len(getattr(resp, "citations", []) or []),
        # F12 重写（记录改写后查询字符串，便于归因"重写对但检索错"类问题）
        "rewritten": bool(getattr(resp, "rewritten_query", "")),
        "rewritten_query": getattr(resp, "rewritten_query", "") or "",
        # 观测
        "query_type": rr.query_type, "iterations": rr.iterations_used,
        "stop_reason": rr.iterative_stop_reason,
        "gate_skipped": rr.gate_skipped,
        "latency_ms": round(ms, 1),
        # F6 答案定位
        "answer_in_top_context": answer_in_top_context(gold, docs),
        "decomposed": rr.decomposed_subqueries,
        "decomposition_chain": rr.decomposition_chain,
        # F13 Agentic RAG 观测
        "agent_steps": len(rr.agent_steps),
        "agent_stop_reason": rr.agent_stop_reason,
        "agent_actions": [s.get("action") for s in rr.agent_steps],
    }
    # 语义裁判（--judge）：仅当开启且 gold 可评分时追加，judge 关时不写该键、不污染聚合
    if judge and normalize_answer(gold):
        result["correctness"] = round(answer_correctness(question, resp.answer, gold), 4)
    return result


def eval_colloquial(chain, query: str) -> dict:
    """口语化查询定性检视：捕获完整响应特征（无 gold）。"""
    t0 = time.time()
    try:
        resp = chain.invoke(query)
    except Exception as e:
        return {"question": query, "ok": False, "error": str(e)}
    ms = (time.time() - t0) * 1000
    rr = resp.retrieval_result
    return {
        "question": query, "ok": True,
        "answer": resp.answer,
        "num_docs": len(resp.sources),
        "gate_skipped": rr.gate_skipped,
        "query_type": rr.query_type,
        "iterations": rr.iterations_used,
        "stop_reason": rr.iterative_stop_reason,
        "pre_autocut": rr.pre_autocut_count,
        "faithful": resp.faithful,
        "faithfulness": round(resp.faithfulness_score, 4),
        "regenerated": resp.regenerated,
        "latency_ms": round(ms, 1),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate(results: list) -> dict:
    """汇总三层指标 + 观测统计"""
    ok = [r for r in results if r.get("ok")]
    n = len(ok)
    if n == 0:
        return {"num_samples": len(results), "num_ok": 0, "num_failed": len(results)}

    faith_vals = [r["faithfulness"] for r in ok if r["faithful"] is not None]
    summary = {
        "num_samples": len(results),
        "num_ok": n,
        "num_failed": len(results) - n,
        "retrieval": {
            "hit_rate": _mean([int(r["retrieval_ok"]) for r in ok]),
            "avg_coverage": _mean([r["coverage"] for r in ok]),
            "avg_num_docs": _mean([r["num_docs"] for r in ok]),
            "avg_pre_autocut": _mean([r["pre_autocut"] for r in ok]),
        },
        "generation": {
            "avg_faithfulness": _mean(faith_vals),
            "regen_rate": _mean([int(r["regenerated"]) for r in ok]),
        },
        "end_to_end": {
            "avg_f1": _mean([r["f1"] for r in ok]),
            "em_rate": _mean([int(r["em"]) for r in ok]),
            "hit_rate": _mean([int(r["hit"]) for r in ok]),
            "answer_in_top_context_rate": _mean([int(r["answer_in_top_context"]) for r in ok]),
        },
        # F10 短答案对齐（改善 EM=0 的关键观测）
        "answer_quality": {
            "avg_f1_short": _mean([r.get("f1_short", 0.0) for r in ok]),
            "em_rate_short": _mean([int(r.get("em_short", False)) for r in ok]),
            "hit_rate_short": _mean([int(r.get("hit_short", False)) for r in ok]),
        },
        # F7 引用溯源 / F12 重写观测
        "rag3": {
            "avg_num_citations": _mean([r.get("num_citations", 0) for r in ok]),
            "rewrite_rate": _mean([int(r.get("rewritten", False)) for r in ok]),
        },
        "avg_latency_ms": _mean([r["latency_ms"] for r in ok]),
        "routing_dist": dict(Counter(r["query_type"] or "n/a" for r in ok)),
        "iterative": {
            "avg_iterations": _mean([r["iterations"] for r in ok]),
            "stop_reasons": dict(Counter(r["stop_reason"] or "n/a" for r in ok)),
        },
        "decomposition": {
            "decomposed_rate": _mean([int(bool(r["decomposed"])) for r in ok]),
            "chain_rate": _mean([int(r["decomposition_chain"]) for r in ok]),
        },
        "agentic": {
            "avg_steps": _mean([r.get("agent_steps", 0) for r in ok]),
            "stop_reasons": dict(Counter(r.get("agent_stop_reason") or "n/a" for r in ok)),
            "action_dist": dict(Counter(a for r in ok for a in r.get("agent_actions", []))),
        },
    }
    # 语义正确率（--judge）：仅当至少一个样本带 correctness 时才 emit，
    # judge 关时键缺失 → gate 记 info 跳过（非 violation），向后兼容快路径
    corr_vals = [r["correctness"] for r in ok if r.get("correctness") is not None]
    if corr_vals:
        summary["end_to_end"]["avg_correctness"] = _mean(corr_vals)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset_cmrc_full.json",
                        help="默认用全量评测集（300 题，CMRC2018 人工标注 gold，广覆盖；"
                             "语料扩到 ~3700 块作大规模检索干扰项）；"
                             "旧 31 题集 data/eval_dataset_cmrc.json 仍可显式指定做快速冒烟")
    parser.add_argument("--mode", default="full", choices=["baseline", "full"])
    parser.add_argument("--only", default=None, help="单特性归因: F1|F2|F3|F4|F6")
    parser.add_argument("--colloquial", action="store_true", help="追加口语化查询检视")
    parser.add_argument("--judge", action="store_true",
                        help="追加语义正确率（LLM-as-judge，每样本+1次LLM调用）；"
                             "比词法F1/hit更真的质量信号，默认关以保住零LLM快路径")
    parser.add_argument("--limit", type=int, default=0, help="仅跑前N题（0=全部）")
    parser.add_argument("--slice", default="", choices=["", "multihop", "finegrained", "multiturn"],
                        help="按样本 slice 字段过滤（多跳/细粒度/多轮子集）")
    parser.add_argument("--output", default="")
    # —— 质量闸门（eval-as-gate）：默认不触发，行为中性 ——
    parser.add_argument("--gate", action="store_true",
                        help="跑完后做质量闸门判定，退化即 exit 2")
    parser.add_argument("--gate-mode", default=None, choices=["smoke", "full"],
                        help="闸门模式：smoke=鲁棒集 / full=叠加端到端基线退化；"
                             "缺省随 --limit 推（>0→smoke 否则 full）")
    parser.add_argument("--baseline", default="data/eval_gate_baseline.json",
                        help="full 模式相对退化比对的基线 summary 路径")
    parser.add_argument("--update-baseline", action="store_true",
                        help="把本次 summary 写为基线（更新前须人工确认质量未退）")
    parser.add_argument("--smoke", action="store_true",
                        help="便捷别名：--limit 8 --gate-mode smoke（配合 --gate 用）")
    args = parser.parse_args()

    # --smoke 别名展开 + gate-mode 缺省推断
    if args.smoke:
        if args.limit == 0:
            args.limit = 8
        args.gate_mode = "smoke"
    gate_mode = args.gate_mode or ("smoke" if args.limit > 0 else "full")

    feature_state = apply_feature_mode(args.mode, args.only)
    tag = args.only.upper() if args.only else args.mode
    output = args.output or f"data/eval_e2e_{tag.lower()}.json"

    print(f"=== 端到端评估 [{tag}] 特性状态: {feature_state} ===")

    # 构造 chain（在 apply_feature_mode 之后，确保读到改写后的 settings）
    from app.generation.chain import RAGChain
    chain = RAGChain()
    try:
        chain.sparse_retriever.build_index()
    except Exception as e:
        logger.warning(f"BM25 索引构建跳过: {e}")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = dataset["samples"]
    if args.slice:
        samples = [s for s in samples if s.get("slice") == args.slice]
    # 数据卫生：剔除 gold 归一化为空的不可评分样本（结构性恒 0，拉低均值），透明上报
    samples, ungradable = filter_gradable(samples)
    if ungradable:
        ids = [s.get("id") for s in ungradable]
        print(f"[数据卫生] 跳过 {len(ids)} 条不可评分样本（gold 归一化为空）: ids={ids}")
    if args.limit > 0:
        samples = samples[: args.limit]

    # 增量评估 + 写盘（断点不丢）
    results = []
    for i, sample in enumerate(samples, 1):
        r = eval_sample(chain, sample, judge=args.judge)
        results.append(r)
        status = "✓" if r.get("ok") else "✗"
        extra = (
            f"f1={r.get('f1', 0):.2f} hit={int(r.get('hit', False))} "
            f"faith={r.get('faithfulness')}" if r.get("ok") else r.get("error", "")
        )
        if r.get("correctness") is not None:
            extra += f" corr={r['correctness']:.2f}"
        print(f"[{i}/{len(samples)}] {status} {sample['question'][:30]} {extra}")
        _save(output, tag, feature_state, results, None)

    summary = aggregate(results)

    # 口语化查询定性检视
    colloquial = None
    if args.colloquial:
        print(f"\n=== 口语化查询检视（{len(COLLOQUIAL_QUERIES)} 条）===")
        colloquial = []
        for q in COLLOQUIAL_QUERIES:
            c = eval_colloquial(chain, q)
            colloquial.append(c)
            ans_preview = (c.get("answer", "") or "")[:50].replace("\n", " ")
            print(f"  Q: {q}\n    -> [{c.get('query_type', '?')}] "
                  f"docs={c.get('num_docs')} gate_skip={c.get('gate_skipped')} "
                  f"faith={c.get('faithfulness')} | {ans_preview}")
        _save(output, tag, feature_state, results, colloquial)

    _save(output, tag, feature_state, results, colloquial, summary)

    print(f"\n=== 汇总 [{tag}] ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n报告已写入: {output}")

    # —— 质量闸门分支（默认不进入，行为中性）——
    if args.update_baseline:
        Path(args.baseline).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n⚠️  基线已更新为本次 summary: {args.baseline}")
        print("   （请确认本次质量未退化后再更新，否则卡点失效）")
        return

    if args.gate:
        baseline = None
        bp = Path(args.baseline)
        if bp.exists():
            try:
                baseline = json.loads(bp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️  基线读取失败 {bp}: {e}（full 退化比对将跳过）")
        res = evaluate_gate(summary, mode=gate_mode, baseline=baseline)
        print(f"\n=== 质量闸门 [{gate_mode}] ===")
        print(format_gate_report(res))
        if not res.passed:
            sys.exit(2)


def _save(output, tag, feature_state, results, colloquial, summary=None):
    """增量写盘"""
    payload = {
        "mode": tag,
        "provider": get_settings().llm_provider,  # 记录口径：不同 provider 数字不可直接互比
        "feature_state": feature_state,
        "summary": summary or aggregate(results),
        "results": results,
    }
    if colloquial is not None:
        payload["colloquial"] = colloquial
    Path(output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
