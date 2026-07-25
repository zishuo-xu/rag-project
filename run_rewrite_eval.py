"""F12 重写层评估（零 LLM、离线、秒级）：多轮追问改写的关键词命中率。

对 data/eval_multiturn.json 逐条调 ConversationRewriter.rewrite()，判定：
- 触发率：needs_rewrite 是否识别出指代/省略型追问
- 命中率：改写后查询是否包含 rewrite_gold 的全部关键词
- 保真检查：非追问样本（若有）不应被误改写

用法:
  uv run python run_rewrite_eval.py                 # 启发式路径（默认）
  uv run python run_rewrite_eval.py --use-llm       # 对比 LLM 重写路径（需 API）
  uv run python run_rewrite_eval.py --dataset data/eval_multiturn.json
"""

import argparse
import json
import logging
from pathlib import Path

from config import get_settings
from app.retrieval.conversation import ConversationRewriter, needs_rewrite

logging.basicConfig(level=logging.WARNING)


def eval_rewrite(rewriter: ConversationRewriter, sample: dict) -> dict:
    """单样本重写评估：触发判定 + gold 关键词命中。"""
    question = sample["question"]
    history = sample.get("history", [])
    gold_keywords = sample.get("rewrite_gold", [])

    triggered = needs_rewrite(question)
    rewritten = rewriter.rewrite(question, history)
    changed = rewritten != question
    hits = [kw for kw in gold_keywords if kw in rewritten]
    missed = [kw for kw in gold_keywords if kw not in rewritten]

    return {
        "id": sample.get("id"),
        "rewrite_type": sample.get("rewrite_type", ""),
        "question": question,
        "triggered": triggered,
        "changed": changed,
        "rewritten": rewritten,
        "gold_keywords": gold_keywords,
        "hits": hits,
        "missed": missed,
        "ok": triggered and changed and not missed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_multiturn.json")
    parser.add_argument("--use-llm", action="store_true", help="启用 LLM 重写路径（默认启发式零LLM）")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    settings = get_settings()
    settings.use_history_rewrite = True
    settings.history_rewrite_use_llm = args.use_llm
    rewriter = ConversationRewriter(llm=args.use_llm or None)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = dataset["samples"]

    results = []
    for s in samples:
        r = eval_rewrite(rewriter, s)
        results.append(r)
        mark = "✓" if r["ok"] else "✗"
        print(f"{mark} [{r['id']}] ({r['rewrite_type']}) {r['question']}")
        print(f"    -> {r['rewritten']}")
        if r["missed"]:
            print(f"    未命中关键词: {r['missed']}")

    n = len(results)
    summary = {
        "path": "llm" if args.use_llm else "heuristic",
        "num_samples": n,
        "trigger_rate": round(sum(int(r["triggered"]) for r in results) / n, 4) if n else 0,
        "rewrite_rate": round(sum(int(r["changed"]) for r in results) / n, 4) if n else 0,
        "keyword_hit_rate": round(sum(int(r["ok"]) for r in results) / n, 4) if n else 0,
        "by_type": {},
    }
    for t in {r["rewrite_type"] for r in results}:
        sub = [r for r in results if r["rewrite_type"] == t]
        summary["by_type"][t] = {
            "n": len(sub),
            "keyword_hit_rate": round(sum(int(r["ok"]) for r in sub) / len(sub), 4),
        }

    print(f"\n=== 汇总 [{summary['path']}] ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = args.output or f"data/eval_rewrite_{summary['path']}.json"
    Path(output).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入: {output}")


if __name__ == "__main__":
    main()
