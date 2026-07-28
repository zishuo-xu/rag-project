"""检索级评估：CMRC 测试集命中率 + 关键词覆盖率（无 LLM judge，只评估检索）

用法: uv run python run_retrieval_eval.py [--dataset data/eval_dataset_cmrc.json]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

from app.generation.chain import RAGChain

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def keyword_coverage(question: str, ground_truth: str, documents) -> float:
    """ground_truth 中的关键词在检索结果中的覆盖率"""
    keywords = set(re.findall(r"[一-鿿]{2,}|\d+", ground_truth))
    if not keywords:
        return 1.0
    context = " ".join(d.page_content for d in documents)
    hits = sum(1 for kw in keywords if kw in context)
    return hits / len(keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_dataset_cmrc_full.json")
    parser.add_argument("--output", default="data/eval_report_cmrc.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples = dataset["samples"]

    chain = RAGChain()
    try:
        chain.sparse_retriever.build_index()
    except Exception as e:
        logger.warning(f"BM25 索引构建跳过: {e}")

    details = []
    hit_count = 0
    total_coverage = 0.0
    total_ms = 0.0

    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        expected_source = sample.get("metadata", {}).get("source", "")

        t0 = time.time()
        result = chain.retrieve(question)
        ms = (time.time() - t0) * 1000

        coverage = keyword_coverage(question, ground_truth, result.documents)
        source_hit = any(
            expected_source and expected_source in d.metadata.get("source", "")
            for d in result.documents
        )
        ok = coverage >= 0.5 or source_hit

        hit_count += int(ok)
        total_coverage += coverage
        total_ms += ms
        details.append({"q": question, "rate": round(coverage, 4), "ok": ok, "ms": ms})
        print(f"[{i}/{len(samples)}] {'✓' if ok else '✗'} {question[:40]} "
              f"coverage={coverage:.2f} {ms:.0f}ms")

    n = len(samples)
    report = {
        "dataset": "cmrc2018",
        "num_samples": n,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / n, 4),
        "avg_keyword_coverage": round(total_coverage / n, 4),
        "avg_retrieval_ms": round(total_ms / n, 1),
        "details": details,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n命中率: {report['hit_rate']:.2%}  覆盖率: {report['avg_keyword_coverage']:.2%}  "
          f"平均耗时: {report['avg_retrieval_ms']:.0f}ms -> {args.output}")


if __name__ == "__main__":
    main()
