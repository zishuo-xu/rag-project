"""运行 RAG 评估脚本"""
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from app.evaluation.dataset import load_eval_dataset
from app.evaluation.metrics import evaluate_rag

DATASET_PATH = "data/eval_dataset.json"


def main():
    # 加载评估数据集
    if not os.path.exists(DATASET_PATH):
        sys.exit(
            f"❌ {DATASET_PATH} 不存在（20 题 RAGAS 抽样集未入库，仓库只提交 300 题 CMRC 全量集）。\n"
            f"   主力评估请用: uv run python run_e2e_eval.py --mode full   # 300 题，与 gate 基线可比\n"
            f"   零 LLM 检索评估: uv run python run_retrieval_eval.py"
        )
    dataset = load_eval_dataset(DATASET_PATH)
    samples = dataset["samples"]

    questions = [s["question"] for s in samples]
    ground_truths = [s["ground_truth"] for s in samples]

    print(f"\n{'='*60}")
    print(f"  RAG 系统评估 - {len(questions)} 个问题")
    print(f"{'='*60}\n")

    start = time.time()
    report = evaluate_rag(questions, ground_truths)
    elapsed = time.time() - start

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  评估结果 (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"\n  指标得分:")
    for k, v in report["metrics"].items():
        status = "✓" if v >= 0.85 else "✗"
        print(f"    {status} {k:25s}: {v:.4f}")

    print(f"\n  优秀标准: faithfulness>0.9, answer_relevancy>0.85,")
    print(f"            context_precision>0.85, context_recall>0.9")

    print(f"\n  逐题详情:")
    for d in report["details"]:
        print(f"    Q: {d['question'][:40]}...")
        print(f"       F={d['faithfulness']:.2f} R={d['answer_relevancy']:.2f} P={d['context_precision']:.2f} RC={d['context_recall']:.2f}")

    # 保存报告
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: data/eval_report.json")


if __name__ == "__main__":
    main()
