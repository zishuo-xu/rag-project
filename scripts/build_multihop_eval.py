"""半自动构造多跳评估集：列出知识库实体候选，输出待人工填写 gold 的骨架。

用法: uv run python scripts/build_multihop_eval.py --out data/eval_multihop.json
产出后必须人工核对每条 ground_truth 与 source，再提交。
"""
import argparse
import json
from pathlib import Path


# 基于知识库真实实体（范廷颂/天主教/总主教/教区）的多跳问题模板。
# ground_truth 为占位，需人工对照 data/sample_docs 核对后填写真实答案。
SEED = [
    {"id": "mh1", "question": "范廷颂担任总主教的那个教区在哪里？",
     "ground_truth": "TODO_核对知识库", "chain": True, "slice": "multihop"},
    {"id": "mh2", "question": "范廷颂受封主教那一年的教宗是谁？",
     "ground_truth": "TODO_核对知识库", "chain": True, "slice": "multihop"},
    {"id": "mh3", "question": "总主教和主教在天主教圣统制中的区别是什么？",
     "ground_truth": "TODO_核对知识库", "chain": False, "slice": "multihop"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/eval_multihop.json")
    args = parser.parse_args()

    payload = {"samples": [
        {**s, "metadata": {"source": ""}} for s in SEED
    ]}
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 {args.out}（{len(SEED)} 条骨架）。")
    print("⚠️ 请人工核对每条 ground_truth 与 metadata.source 后再用于评估。")


if __name__ == "__main__":
    main()
