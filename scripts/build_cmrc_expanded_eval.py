"""从 hfl/cmrc2018 validation 集扩充单跳评测集 + 语料（人工标注 gold，零 LLM 造数）。

用法: uv run python scripts/build_cmrc_expanded_eval.py \
          [--contexts 40] [--questions 100] [--per-context 3] [--seed 42]

产出:
  - data/cmrc_docs/cmrc_expanded_{1..N}.md  （扩充语料，每篇 ~8 个维基篇章）
  - data/eval_dataset_cmrc_expanded.json     （~100 单跳 QA，gold=CMRC2018 人工标注）

设计动机:
  旧评测仅 31 题 / 语料仅 ~18 篇，correctness 置信区间宽（n=30 → ±0.09），且语料太小
  使检索 hit_rate=1.0 几乎无区分度（无干扰项）。本脚本从 CMRC2018 标准 dev 集
  （3219 题 / 848 篇章，人工标注）扩充：新增篇章既作可答语料、又互为检索干扰项，
  同时收窄正确率置信区间并真正压测规模化检索精度。

纪律:
  - ground_truth 取标注者众数（answers.text 的 mode），短 span，适配 char-F1/子串 hit。
  - 与现有 cmrc_docs 去重（跳过首 50 字重合的篇章），避免语料重复。
  - 固定 seed 保证可复现；篇章按确定顺序分配文件，metadata.source 指向所属 md。
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

CMRC_DOCS_DIR = Path("data/cmrc_docs")
EVAL_OUT = Path("data/eval_dataset_cmrc_expanded.json")
PASSAGES_PER_FILE = 8
MAX_GOLD_LEN = 80          # 跳过极长 gold（罕见、噪声大）
MIN_CTX_LEN, MAX_CTX_LEN = 120, 1500   # 篇章长度过滤（太短信息不足、太长易跨块）


def pick_gold(texts: list) -> str | None:
    """取标注者众数 gold；空则 None。"""
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return None
    return Counter(texts).most_common(1)[0][0]


def make_heading(context: str, maxlen: int = 30) -> str:
    """从篇章首句派生 ## 标题（对齐现有 cmrc_docs 风格）。"""
    first = re.split(r"[。！？\n]", context.strip(), 1)[0].strip()
    first = re.sub(r"\s+", " ", first)
    return first[:maxlen] or context[:maxlen]


def load_existing_context_prefixes() -> set:
    """现有 cmrc_docs 各篇章首 50 字，用于去重。"""
    prefixes = set()
    for md in CMRC_DOCS_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        # 按 ## 切篇，取每篇正文首 50 字
        for block in re.split(r"\n## ", text):
            body = re.sub(r"^#[^\n]*\n", "", block).strip()
            body = re.sub(r"^[^\n]*\n", "", body, 1).strip()  # 去掉标题行
            if body:
                prefixes.add(body[:50])
    return prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, default=40, help="选取篇章数")
    ap.add_argument("--questions", type=int, default=100, help="目标题数")
    ap.add_argument("--per-context", type=int, default=3, help="每篇最多取题数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("加载 hfl/cmrc2018 validation …")
    ds = load_dataset("hfl/cmrc2018", split="validation")

    # 按篇章聚合问题
    by_ctx: dict[str, list] = defaultdict(list)
    for r in ds:
        by_ctx[r["context"]].append(r)

    existing = load_existing_context_prefixes()
    print(f"validation: {len(ds)} 题 / {len(by_ctx)} 篇章；现有语料篇章首缀 {len(existing)} 个（去重用）")

    # 候选篇章：题量充足 + 长度适中 + 与现有语料不重合
    rng = random.Random(args.seed)
    candidates = []
    for ctx, qas in by_ctx.items():
        if len(qas) < 2:
            continue
        if not (MIN_CTX_LEN <= len(ctx) <= MAX_CTX_LEN):
            continue
        if ctx.strip()[:50] in existing:
            continue
        candidates.append(ctx)
    rng.shuffle(candidates)
    selected = candidates[: args.contexts]
    if len(selected) < args.contexts:
        print(f"⚠️ 候选篇章不足，实际选取 {len(selected)}（< {args.contexts}）")

    # 逐篇取题（确定顺序：篇章内按 id 排序），命中目标题数即停
    samples = []
    ctx_to_file: dict[str, str] = {}
    file_blocks: dict[str, list[str]] = defaultdict(list)
    qid = 0
    for ci, ctx in enumerate(selected):
        fname = f"cmrc_expanded_{ci // PASSAGES_PER_FILE + 1}.md"
        ctx_to_file[ctx] = fname
        heading = make_heading(ctx)
        file_blocks[fname].append(f"## {heading}\n\n{ctx.strip()}")

        qas = sorted(by_ctx[ctx], key=lambda r: r["id"])
        taken = 0
        for r in qas:
            if taken >= args.per_context or len(samples) >= args.questions:
                break
            gold = pick_gold(r["answers"]["text"])
            if not gold or len(gold) > MAX_GOLD_LEN:
                continue
            qid += 1
            taken += 1
            samples.append({
                "id": f"cx{qid:03d}",
                "question": r["question"].strip(),
                "ground_truth": gold,
                "metadata": {"source": fname, "dataset": "cmrc2018",
                             "type": "reading_comprehension"},
            })

    # 写语料 md（对齐现有 cmrc_docs 格式：# 总标题 + ## 各篇章）
    CMRC_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, blocks in file_blocks.items():
        idx = re.search(r"(\d+)", fname).group(1)
        title = f"# 百科知识扩充（{idx}）\n\n"
        (CMRC_DOCS_DIR / fname).write_text(
            title + "\n\n".join(blocks) + "\n", encoding="utf-8"
        )

    # 写评测集
    payload = {
        "version": "1.0",
        "num_samples": len(samples),
        "description": "基于 hfl/cmrc2018 validation（人工标注）扩充的单跳 RAG 评测集；"
                       "gold 取标注者众数，篇章同时入库作检索干扰项。",
        "samples": samples,
    }
    EVAL_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    gold_lens = [len(s["ground_truth"]) for s in samples]
    per_file = Counter(s["metadata"]["source"] for s in samples)
    print(f"\n✅ 语料：{len(file_blocks)} 个新 md（{len(selected)} 篇章）→ {CMRC_DOCS_DIR}/cmrc_expanded_*.md")
    print(f"✅ 评测集：{len(samples)} 题 → {EVAL_OUT}")
    print(f"   gold 长度: min={min(gold_lens)} max={max(gold_lens)} "
          f"avg={sum(gold_lens)/len(gold_lens):.1f}")
    print(f"   按文件分布: {dict(sorted(per_file.items()))}")
    print("\n下一步：把新语料灌入索引（chroma/bm25/graph）后跑 run_e2e_eval.py --judge。")


if __name__ == "__main__":
    main()
