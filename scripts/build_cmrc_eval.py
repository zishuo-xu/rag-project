"""从 hfl/cmrc2018（train+validation，人工标注）构造 canonical 语料 + 评测集。

用法: uv run python scripts/build_cmrc_eval.py \
          [--splits train+validation] [--contexts 0] [--questions 300] \
          [--per-context 1] [--passages-per-file 20] [--seed 42]

产出:
  - data/cmrc_docs/cmrc_full_{001..N}.md   （全量语料，每文件 ~20 维基篇章）
  - data/eval_dataset_cmrc_full.json        （~300 单跳 QA，gold=人工标注众数）

设计动机:
  扩容检索语料到 CMRC2018 全量（train+val 去重 ~3251 篇章 / ~3700 块），让检索
  hit_rate 在大规模干扰项下有真实区分度；评测集同步扩到 ~300 题（广覆盖，每篇 1 题），
  把正确率置信区间从 n=100 的 ±0.041 收窄到 ±0.023。

纪律:
  - ground_truth 取标注者众数（answers.text 的 mode），短 span，适配 char-F1/子串 hit。
  - 与现存 cmrc_docs 去重（跳过首 50 字重合的篇章）：保留手工 3 篇，避免语料重复。
    （5 个 cmrc_expanded_*.md 应在运行前删除，其篇章会被本脚本重新纳入全量集。）
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
EVAL_OUT = Path("data/eval_dataset_cmrc_full.json")
OUT_PREFIX = "cmrc_full"
MAX_GOLD_LEN = 80          # 跳过极长 gold（罕见、噪声大）
MIN_CTX_LEN, MAX_CTX_LEN = 80, 2000   # 篇章长度过滤（去近空 stub / 极长跨块噪声）


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
    ap.add_argument("--splits", default="train+validation",
                    help="HF splits，'+' 连接（默认 train+validation 全量）")
    ap.add_argument("--contexts", type=int, default=0,
                    help="选取篇章数；0=全部（过滤后）")
    ap.add_argument("--questions", type=int, default=300, help="目标题数")
    ap.add_argument("--per-context", type=int, default=1,
                    help="每篇最多取题数（1=广覆盖，每题来自不同篇章）")
    ap.add_argument("--passages-per-file", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split("+") if s.strip()]
    print(f"加载 hfl/cmrc2018 splits={splits} …")

    # 按篇章聚合问题（跨 split 去重：同 context 合并）
    by_ctx: dict[str, list] = defaultdict(list)
    total_q = 0
    for split in splits:
        ds = load_dataset("hfl/cmrc2018", split=split)
        total_q += len(ds)
        for r in ds:
            by_ctx[r["context"]].append(r)

    existing = load_existing_context_prefixes()
    print(f"{splits}: {total_q} 题 / {len(by_ctx)} 去重篇章；"
          f"现有语料篇章首缀 {len(existing)} 个（去重用）")

    # 候选篇章：长度适中 + 与现有语料不重合（保留手工 3 篇，不重复生成）
    candidates = []
    for ctx, qas in by_ctx.items():
        if not (MIN_CTX_LEN <= len(ctx) <= MAX_CTX_LEN):
            continue
        if ctx.strip()[:50] in existing:
            continue
        candidates.append(ctx)
    # 确定顺序（按篇章文本排序）保证可复现，再按需截断
    candidates.sort()
    if args.contexts > 0:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        candidates = candidates[: args.contexts]
    selected = candidates
    print(f"过滤后选取篇章: {len(selected)}（长度 {MIN_CTX_LEN}-{MAX_CTX_LEN}，去重现有语料）")

    # 篇章 → 文件分配（确定顺序），同时建 ## 块
    ctx_to_file: dict[str, str] = {}
    file_blocks: dict[str, list[str]] = defaultdict(list)
    for ci, ctx in enumerate(selected):
        fname = f"{OUT_PREFIX}_{ci // args.passages_per_file + 1:03d}.md"
        ctx_to_file[ctx] = fname
        heading = make_heading(ctx)
        file_blocks[fname].append(f"## {heading}\n\n{ctx.strip()}")

    # 评测题：广覆盖——篇章洗牌后每篇取 per_context 个干净短 gold 题，命中目标即停
    rng = random.Random(args.seed)
    eval_ctxs = list(selected)
    rng.shuffle(eval_ctxs)
    samples = []
    qid = 0
    for ctx in eval_ctxs:
        if len(samples) >= args.questions:
            break
        qas = sorted(by_ctx[ctx], key=lambda r: r["id"])
        taken = 0
        for r in qas:
            if taken >= args.per_context:
                break
            gold = pick_gold(r["answers"]["text"])
            if not gold or len(gold) > MAX_GOLD_LEN:
                continue
            qid += 1
            taken += 1
            samples.append({
                "id": f"cf{qid:03d}",
                "question": r["question"].strip(),
                "ground_truth": gold,
                "metadata": {"source": ctx_to_file[ctx], "dataset": "cmrc2018",
                             "type": "reading_comprehension"},
            })

    # 写语料 md（对齐现有 cmrc_docs 格式：# 总标题 + ## 各篇章）
    CMRC_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, blocks in file_blocks.items():
        idx = re.search(r"(\d+)", fname).group(1)
        title = f"# CMRC2018 百科语料（{int(idx)}）\n\n"
        (CMRC_DOCS_DIR / fname).write_text(
            title + "\n\n".join(blocks) + "\n", encoding="utf-8"
        )

    # 写评测集
    payload = {
        "version": "1.0",
        "num_samples": len(samples),
        "description": "基于 hfl/cmrc2018 train+validation（人工标注）的全量单跳 RAG 评测集；"
                       "gold 取标注者众数，广覆盖（每题来自不同篇章）；"
                       "全量篇章入库作大规模检索干扰项。",
        "samples": samples,
    }
    EVAL_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计
    gold_lens = [len(s["ground_truth"]) for s in samples]
    n_files = len(file_blocks)
    print(f"\n✅ 语料：{n_files} 个新 md（{len(selected)} 篇章）→ {CMRC_DOCS_DIR}/{OUT_PREFIX}_*.md")
    print(f"✅ 评测集：{len(samples)} 题（覆盖 {len(set(s['metadata']['source'] for s in samples))} 个文件）→ {EVAL_OUT}")
    if gold_lens:
        print(f"   gold 长度: min={min(gold_lens)} max={max(gold_lens)} "
              f"avg={sum(gold_lens)/len(gold_lens):.1f}")
    print("\n下一步：干净重建索引（清 chroma/graph/bm25 → 重灌 sample_docs + cmrc_docs）后跑评测。")


if __name__ == "__main__":
    main()
