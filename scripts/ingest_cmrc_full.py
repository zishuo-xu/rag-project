"""把全量语料 data/cmrc_docs/cmrc_full_*.md 灌入全部索引（一次性，复用现有管线）。

用法: uv run python scripts/ingest_cmrc_full.py [--skip-graph]

复现链（陌生人 clone 后，按序）:
  1. uv run python scripts/build_cmrc_eval.py        # 生成 162 篇 cmrc_full_*.md + 300 题 eval
                                                      # （注意：脚本名虽叫 build_cmrc_eval，
                                                      #  实产出 full 全量集，非旧 31 题版）
  2. uv run python scripts/ingest_cmrc_full.py       # 本脚本：灌 chroma/summary/parent-child/BM25 + 图谱
  3. uv run python scripts/rebuild_graph_fast.py     # 零 LLM 共现图重置（秒级；rebuild_graph_typed.py 为 LLM 类型化可选路径）
  4. uv run python run_retrieval_eval.py             # 检索评估（零 LLM）
  5. uv run python run_e2e_eval.py --mode full       # 端到端三层评估

流程（与 API ingest 同源，零新架构）:
  逐文件 ingest_file(recursive)：smart_chunk → 明细索引(chroma) → L1 摘要(每文件 LLM)
  → F6a 上下文增强(每块 LLM) → BM25 增量 → 知识图谱增量构建。
chroma 持久化于 data/chroma_db，灌入即生效。

幂等：重复运行不会重复灌同一段（按 chunk_id），图谱增量按 processed_ids 跳过已处理块。
图谱降级：构建失败不影响 dense/sparse/parent_child/summary 检索（仅 graph 通道）。
"""
import argparse
import asyncio
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from app.generation.chain import RAGChain
from app.ingestion.service import ingest_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-graph", action="store_true", help="跳过图谱增量构建")
    ap.add_argument("--pattern", default="data/cmrc_docs/cmrc_full_*.md")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"未找到匹配文件: {args.pattern}（请先跑 scripts/build_cmrc_eval.py 生成语料）")
        return

    chain = RAGChain()
    before = len(chain.indexer.get_all_chunks())
    print(f"灌入前 chroma 分块数: {before}；待灌文件 {len(files)} 个")

    total_chunks = 0
    for i, path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 灌入 {path} …")
        docs, chunks = ingest_file(chain, path, chunk_strategy="recursive")
        total_chunks += len(chunks)
        print(f"    → {len(docs)} 文档 / {len(chunks)} 分块")

    after = len(chain.indexer.get_all_chunks())
    print(f"\n✅ 明细索引完成：新增 {after - before} 分块（{before} → {after}），本次共切 {total_chunks} 块")

    if args.skip_graph:
        print("⏭️  跳过图谱（--skip-graph，需单独跑 rebuild_graph_fast.py 零 LLM 重建）")
        return

    print("\n构建知识图谱（增量，仅新块）…")
    try:
        from app.ingestion.graph_extractor import get_graph_builder
        builder = get_graph_builder()
        all_chunks = chain.indexer.get_all_chunks()
        stats = asyncio.run(
            builder.build_from_documents_async(all_chunks, incremental=True)
        )
        print(f"✅ 图谱增量构建完成: {stats}")
        print(f"   图谱规模: 节点 {builder.graph.number_of_nodes()} / "
              f"边 {builder.graph.number_of_edges()}")
    except Exception as e:
        logger.warning(f"图谱构建失败（不影响 dense/sparse/parent_child/summary 检索）: {e}")


if __name__ == "__main__":
    main()
