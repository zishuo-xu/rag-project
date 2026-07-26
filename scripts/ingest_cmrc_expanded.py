"""把扩充语料 data/cmrc_docs/cmrc_expanded_*.md 灌入全部索引（一次性，复用现有管线）。

用法: uv run python scripts/ingest_cmrc_expanded.py [--skip-graph]

流程（与 API ingest 同源，零新架构）:
  1. 逐文件 ingest_file(recursive)：smart_chunk → 明细索引(chroma) → L1 摘要(LLM/文件)
     → F6a 上下文增强(LLM/块) → BM25 增量。
  2. 知识图谱增量构建：build_from_documents_async(incremental=True) 仅抽新增块
     （已处理块按 processed_ids 跳过，不重抽旧语料）。

说明: chroma 持久化于 data/chroma_db，灌入即生效；BM25 在 eval 脚本里会再从
get_all_chunks() 重建，故此处增量即可。contextual/summary/graph 为索引时一次性 LLM。
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
    ap.add_argument("--pattern", default="data/cmrc_docs/cmrc_expanded_*.md")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"未找到匹配文件: {args.pattern}")
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
    print(f"\n✅ 明细索引完成：新增 {after - before} 分块（{before} → {after}），"
          f"本次共切 {total_chunks} 块")

    if args.skip_graph:
        print("⏭️  跳过图谱（--skip-graph）")
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
