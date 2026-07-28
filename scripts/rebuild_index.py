"""干净重建全部索引：清 chroma/graph/bm25 → 重灌规范语料（sample_docs + cmrc_docs）。

用法: uv run python scripts/rebuild_index.py [--clean] [--skip-contextual] [--skip-graph]
          [--dirs data/sample_docs data/cmrc_docs]

为什么需要:
  历史索引由 API 上传灌入，source 元数据被临时文件名污染（21 个 tmp*.md），且语料仅
  ~204 块、检索 hit_rate 无区分度。本脚本从规范语料（真文件名）一次性干净重建，消除
  污染并把检索语料扩到 CMRC2018 全量（~3700 块）。

流程（与 API ingest 同源，零新架构）:
  1. --clean: 删除 chroma_db / knowledge_graph(.meta).json / bm25_index.pkl（全部可再生）。
  2. 逐文件 ingest_file(recursive)：smart_chunk → 明细(chroma) → L1 摘要(LLM/文件)
     → F6a 上下文增强(LLM/块) → Parent-Child → BM25 增量。
  3. 知识图谱构建（fresh meta → 处理全部块，并发 + 检查点自动保存）。

幂等：--clean 保证重复运行总从空索引开始，不会重复灌入。
"""

import argparse
import asyncio
import glob
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from config import get_settings


def clean_stores():
    """删除所有可再生的索引产物（在创建 RAGChain / chroma 客户端之前调用）。"""
    settings = get_settings()
    chroma_dir = settings.chroma_persist_dir
    bm25 = os.path.join(os.path.dirname(chroma_dir), "bm25_index.pkl")

    from app.ingestion.graph_extractor import get_graph_builder
    graph_path = str(get_graph_builder().persist_path)
    graph_meta = graph_path + ".meta.json" if not graph_path.endswith(".json") else \
        graph_path.replace(".json", ".meta.json")

    targets = [chroma_dir, bm25, graph_path, graph_meta]
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t)
            logger.info(f"已删除目录: {t}")
        elif os.path.exists(t):
            os.remove(t)
            logger.info(f"已删除文件: {t}")


def collect_files(dirs):
    exts = (".md", ".markdown", ".txt", ".pdf")
    files = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if os.path.isfile(f) and f.lower().endswith(exts):
                files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="先清空 chroma/graph/bm25（幂等重建）")
    ap.add_argument("--skip-contextual", action="store_true", help="跳过 F6a 上下文增强（省 N 次 LLM）")
    ap.add_argument("--skip-graph", action="store_true", help="跳过知识图谱构建")
    ap.add_argument("--dirs", nargs="+", default=["data/sample_docs", "data/cmrc_docs"])
    args = ap.parse_args()

    if args.clean:
        logger.info("== 清空索引产物 ==")
        clean_stores()

    from app.generation.chain import RAGChain
    from app.ingestion.service import ingest_file

    if args.skip_contextual:
        # get_settings 有缓存，运行期直接改单例属性即可让 _index_contextual 读到 False
        get_settings().use_contextual_chunks = False

    files = collect_files(args.dirs)
    logger.info(f"== 待灌文件 {len(files)} 个（dirs={args.dirs}）==")

    chain = RAGChain()
    total_chunks = 0
    failed = []
    for i, path in enumerate(files, 1):
        try:
            docs, chunks = ingest_file(chain, path, chunk_strategy="recursive")
            total_chunks += len(chunks)
            if i % 10 == 0 or i == len(files):
                logger.info(f"[{i}/{len(files)}] {os.path.basename(path)} → {len(chunks)} 块 "
                            f"(累计 {total_chunks})")
        except Exception as e:
            logger.warning(f"[{i}/{len(files)}] 灌入失败 {path}: {e}")
            failed.append(path)

    all_chunks = chain.indexer.get_all_chunks()
    logger.info(f"== 明细索引完成：chroma 共 {len(all_chunks)} 块（本次切 {total_chunks}），失败 {len(failed)} ==")

    if args.skip_graph:
        logger.info("⏭️  跳过图谱（--skip-graph）")
        return

    logger.info("== 构建知识图谱（full，并发 + 检查点）==")
    try:
        from app.ingestion.graph_extractor import get_graph_builder
        builder = get_graph_builder()
        stats = asyncio.run(builder.build_from_documents_async(all_chunks, incremental=True))
        logger.info(f"✅ 图谱完成: {stats}；节点 {builder.graph.number_of_nodes()} / "
                    f"边 {builder.graph.number_of_edges()}")
    except Exception as e:
        logger.warning(f"图谱构建失败（不影响 dense/sparse/parent_child/summary/contextual）: {e}")


if __name__ == "__main__":
    main()
