"""全量重建知识图谱：清空旧 jieba 共现图 → 升级 LLM 路径（类型化三元组+chunk溯源）重抽全部分块"""
import asyncio

from app.generation.chain import RAGChain
from app.ingestion.graph_extractor import get_graph_builder

chain = RAGChain()
chunks = chain.indexer.get_all_chunks()
print(f"total chunks: {len(chunks)}")

builder = get_graph_builder()
builder.clear()  # 清空旧图 + meta（增量标记），强制全量重抽

stats = asyncio.run(builder.build_from_documents_async(chunks, incremental=False))
print("build stats:", stats)

# 质量快照：关系 top10 + 节点类型分布 + chunk 溯源覆盖率
rels, types, with_chunk = {}, {}, 0
for h, t, d in builder.graph.edges(data=True):
    rels[d.get("relation", "?")] = rels.get(d.get("relation", "?"), 0) + 1
    if d.get("chunk_id"):
        with_chunk += 1
for n, d in builder.graph.nodes(data=True):
    types[d.get("type", "?")] = types.get(d.get("type", "?"), 0) + 1
total_edges = builder.graph.number_of_edges()
print("top relations:", sorted(rels.items(), key=lambda x: -x[1])[:10])
print("node types:", types)
print(f"edges with chunk_id: {with_chunk}/{total_edges}")
